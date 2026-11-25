"""Fine-tune the pinned TableFormer ACCURATE model on declared TRAIN fragments."""

import argparse
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import sys

from assets import BASE_REVISION, BASE_SHA, CONFIG_SHA, UPSTREAM, WEIGHT, digest, stock_assets
from baseline import offline, read_asset
from targets import otsl

ABI = {"docling-ibm-models": "4.0.4", "torch": "2.15.0", "torchvision": "0.29.2",
       "safetensors": "0.8.0", "pillow": "13.0.0", "numpy": "2.5.5",
       "opencv-python-headless": "5.1.0.95"}


def load_manifest(args):
    root = args.root.resolve(strict=True)
    raw = args.manifest.read_bytes()
    if sha256(raw).hexdigest() != args.manifest_sha256:
        raise ValueError("TRAIN manifest SHA-256 differs")
    clean_raw = args.clean.read_bytes()
    if sha256(clean_raw).hexdigest() != args.clean_sha256:
        raise ValueError("signed CLEAN receipt SHA-256 differs")
    clean = json.loads(clean_raw)
    manifest = json.loads(raw)
    if (manifest["schema"] != "heocr.tableformer_train.v1" or manifest["split"] != "train"
            or clean["split"] != "train" or clean["verdict"] != "TRAIN CLEAN"):
        raise ValueError("expected a TRAIN manifest and its independent CLEAN receipt")
    if clean["files"][args.manifest.name]["sha256"] != args.manifest_sha256:
        raise ValueError("CLEAN receipt does not bind the exact physical-fragment manifest")
    p3 = getattr(args, "recipe", "head") == "p3"
    if p3:
        from p3 import bbox_targets, capture_contract, checked_crop

        capture = capture_contract(manifest)
        if capture.get("mode") == "direct_phone" and clean.get("remote_transfer") is not True:
            raise ValueError("direct phone P3 requires exact TRAIN and remote-transfer approval")
    examples = manifest["examples"]
    if not examples or len({x["id"] for x in examples}) != len(examples):
        raise ValueError("TRAIN manifest must have unique fragment IDs")
    if p3 and len(examples) > 256:
        raise ValueError("P3 pilot allows at most 256 TRAIN fragments")
    seen = set()
    checked = []
    for item in examples:
        if item["split"] != "train" or not item["source_id"] or item.get("dev_only") or item.get("claude_drafted"):
            raise ValueError("example outside TRAIN scope")
        raw_image = read_asset(root, item["image"])
        raw_gold = read_asset(root, item["gold"])
        identity = (item["image"]["sha256"], item["gold"]["sha256"])
        if identity in seen:
            raise ValueError("duplicate physical fragment")
        seen.add(identity)
        table = json.loads(raw_gold)
        if p3:
            from PIL import Image

            with Image.open(BytesIO(raw_image)) as image:
                checked_crop(item, capture, image.size)
                tags, boxes = bbox_targets(table, image.size)
            checked.append((raw_image, tags, boxes))
        else:
            checked.append((raw_image, otsl(table)))
    return checked


def invented(p3=False):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(img)
    for y in (10, 95, 185):
        draw.line((12, y, 386, y), fill="black", width=2)
    for x in (12, 200, 385):
        draw.line((x, 10, x, 185), fill="black", width=2)
    draw.text((45, 32), "Sample", fill="black")
    draw.text((260, 32), "Count", fill="black")
    draw.text((45, 124), "Blue", fill="black")
    draw.text((260, 124), "42", fill="black")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    table = {"schema": "heocr_table.v1", "rows": 2, "cols": 2, "header_rows": 1, "direction": "rtl",
             "cells": [{"r": i // 2, "c": i % 2, "rowspan": 1, "colspan": 1, "text": text}
                       for i, text in enumerate(("Count", "Sample", "42", "Blue"))]}
    if p3:
        from p3 import bbox_targets

        for cell in table["cells"]:
            x = (200, 12)[cell["c"]]
            y = (10, 95)[cell["r"]]
            cell["bbox"] = [x, y, x + 185, y + 85]
        return [(buffer.getvalue(), *bbox_targets(table, img.size))]
    return [(buffer.getvalue(), otsl(table))]


def memory_for(model, enc):
    t = model._tag_transformer
    filtered = t._input_filter(enc.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    positions = filtered.flatten(1, 2).permute(1, 0, 2).contiguous()
    return t._encoder(positions)


def teacher_hidden(transformer, memory, tokens):
    """Vectorized causal equivalent of the upstream last-token cached decoder."""
    import torch

    hidden = transformer._positional_encoding(transformer._embedding(tokens))
    mask = torch.ones((len(tokens), len(tokens)), dtype=torch.bool, device=tokens.device).triu(1)
    for layer in transformer._decoder.layers:
        hidden = torch.nn.TransformerDecoderLayer.forward(layer, hidden, memory, tgt_mask=mask)
    return hidden


def check_teacher(transformer, memory, tokens, vectorized):
    import torch

    cache = None
    embedded = transformer._positional_encoding(transformer._embedding(tokens))
    for i in range(len(tokens)):
        decoded, cache = transformer._decoder(embedded[:i + 1], memory, cache)
        torch.testing.assert_close(decoded[-1], vectorized[i], atol=3e-4, rtol=1e-4)


def p3_loss(model, teacher, enc, memory, hidden, ids, events):
    import torch
    import torch.nn.functional as F

    teacher_transformer, teacher_boxes = teacher
    positions, coordinates = zip(*events)
    selected = list(positions)
    classes, boxes = model._bbox_decoder.inference(enc, hidden[selected])
    target = boxes.new_tensor(coordinates)
    with torch.no_grad():
        reference = teacher_hidden(teacher_transformer, memory, ids[:-1])
        reference_classes, _ = teacher_boxes.inference(enc, reference[selected])
    box_loss = F.smooth_l1_loss(boxes, target, beta=0.05)
    class_loss = F.kl_div(classes.log_softmax(-1), reference_classes.softmax(-1), reduction="batchmean")
    return box_loss, class_loss


def run(args):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    sys.addaudithook(offline)
    real = {package: version(package) for package in ABI}
    if real != ABI:
        raise ValueError(f"training ABI differs: {real}")
    import numpy as np
    from PIL import Image
    import torch
    from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
    from safetensors.torch import save_file

    config_file, weight_file = stock_assets(args.models)
    config = json.loads(config_file.read_bytes())
    config["model"]["save_dir"] = str(weight_file.parent)
    vocab = config["dataset_wordmap"]["word_map_tag"]
    p3 = args.recipe == "p3"
    if p3:
        from p3 import checkpoint_diff, trainable_groups

    if args.smoke:
        examples = invented(p3)
    else:
        examples = load_manifest(args)
    for _, tags, *_ in examples:
        if not all(tag in vocab for tag in tags) or len(tags) + 1 > config["predict"]["max_steps"]:
            raise ValueError("target does not fit the pinned model")
    randomizer = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    predictor = TFPredictor(config, device=args.device, num_threads=1)
    model = predictor.get_model()
    model.eval()
    model.requires_grad_(False)
    transformer = model._tag_transformer
    head = transformer._fc
    groups = trainable_groups(model) if p3 else {"token_head": ("_tag_transformer._fc", head)}
    teacher = (deepcopy(transformer), deepcopy(model._bbox_decoder)) if p3 else None
    parameters = [parameter for _, module in groups.values()
                  for parameter in module.requires_grad_(True).parameters()]
    before = {} if p3 else {key: p.detach().clone() for key, p in head.named_parameters()}
    opt = torch.optim.AdamW(parameters, lr=args.lr, foreach=False, fused=False,
                            weight_decay=0.0 if p3 else 0.01)
    steps, losses = 0, []
    p3_losses = []
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        randomizer.shuffle(order)
        total = box_total = class_total = 0.0
        for idx in order:
            raw, tokens, *geometry = examples[idx]
            with Image.open(BytesIO(raw)) as img:
                # Match Docling's page.get_image() array passed to TFPredictor.
                rgb = np.asarray(img.convert("RGB"))
            tokens = ["<start>", *tokens, "<end>"]
            ids = torch.tensor([vocab[tag] for tag in tokens], device=args.device).view(-1, 1)
            with torch.no_grad():
                enc = model._encoder(predictor._prepare_image(rgb))
                memory = memory_for(model, enc)
                if not p3:
                    hidden = teacher_hidden(transformer, memory, ids[:-1])
                if args.smoke and steps == 0:
                    check_teacher(transformer, memory, ids[:-1],
                                  teacher_hidden(transformer, memory, ids[:-1]) if p3 else hidden)
            if p3:
                hidden = teacher_hidden(transformer, memory, ids[:-1])
            logits = head(hidden).squeeze(1)
            loss = torch.nn.functional.cross_entropy(logits, ids[1:, 0])
            if p3:
                box, cls = p3_loss(model, teacher, enc, memory, hidden, ids, geometry[0])
                loss = loss + 2.0 * box + 0.1 * cls
                box_total += box.item()
                class_total += cls.item()
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if p3:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            opt.step()
            steps += 1
            total += loss.item()
        losses.append(total / len(examples))
        if p3:
            p3_losses.append({"box": box_total / len(examples), "class_kl": class_total / len(examples)})
    changed = [] if p3 else sorted(key for key, p in head.named_parameters()
                                    if not torch.equal(p.detach(), before[key]))
    if not p3 and not changed:
        raise ValueError("no token-head parameters changed")
    state = model.state_dict()
    args.output.mkdir(parents=False, exist_ok=False)
    candidate = args.output / WEIGHT
    pending = candidate.with_suffix(".tmp")
    save_file({k: value.detach().cpu().contiguous() for k, value in sorted(state.items())}, pending)
    difference = checkpoint_diff(weight_file, pending, groups) if p3 else None
    pending.replace(candidate)
    record = {"schema": "heocr.tableformer_train_receipt.v1", "smoke": args.smoke, "upstream_commit": UPSTREAM,
              "base_revision": BASE_REVISION, "base_sha256": BASE_SHA, "config_sha256": CONFIG_SHA,
              "input_manifest_sha256": args.manifest_sha256, "clean_sha256": args.clean_sha256,
              "harness_files": {name: digest(Path(__file__).with_name(name)) for name in
                                ("assets.py", "baseline.py", "targets.py", "train.py")},
              "candidate_sha256": digest(candidate), "versions": ABI, "python": sys.version.split()[0],
              "device": args.device, "cuda": torch.version.cuda, "seed": args.seed, "learning_rate": args.lr,
              "examples": len(examples), "steps": steps, "epoch_mean_loss": losses,
              "updated": [f"_tag_transformer._fc.{key}" for key in changed]}
    if p3:
        record.update(recipe="p3", box_losses=p3_losses, weight_diff=difference)
        record["updated"] = [key for key, row in difference["trainable_tensors"].items()
                             if row["changed_elements"]]
        record["harness_files"]["p3.py"] = digest(Path(__file__).with_name("p3.py"))
    (args.output / "receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("root", "manifest", "clean"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--clean-sha256")
    parser.add_argument("--smoke", action="store_true", help="train on one invented table embedded in this script")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--recipe", choices=("head", "p3"), default="head")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()
    pins = (args.root, args.manifest, args.clean, args.manifest_sha256, args.clean_sha256)
    if (args.smoke and any(pins)) or (not args.smoke and not all(pins)):
        parser.error("pass either --smoke or a manifest and CLEAN receipt with both digests")
    if args.epochs is None:
        args.epochs = (2 if args.smoke else 3) if args.recipe == "p3" else 1
    if args.lr is None:
        args.lr = 1e-5 if args.recipe == "p3" else 2e-5
    if args.recipe == "p3" and not (2 <= args.epochs <= 3 and 0 < args.lr <= 2e-5):
        parser.error("P3 expects two or three epochs and at most 2e-5 LR")
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs and finite learning rate must be positive")
    run(args)


if __name__ == "__main__":
    main()
