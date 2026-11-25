"""Fine-tune only the pinned TableFormer ACCURATE OTSL token head."""

import argparse
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
    examples = manifest["examples"]
    if not examples or len({x["id"] for x in examples}) != len(examples):
        raise ValueError("TRAIN manifest must have unique fragment IDs")
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
        tags = otsl(json.loads(raw_gold))
        checked.append((raw_image, tags))
    return checked


def invented():
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
    return [(buffer.getvalue(), otsl(table))]


def memory_for(model, image):
    t = model._tag_transformer
    enc = model._encoder(image)
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
    if args.smoke:
        examples = invented()
    else:
        examples = load_manifest(args)
    for _, tags in examples:
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
    head = transformer._fc.requires_grad_(True)
    before = {key: p.detach().clone() for key, p in head.named_parameters()}
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, foreach=False, fused=False)
    steps, losses = 0, []
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        randomizer.shuffle(order)
        total = 0.0
        for idx in order:
            raw, tokens = examples[idx]
            with Image.open(BytesIO(raw)) as img:
                # Match Docling's page.get_image() array passed to TFPredictor.
                rgb = np.asarray(img.convert("RGB"))
            tokens = ["<start>", *tokens, "<end>"]
            ids = torch.tensor([vocab[tag] for tag in tokens], device=args.device).view(-1, 1)
            with torch.no_grad():
                memory = memory_for(model, predictor._prepare_image(rgb))
                hidden = teacher_hidden(transformer, memory, ids[:-1])
                if args.smoke and steps == 0:
                    check_teacher(transformer, memory, ids[:-1], hidden)
            logits = head(hidden).squeeze(1)
            loss = torch.nn.functional.cross_entropy(logits, ids[1:, 0])
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            total += loss.item()
        losses.append(total / len(examples))
    changed = sorted(key for key, p in head.named_parameters() if not torch.equal(p.detach(), before[key]))
    if not changed:
        raise ValueError("no token-head parameters changed")
    state = model.state_dict()
    args.output.mkdir(parents=False, exist_ok=False)
    candidate = args.output / WEIGHT
    pending = candidate.with_suffix(".tmp")
    save_file({k: value.detach().cpu().contiguous() for k, value in sorted(state.items())}, pending)
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    pins = (args.root, args.manifest, args.clean, args.manifest_sha256, args.clean_sha256)
    if (args.smoke and any(pins)) or (not args.smoke and not all(pins)):
        parser.error("pass either --smoke or a manifest and CLEAN receipt with both digests")
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs and finite learning rate must be positive")
    run(args)


if __name__ == "__main__":
    main()
