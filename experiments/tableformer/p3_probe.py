"""Compare raw stock/candidate outputs on the pinned TRAIN-internal P3 crops."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

from assets import BASE_SHA, WEIGHT, digest, stock_assets
from baseline import offline, read_asset
from p3 import capture_contract, check_schedule


def raw_probe(config, stock, candidate, rows, root):
    import numpy as np
    from PIL import Image
    import torch
    from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    outputs = []
    for path in (stock, candidate):
        cfg = config | {"model": config["model"] | {"save_dir": str(path.parent.resolve())}}
        predictor = TFPredictor(cfg, device="cpu", num_threads=1)
        model = predictor.get_model().eval()
        arm = []
        for row in rows:
            with Image.open(BytesIO(read_asset(root, row["image"]))) as image:
                rgb = np.asarray(image.convert("RGB"))
            with torch.no_grad():
                tags, cls, boxes = model.predict(
                    predictor._prepare_image(rgb), cfg["predict"]["max_steps"], cfg["predict"]["beam_size"]
                )
            arm.append({"tags": list(tags), "classes": cls.tolist() if cls is not None else [],
                        "boxes": boxes.tolist() if boxes is not None else []})
        outputs.append(arm)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "models", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    args = parser.parse_args()
    sys.addaudithook(offline)
    raw = args.manifest.read_bytes()
    if sha256(raw).hexdigest() != args.manifest_sha256:
        raise ValueError("Probe TRAIN manifest differs")
    doc = json.loads(raw)
    if doc["split"] != "train" or capture_contract(doc)["mode"] != "proposal_crops":
        raise ValueError("Raw probes require the exact TRAIN proposal packet")
    check_schedule(doc)
    index = {r["id"]: r for r in doc["examples"]}
    rows = [index[name] for name in doc["p3_schedule"]["probe_ids"]]
    if not all(r["split"] == "train" for r in rows):
        raise ValueError("Probe roster leaves TRAIN")
    config, stock = stock_assets(args.models)
    if (args.candidate.name != WEIGHT or digest(args.candidate) != args.candidate_sha256
            or args.candidate_sha256 == BASE_SHA):
        raise ValueError("Expected the distinct exact P3 terminal")
    a, b = raw_probe(json.loads(config.read_bytes()), stock, args.candidate, rows, args.manifest.parent)
    comparisons = [{"id": row["id"], "same": stock_raw == candidate_raw,
                    "same_structure": stock_raw["tags"] == candidate_raw["tags"],
                    "same_boxes": stock_raw["boxes"] == candidate_raw["boxes"],
                    "stock": stock_raw, "candidate": candidate_raw}
                   for row, stock_raw, candidate_raw in zip(rows, a, b, strict=True)]
    result = {"split": "train", "manifest_sha256": args.manifest_sha256, "stock_sha256": BASE_SHA,
              "candidate_sha256": args.candidate_sha256, "rows": comparisons,
              "changed": sum(not x["same"] for x in comparisons),
              "changed_structure": sum(not x["same_structure"] for x in comparisons),
              "changed_boxes": sum(not x["same_boxes"] for x in comparisons)}
    with args.output.open("x") as out:
        json.dump(result, out, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
