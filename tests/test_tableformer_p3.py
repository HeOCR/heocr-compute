"""Offline dry runs for geometry, the saved-weight gate, and the one-job compiler."""

import hashlib
from importlib.util import find_spec
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

HARNESS = Path(__file__).resolve().parents[1] / "experiments/tableformer"
sys.path.insert(0, str(HARNESS))
try:
    from assets import BASE_SHA, CONFIG, CONFIG_SHA, MODEL_SUBDIR, WEIGHT, digest
    from p3 import bbox_targets, capture_contract, checked_crop, checkpoint_diff
    from p3_job import CODE, compile_job
    from train import load_manifest
finally:
    sys.path.remove(str(HARNESS))

REPO = "HeOCR/fixture"


class P3Geometry(unittest.TestCase):
    def test_visual_rtl_span_supervises_docling_boundary_states(self):
        table = {
            "schema": "heocr_table.v1", "direction": "rtl", "rows": 1, "cols": 3,
            "cells": [
                {"r": 0, "c": 0, "rowspan": 1, "colspan": 2, "text": "span", "bbox": [50, 0, 100, 40]},
                {"r": 0, "c": 2, "rowspan": 1, "colspan": 1, "text": "left", "bbox": [0, 0, 50, 40]},
            ],
        }
        tags, events = bbox_targets(table, (100, 40))
        self.assertEqual(tags, ["fcel", "fcel", "lcel", "nl"])
        self.assertEqual(events, [
            (1, (0.25, 0.5, 0.5, 1)), (2, (0.75, 0.5, 0.5, 1)), (3, (0.75, 0.5, 0.5, 1)),
        ])
        table["cells"][0]["bbox"] = [50, 0, 101, 40]
        with self.assertRaisesRegex(ValueError, "outside"):
            bbox_targets(table, (100, 40))

    def test_rectifier_acceptance_and_geometry_are_bound(self):
        pin = "a" * 64
        route = {
            "source_page_sha256": pin, "captured_page_sha256": pin,
            "rectified_page_sha256": pin, "phone_fit_sha256": pin,
            "rectifier_sha256": pin, "rectifier_config_sha256": pin,
            "rectified_size": [150, 100], "crop_xyxy": [5, 10, 105, 50], "accepted": True,
        }
        scope = {
            "hdocsyngen_recipe_sha256": pin, "rectifier_verdict_sha256": pin,
            "rectifier_sha256": pin, "rectifier_config_sha256": pin, "scanifier_fit_sha256": [pin],
        }
        capture = capture_contract({"p3": scope})
        checked_crop({"p3": route}, capture, (100, 40))
        for override in ({"accepted": False}, {"phone_fit_sha256": "b" * 64},
                         {"rectifier_sha256": "b" * 64}, {"crop_xyxy": [0, 0, 99, 40]}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                checked_crop({"p3": {**route, **override}}, capture, (100, 40))
        with self.assertRaisesRegex(ValueError, "provenance"):
            capture_contract({"p3": {**scope, "rectifier_verdict_sha256": None}})


    def test_direct_phone_excludes_rectification_and_uses_captured_extent(self):
        pin = "a" * 64
        revision = "b" * 40
        capture = capture_contract({"p3": {
            "mode": "direct_phone", "hdocsyngen_recipe_sha256": pin,
            "scanifier_fit_sha256": [pin], "scanifier_fit_input_sha256": pin,
            "scanifier_bank_sha256": pin, "phone_route_revision": revision,
        }})
        route = {
            "mode": "direct_phone", "phone_route_revision": revision, "accepted": True,
            "source_page_sha256": pin, "captured_page_sha256": pin, "phone_fit_sha256": pin,
            "phone_warp_sha256": pin, "captured_size": [150, 100], "crop_xyxy": [5, 10, 105, 50],
        }
        checked_crop({"p3": route}, capture, (100, 40))
        for override in (
            {"phone_warp_sha256": None}, {"phone_route_revision": "c" * 40},
            {"rectified_page_sha256": pin}, {"captured_size": [90, 100]},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                checked_crop({"p3": route | override}, capture, (100, 40))


    @unittest.skipUnless(find_spec("PIL"), "requires Pillow for image geometry")
    def test_full_train_intake_rejects_prior_receipt_and_rejected_rectification(self):
        from PIL import Image

        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            def asset(name, data):
                (root / name).write_bytes(data)
                return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

            pin = "a" * 64
            capture = {
                "hdocsyngen_recipe_sha256": pin, "rectifier_verdict_sha256": pin,
                "rectifier_sha256": pin, "rectifier_config_sha256": pin, "scanifier_fit_sha256": [pin],
            }
            route = {
                "source_page_sha256": pin, "captured_page_sha256": pin, "phone_fit_sha256": pin,
                "rectified_page_sha256": pin, "rectifier_sha256": pin, "rectifier_config_sha256": pin,
                "rectified_size": [20, 20], "crop_xyxy": [1, 2, 11, 12], "accepted": True,
            }
            table = {
                "schema": "heocr_table.v1", "direction": "rtl", "rows": 1, "cols": 1,
                "cells": [{"r": 0, "c": 0, "rowspan": 1, "colspan": 1, "text": "cell", "bbox": [1, 1, 9, 9]}],
            }
            pixels = BytesIO()
            Image.new("RGB", (10, 10), "white").save(pixels, format="PNG")
            entry = {"id": "fixture", "source_id": "page", "split": "train", "p3": route,
                     "image": asset("crop.png", pixels.getvalue()), "gold": asset("gold.json", json.dumps(table).encode())}
            original = {"schema": "heocr.tableformer_train.v1", "split": "train", "p3": capture,
                        "examples": [entry]}

            def args(manifest, clean=None, **authority):
                manifest_pin = asset("manifest.json", json.dumps(manifest).encode())
                receipt = clean or {"split": "train", "verdict": "TRAIN CLEAN",
                                    "files": {"manifest.json": manifest_pin}, **authority}
                clean_pin = asset("clean.json", json.dumps(receipt).encode())
                return SimpleNamespace(
                    recipe="p3", root=root, manifest=root / "manifest.json", clean=root / "clean.json",
                    manifest_sha256=manifest_pin["sha256"], clean_sha256=clean_pin["sha256"],
                )

            approved = args(original)
            self.assertEqual(load_manifest(approved)[0][1:], (["fcel", "nl"], [(1, (0.5, 0.5, 0.8, 0.8))]))
            prior_receipt = json.loads(approved.clean.read_bytes())
            route["accepted"] = False
            mismatched = args(original, prior_receipt)
            with self.assertRaisesRegex(ValueError, "does not bind"):
                load_manifest(mismatched)
            with self.assertRaisesRegex(ValueError, "accepted phone"):
                load_manifest(args(original))


            revision = "b" * 40
            clean_route = {key: value for key, value in route.items() if not key.startswith("rectif")}
            entry["p3"] = clean_route | {
                "mode": "direct_phone", "phone_route_revision": revision, "phone_warp_sha256": pin,
                "captured_size": [20, 20], "accepted": True,
            }
            original["p3"] = {
                "mode": "direct_phone", "phone_route_revision": revision, "hdocsyngen_recipe_sha256": pin,
                "scanifier_fit_sha256": [pin], "scanifier_fit_input_sha256": pin, "scanifier_bank_sha256": pin,
            }
            with self.assertRaisesRegex(ValueError, "remote-transfer"):
                load_manifest(args(original))
            self.assertEqual(load_manifest(args(original, remote_transfer=True))[0][1],
                             ["fcel", "nl"])


class P3Job(unittest.TestCase):
    def test_compiles_a_single_pinned_job_and_rejects_other_bytes(self):
        with TemporaryDirectory() as temporary:
            transport = Path(temporary) / "transport.py"
            transport.write_text("# fixture\n")
            folder = Path("models") / MODEL_SUBDIR
            expected = {name: digest(HARNESS / name) for name in CODE}
            expected.update({
                (folder / CONFIG).as_posix(): CONFIG_SHA, (folder / WEIGHT).as_posix(): BASE_SHA,
                "train.tar": "a" * 64, "transport.py": digest(transport),
            })
            pins = [{"repo": REPO, "type": "dataset", "path": f"inputs/fixture/{name}", "target": name,
                     "revision": "a" * 40, "size": 1, "sha256": sha}
                    for name, sha in expected.items()]
            args = ("manifest.json", "b" * 64, "c" * 64, "a" * 64, transport, REPO)
            raw, approved = compile_job(pins, *args)
            job = json.loads(raw)
            self.assertEqual(approved, hashlib.sha256(raw).hexdigest())
            self.assertEqual(job["entrypoint"], "train-p3.sh")
            self.assertEqual([item["name"] for item in job["snapshots"]], [WEIGHT])
            self.assertEqual(job["output"], {"repo": REPO, "type": "dataset"})
            with self.assertRaisesRegex(ValueError, "byte pins differ"):
                compile_job([{**pins[0], "sha256": "d" * 64}, *pins[1:]], *args)


@unittest.skipUnless(find_spec("torch") and find_spec("safetensors"), "requires the TableFormer runtime")
class P3WeightDiff(unittest.TestCase):
    def test_saved_tensor_audit_detects_head_only_and_frozen_changes(self):
        import torch
        from safetensors.torch import save_file

        groups = {
            "decoder_0": ("decoder", torch.nn.Linear(1, 1)),
            "box_head": ("_bbox_decoder", torch.nn.ModuleDict({
                "_bbox_embed": torch.nn.Linear(1, 4),
                "_class_embed": torch.nn.Linear(1, 2),
            })),
            "token_head": ("classifier", torch.nn.Linear(1, 1)),
        }
        base = {prefix + "." + name: parameter.detach().clone()
                for prefix, module in groups.values() for name, parameter in module.named_parameters()}
        base["encoder.weight"] = torch.ones(1)
        with TemporaryDirectory() as temporary:
            stock = Path(temporary) / "stock.safetensors"
            output = Path(temporary) / "output.safetensors"
            save_file(base, stock)

            def write(prefixes):
                save_file({name: value + 1 if name.startswith(prefixes) else value
                           for name, value in base.items()}, output)

            write(("classifier.",))
            with self.assertRaisesRegex(ValueError, "group did not move"):
                checkpoint_diff(stock, output, groups)
            write(("decoder.", "_bbox_decoder.", "classifier.", "encoder."))
            with self.assertRaisesRegex(ValueError, "frozen tensor changed"):
                checkpoint_diff(stock, output, groups)
            write(("decoder.", "_bbox_decoder.", "classifier."))
            result = checkpoint_diff(stock, output, groups)
            self.assertEqual(result["changed_tensors_by_group"], {"decoder_0": 2, "box_head": 4, "token_head": 2})
            self.assertEqual(result["frozen_changes"], 0)
