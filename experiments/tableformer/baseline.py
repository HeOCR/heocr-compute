"""Local TableFormer DEV baseline. Write scores and pins, never cell text."""

import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def offline(event, args):
    if event == "socket.connect" and not isinstance(args[1], (str, bytes)):
        raise RuntimeError("DEV table baseline is local")
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        raise RuntimeError("DEV table baseline is local")


def read_asset(root, asset):
    member = Path(asset["path"])
    if member.is_absolute():
        raise ValueError("asset must have a relative path")
    path = (root / member).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("asset path leaves the declared packet root")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != asset["sha256"]:
        raise ValueError(f"asset SHA-256 differs: {path.name}")
    return raw


def module_pin(module, expected):
    root = Path(module.__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"])
    if head != expected or dirty:
        raise ValueError(f"unexpected source revision for {module.__name__}")


def single_page(raw, ordinal):
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(raw) as source, pdfium.PdfDocument.new() as target:
        if not 1 <= ordinal <= len(source):
            raise ValueError("PDF page number out of range")
        target.import_pages(source, [ordinal - 1])
        out = BytesIO()
        target.save(out)
        return out.getvalue()


def image_pdf(raw):
    import pypdfium2 as pdfium
    from PIL import Image

    with Image.open(BytesIO(raw)) as image, pdfium.PdfDocument.new() as pdf:
        scale = 72 / image.info.get("dpi", (300, 300))[0]
        width, height = image.width * scale, image.height * scale
        with pdf.new_page(width, height) as page, pdfium.PdfBitmap.from_pil(image.convert("RGB")) as bitmap:
            obj = pdfium.PdfImage.new(pdf)
            obj.set_bitmap(bitmap)
            obj.set_matrix(pdfium.PdfMatrix(width, 0, 0, height, 0, 0))
            page.insert_obj(obj)
            page.gen_content()
        out = BytesIO()
        pdf.save(out)
        return out.getvalue()


def docling_tables(docling_doc, rtl):
    for t in docling_doc.tables:
        data = t.data
        cells = {}
        for cell in data.table_cells:
            r, c = cell.start_row_offset_idx, cell.start_col_offset_idx
            rs = cell.end_row_offset_idx - r
            cs = cell.end_col_offset_idx - c
            if rs < 1 or cs < 1:
                continue
            key = (r, data.num_cols - c - cs if rtl else c)
            entry = {"r": key[0], "c": key[1], "rowspan": rs, "colspan": cs, "text": "",
                     "_header": cell.column_header}
            if key in cells and cells[key] != entry:
                cells[key]["_conflict"] = True
            else:
                cells[key] = entry
        headers = 0
        for r in range(data.num_rows):
            anchors = [c for c in cells.values() if c["r"] <= r < c["r"] + c["rowspan"]]
            if not anchors or not all(c["_header"] for c in anchors):
                break
            headers += 1
        yield {"schema": "heocr_table.v1", "rows": data.num_rows, "cols": data.num_cols,
               "header_rows": headers, "cells": list(cells.values())}


def heocr_tables(doc):
    from heocr.table_export import table_grid

    for block in doc.blocks:
        if block.kind != "table" or not block.cells:
            continue
        grid, merges = table_grid(block)
        spans = {(r, c): (rs, cs) for r, c, rs, cs in merges}
        covered = {(i, j) for r, c, rs, cs in merges for i in range(r, r + rs)
                   for j in range(c, c + cs) if (i, j) != (r, c)}
        cells = []
        for r, row in enumerate(grid):
            for c, text in enumerate(row):
                if (r, c) not in covered:
                    rs, cs = spans.get((r, c), (1, 1))
                    cells.append({"r": r, "c": c, "rowspan": rs, "colspan": cs, "text": text})
        # Current HeOCR Block/Cell records carry spans and reading order,
        # with no leading-header designation. Preserve this limitation.
        yield {"schema": "heocr_table.v1", "rows": len(grid), "cols": len(grid[0]),
               "header_rows": 0, "cells": cells}


def structure_only(table):
    return {**table, "cells": [{**cell, "text": ""} for cell in table["cells"]]}


def score(gold, predictions, structures=False):
    from heocr_bench.tables import load_table, score_document

    valid = []
    rejected = 0
    for table in predictions:
        try:
            if table["rows"] < 1 or table["cols"] < 1 or any(c.get("_conflict") for c in table["cells"]):
                raise ValueError("invalid predicted grid")
            load_table(table)
            valid.append(table)
        except (ValueError, KeyError, TypeError):
            rejected += 1
    if structures:
        gold = [structure_only(table) for table in gold]
        valid = [structure_only(table) for table in valid]
    result = score_document(gold, valid)
    result["invalid_predicted"] = rejected
    return result


def run(args):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      DOCLING_DEVICE="cpu", OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1")
    sys.addaudithook(offline)
    import heocr
    import heocr_bench
    import torch
    from heocr_bench.tables import pool
    from heocr.tableformer import TableFormerSession
    from assets import CONFIG_SHA, LAYOUT_CONFIGS, candidate_session, digest, stock_assets, verify_layout_configs

    module_pin(heocr, args.heocr_commit)
    module_pin(heocr_bench, args.bench_commit)
    if version("docling") != "2.152.0" or version("docling-ibm-models") != "4.0.4":
        raise ValueError("unexpected Docling runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.root.resolve(strict=True)
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    if sha256(raw).hexdigest() != args.manifest_sha256:
        raise ValueError("DEV manifest SHA-256 differs")
    if manifest["schema"] != "heocr.tableformer_dev.v1" or manifest["split"] != "dev":
        raise ValueError("expected an authorized DEV page manifest")
    pages = manifest["pages"]
    if not pages or len({page["id"] for page in pages}) != len(pages):
        raise ValueError("DEV page manifest must contain unique page IDs")

    versions = {name: version(name) for name in ("docling", "docling-ibm-models", "torch", "pypdfium2")}
    verify_layout_configs(args.models)
    scores = defaultdict(lambda: defaultdict(list))
    page_rows = []
    with TemporaryDirectory() as temp:
        config, _ = stock_assets(args.models)
        if args.candidate:
            session = candidate_session(args.models, config, args.candidate, args.candidate_sha256, Path(temp))
        else:
            session = TableFormerSession(args.models)
        for page in pages:
            expected = [json.loads(read_asset(root, a)) for a in page["gold"]]
            if not expected or any(table["schema"] != "heocr_table.v1" for table in expected):
                raise ValueError("expected exact table structure for each DEV page")
            if page["direction"] not in {"rtl", "ltr"}:
                raise ValueError("expected a declared page reading direction")
            image_or_pdf = read_asset(root, page["source"])
            if page["capture"] not in {"scanner", "issuer-render", "phone"}:
                raise ValueError("expected a declared DEV capture")
            is_image = page["page"] is None
            payload = image_pdf(image_or_pdf) if is_image else single_page(image_or_pdf, page["page"])
            stock = session._convert(payload)
            raw_result = score(expected, docling_tables(stock, page["direction"] == "rtl"), structures=True)
            integrated = heocr.extract(image_or_pdf if is_image else payload, tables="tableformer",
                                       tables_models=session, ocr_model="heocr_modern_v1", tesseract=str(args.tesseract),
                                       force_ocr=page["capture"] == "scanner", check_numbers=False, convert=False)
            bad = {"table_proposer_failed", "ocr_call_failed", "bundled_model_unavailable", "routed_model_unavailable"}
            if set(integrated.flags) & bad:
                raise RuntimeError(f"integrated DEV read failed: {page['id']}")
            result = score(expected, heocr_tables(integrated))
            for lane, lane_row in (("stock_structure", raw_result), ("heocr_tableformer", result)):
                scores[lane][page["cluster"]].append(lane_row)
            page_rows.append({"id": page["id"], "gold": len(expected),
                              "source_sha256": page["source"]["sha256"], "source_page": page["page"],
                              "ocr_components": {k:v for k,v in integrated.components.items() if k.startswith("tesseract:")},
                              "stock_structure": raw_result, "heocr_tableformer": result})
    lanes = {}
    for lane, clusters in scores.items():
        grouped = {key: {**{k: sum(r[k] for r in value) for k in ("matched", "unmatched_predictions", "invalid_predicted")},
                         "tables": [row for d in value for row in d["tables"]]}
                   for key, value in clusters.items()}
        lanes[lane] = pool(grouped, seed=0, draws=2000)
        lanes[lane]["invalid_predicted"] = sum(v["invalid_predicted"] for v in grouped.values())
        if lane == "stock_structure":
            # This lane deliberately uses blank cells for structural matching.
            lanes[lane].pop("teds")
            lanes[lane].pop("grits_con")
            lanes[lane].pop("exact")
    return {"schema": "heocr.tableformer_dev_baseline.v1", "scope": "DEV selection only", "versions": versions,
            "heocr_commit": args.heocr_commit, "bench_commit": args.bench_commit,
            "manifest_sha256": args.manifest_sha256, "table_models": session._record["weights_sha256"],
            "table_config_sha256": CONFIG_SHA, "layout_config_sha256": LAYOUT_CONFIGS,
            "tesseract_sha256": digest(args.tesseract), "pooled": lanes, "pages": page_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "models", "manifest", "output", "tesseract"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("manifest-sha256", "heocr-commit", "bench-commit"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-sha256")
    args = parser.parse_args()
    if (args.candidate is None) != (args.candidate_sha256 is None):
        parser.error("candidate weights and SHA-256 must be supplied together")
    result = run(args)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
