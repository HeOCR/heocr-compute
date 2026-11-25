"""Initialize the frozen models and run their full adapters on an invented raster."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from fetch_models import verify
from grids import tesseract_reader


if __name__ == "__main__":
    lane = sys.argv[1]
    if lane == "tatr":
        from tatr import predictor
    elif lane == "paddle":
        from ppstructure import predictor
    else:
        raise ValueError("unknown comparator lane")
    models = Path("/opt/models")
    verify(models, json.loads(Path(__file__).with_name(f"{lane}-models.json").read_bytes()))
    infer = predictor(models, tesseract_reader(Path("/opt/tessdata")))
    with TemporaryDirectory() as temp:
        path = Path(temp) / "invented.png"
        image = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(image)
        for x in (100, 300, 500):
            draw.line((x, 100, x, 300), fill="black", width=2)
        for y in (100, 200, 300):
            draw.line((100, y, 500, y), fill="black", width=2)
        draw.text((160, 135), "001", fill="black")
        image.save(path, dpi=(200, 200))
        tables, issues = infer(image, 1, path)
        print(f"Smoke completed. Predictions: {len(tables)}. Adapter issues: {len(issues)}.")
