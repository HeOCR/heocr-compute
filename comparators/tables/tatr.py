"""Run the paired Microsoft detection and v1.1-all structure models."""

from PIL import Image

from grids import Cell, pixel_box, table

# Microsoft inference.py thresholds and crop margin, pinned alongside postprocess.py.
THRESHOLD = 0.5
MARGIN = 10


def predictor(models, read_cell):
    from transformers import AutoConfig, AutoImageProcessor, TableTransformerForObjectDetection
    import postprocess
    import torch

    torch.set_num_threads(2)

    def load(name):
        root = models / name
        config = AutoConfig.from_pretrained(root, local_files_only=True)
        # The checkpoint includes the backbone. Avoid re-downloading initialization weights.
        config.use_pretrained_backbone = False
        model = TableTransformerForObjectDetection.from_pretrained(
            root, config=config, local_files_only=True, use_safetensors=True).eval()
        processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
        return model, processor

    detection = load("table-transformer-detection")
    structure = load("table-transformer-structure-recognition-v1.1-all")

    def objects(engine, image):
        model, processor = engine
        with torch.inference_mode():
            scores = model(**processor(images=image, return_tensors="pt"))
        result = processor.post_process_object_detection(
            scores, threshold=THRESHOLD, target_sizes=[(image.height, image.width)])[0]
        return [{"label": int(label), "bbox": box.tolist()}
                | {"score": float(score)}
                for label, box, score in zip(result["labels"], result["boxes"], result["scores"], strict=True)]

    def predict(image, page, source):
        tables, issues = [], []
        for proposal in objects(detection, image):
            left, top, right, bottom = proposal["bbox"]
            bbox = pixel_box((left - MARGIN, top - MARGIN, right + MARGIN, bottom + MARGIN), image.size)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                issues.append("invalid_detection_box")
                continue
            crop = image.crop(bbox)
            if detection[0].config.id2label[proposal["label"]] == "table rotated":
                crop = crop.transpose(Image.Transpose.ROTATE_270)
            predictions = objects(structure, crop)
            names = structure[0].config.id2label
            threshold = {name: THRESHOLD for name in names.values()}
            root = {"bbox": [0, 0, crop.width, crop.height], "page_num": page}
            grid, found, _ = postprocess.objects_to_cells(root, predictions, [], names, threshold)
            if not found:
                issues.append("detection_without_structure")
                continue
            cells = [Cell(min(cell["row_nums"]), min(cell["column_nums"]),
                          len(cell["row_nums"]), len(cell["column_nums"]), cell["header"],
                          read_cell(crop, cell["bbox"])) for cell in found]
            try:
                tables.append(table(cells, len(grid["rows"]), len(grid["columns"]), page, bbox))
            except ValueError:
                issues.append("invalid_predicted_grid")
        return tables, issues

    return predict
