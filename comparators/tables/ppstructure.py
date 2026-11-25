"""Run PaddleOCR's default PP-StructureV3 models with pinned asset directories."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory

from grids import paddle_table


def predictor(models, read_cell):
    import yaml
    import paddlex
    from paddleocr import PPStructureV3

    wheel = Path(paddlex.__file__).parent
    config_path, = wheel.glob("configs/pipelines/PP-StructureV3.yaml")
    config = yaml.safe_load(config_path.read_text())

    def bind_defaults(node):
        result = copy.deepcopy(node)
        for submodule in result.get("SubModules", {}).values():
            name = submodule.get("model_name")
            if name and (models / name).is_dir():
                submodule["model_dir"] = str(models / name)
        for key, pipeline in result.get("SubPipelines", {}).items():
            result["SubPipelines"][key] = bind_defaults(pipeline)
        return result

    with TemporaryDirectory(prefix="ppstructure-config-") as temp:
        pinned = Path(temp) / "PP-StructureV3.yaml"
        pinned.write_text(yaml.safe_dump(bind_defaults(config)))
        # Neither Hebrew nor an equivalent model is registered in 3.7.0's
        # PP-Structure language map. Keep its structural models at defaults.
        reader = PPStructureV3(paddlex_config=str(pinned), device="cpu", cpu_threads=2,
                               enable_mkldnn=False)

    def predict(image, page, source):
        pred, = reader.predict(str(source))
        layout = [box["coordinate"] for box in pred["layout_det_res"]["boxes"]
                  if box["label"].lower() == "table"]
        tables, issues = [], []
        for index, item in enumerate(pred["table_res_list"]):
            bbox = list(map(float, layout[index])) if index < len(layout) else None
            boxes = item["cell_box_list"]
            try:
                result, joined = paddle_table(item["pred_html"], boxes, image, page, bbox, read_cell)
            except ValueError:
                issues.append("invalid_predicted_grid")
                continue
            if not joined:
                issues.append("cell_geometry_count_mismatch")
            tables.append(result)
        return tables, issues

    return predict
