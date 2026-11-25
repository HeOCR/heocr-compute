"""Geometry targets and checkpoint audit for the P3 TableFormer recipe."""

import math
import re

from targets import otsl


SHA = re.compile(r"[0-9a-f]{64}")
ANCHORS = {"fcel", "ecel", "ched", "rhed", "srow"}
SKIP_NEXT = {"nl", "ucel", "xcel"}


def require_pins(values, fields):
    if not isinstance(values, dict) or any(
        not isinstance(values.get(key), str) or not SHA.fullmatch(values[key]) for key in fields
    ):
        raise ValueError("missing P3 provenance pins")


def capture_contract(manifest):
    capture = manifest.get("p3")
    require_pins(capture, ("hdocsyngen_recipe_sha256", "rectifier_sha256", "rectifier_config_sha256",
                           "rectifier_verdict_sha256"))
    fits = capture.get("scanifier_fit_sha256")
    if not isinstance(fits, list) or not fits or any(not isinstance(s, str) or not SHA.fullmatch(s) for s in fits):
        raise ValueError("missing pinned Scanifier phone fits")
    return capture


def checked_crop(item, capture, size):
    route = item.get("p3")
    require_pins(route, ("source_page_sha256", "captured_page_sha256", "phone_fit_sha256",
                         "rectified_page_sha256", "rectifier_sha256", "rectifier_config_sha256"))
    if (
        route.get("accepted") is not True
        or route.get("phone_fit_sha256") not in capture["scanifier_fit_sha256"]
        or any(route[key] != capture[key] for key in ("rectifier_sha256", "rectifier_config_sha256"))
    ):
        raise ValueError("P3 requires exactly the accepted phone rectifier route")
    extent = route.get("rectified_size", [])
    crop = route.get("crop_xyxy", [])
    if (
        not isinstance(extent, list) or len(extent) != 2
        or not isinstance(crop, list) or len(crop) != 4
        or any(type(n) is not int for n in (*crop, *extent))
    ):
        raise ValueError("invalid P3 rectified table crop geometry")
    left, top, right, bottom = crop
    if not (0 <= left < right <= extent[0] and 0 <= top < bottom <= extent[1]
            and size == (right - left, bottom - top)):
        raise ValueError("P3 table crop disagrees with the rectified raster")


def cell_box(cell, size):
    raw = cell.get("bbox")
    if (
        not isinstance(raw, (list, tuple)) or len(raw) != 4
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in raw)
    ):
        raise ValueError("P3 cells need crop-local XYXY boxes")
    left, top, right, bottom = raw
    width, height = size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("P3 cell box is empty or outside its crop")
    return ((left + right) / (2 * width), (top + bottom) / (2 * height),
            (right - left) / width, (bottom - top) / height)


def bbox_targets(table, size):
    """Bind boxes to the upstream decoder states that emit them."""
    tags = otsl(table)
    cols = table["cols"]
    owners = [None] * len(tags)
    boxes = []
    for index, cell in enumerate(table["cells"]):
        row, col, rowspan, colspan = (cell[key] for key in ("r", "c", "rowspan", "colspan"))
        if table["direction"] == "rtl":
            col = cols - col - colspan
        boxes.append(cell_box(cell, size))
        for dy in range(rowspan):
            for dx in range(colspan):
                owners[(row + dy) * (cols + 1) + col + dx] = index

    events, covered = [], set()
    previous_tag = "nl"
    for position, tag in enumerate(tags):
        if tag == "lcel" and previous_tag == "ucel":
            raise ValueError("P3 geometry cannot supervise a span rewritten by the upstream decoder")
        # Docling emits the prior cell's box on the following boundary.
        owner = None
        if previous_tag not in SKIP_NEXT and tag in ANCHORS | {"nl", "ucel"}:
            owner = owners[position - 1]
        # Docling merges the first horizontal continuation with the boundary box.
        # Both targets use the visible merged extent, which preserves that union.
        if tag == "lcel" and previous_tag != "lcel":
            owner = owners[position]
        if owner is not None:
            events.append((position, boxes[owner]))
            covered.add(owner)
        previous_tag = tag
    if len(covered) != len(boxes):
        raise ValueError("P3 requires an upstream box event for every cell")
    return tags, events


def trainable_groups(model):
    transformer = model._tag_transformer
    groups = {
        "embedding": ("_tag_transformer._embedding", transformer._embedding),
        "token_head": ("_tag_transformer._fc", transformer._fc),
        "box_head": ("_bbox_decoder", model._bbox_decoder),
    }
    for index, layer in enumerate(transformer._decoder.layers):
        groups[f"decoder_{index}"] = (f"_tag_transformer._decoder.layers.{index}", layer)
    return groups


def checkpoint_diff(base_path, candidate_path, groups):
    """Compare the saved tensors and reject unexpected or missing updates."""
    import torch
    from safetensors import safe_open

    allowed = {f"{prefix}.{key}": group
               for group, (prefix, module) in groups.items() for key, _ in module.named_parameters()}
    changed = {name: 0 for name in groups}
    rows = {}
    with safe_open(base_path, framework="pt") as base, safe_open(candidate_path, framework="pt") as candidate:
        if base.keys() != candidate.keys() or not set(allowed).issubset(base.keys()):
            raise ValueError("P3 export changes checkpoint keys or omits a trained parameter")
        for key in base.keys():
            old, new = base.get_tensor(key), candidate.get_tensor(key)
            if old.shape != new.shape or old.dtype != new.dtype:
                raise ValueError(f"P3 export changes shape or dtype: {key}")
            group = allowed.get(key)
            equal = torch.equal(old.reshape(-1).view(torch.uint8), new.reshape(-1).view(torch.uint8))
            if not equal and group is None:
                raise ValueError(f"frozen tensor changed: {key}")
            if group is None:
                continue
            delta = (new.double() - old.double()).abs()
            if not torch.isfinite(delta).all():
                raise ValueError(f"nonfinite P3 delta: {key}")
            maximum = delta.max().item()
            count = torch.count_nonzero(delta).item()
            changed[group] += int(maximum > 0)
            rows[key] = {"group": group, "changed_elements": count, "max_abs_delta": maximum}
    if not all(changed.values()):
        raise ValueError(f"P3 trainable group did not move: {changed}")
    for subhead in ("_bbox_decoder._bbox_embed.", "_bbox_decoder._class_embed."):
        if not any(key.startswith(subhead) and row["changed_elements"] for key, row in rows.items()):
            raise ValueError(f"P3 box subhead did not move: {subhead}")
    return {"changed_tensors_by_group": changed, "trainable_tensors": rows, "frozen_changes": 0}
