"""Read per-attempt metrics from a pinned Tesseract training forward pass."""
from __future__ import annotations

import math
from itertools import islice

SOURCE_REVISION = "db0ec62f81b0737fbbe184d8fea40af5738f8eef"
VISIT_PREFIX = "HEOCR_VISIT\t"
NATIVE_LOSS = "tesseract_activation_rms"
NATIVE_BCER = "tesseract_recoder_bag_character_error"


def instrument(source):
    """Patch the reviewed 5.5.3 CLI for exact attempted-visit windows and native metrics."""
    replacements = (
        ("int iteration = trainer.training_iteration();", "int iteration = trainer.sample_iteration();"),
        ("iteration = trainer.training_iteration()) {", "iteration = trainer.sample_iteration()) {"),
        ("      trainer.TrainOnLine(&trainer, false);", '''      const auto *sample = trainer.TrainOnLine(&trainer, false);
      if (sample != nullptr) {
        tprintf("HEOCR_VISIT\\t%d\\t%.17g\\t%.17g\\n", trainer.sample_iteration(),
                trainer.LastSingleError(ET_RMS), trainer.LastSingleError(ET_CHAR_ERROR));
      } else {
        tprintf("HEOCR_VISIT\\t%d\\tskip\\n", trainer.sample_iteration());
      }'''),
        ("} while (trainer.best_error_rate() > FLAGS_target_error_rate &&\n"
         "           (trainer.training_iteration() < max_iterations));",
         "} while (trainer.sample_iteration() < max_iterations);"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("unexpected Tesseract source for visit telemetry")
        source = source.replace(old, new)
    return source


def summarize(log, visit_lanes, lanes, start, end):
    """Attribute the complete native window (start, end] to the verified visit ledger."""
    if not 0 <= start < end:
        raise ValueError("invalid attempted-visit window")
    by_lane = {lane: {"attempted": 0, "usable": 0, "skipped": 0} for lane in lanes}
    metrics = {lane: ([], []) for lane in by_lane}
    trace = (line.removeprefix(VISIT_PREFIX).strip().split("\t")
             for line in log if line.startswith(VISIT_PREFIX))
    window = islice(visit_lanes, start, end)
    count = 0
    for count, (lane, row) in enumerate(zip(window, trace, strict=True), start + 1):
        if lane not in by_lane or row[0] != str(count):
            raise ValueError("native visit trace differs from the ordered lane ledger")
        summary = by_lane[lane]
        summary["attempted"] += 1
        if row[1:] == ["skip"]:
            summary["skipped"] += 1
            continue
        if len(row) != 3:
            raise ValueError("malformed native visit telemetry")
        loss, bcer = map(float, row[1:])
        if not (math.isfinite(loss) and loss >= 0 and 0 <= bcer <= 1):
            raise ValueError("out-of-range native loss or BCER")
        summary["usable"] += 1
        metrics[lane][0].append(loss)
        metrics[lane][1].append(bcer)
    if count != end:
        raise ValueError("native visit trace is missing attempts")
    for lane, summary in by_lane.items():
        for name, values in zip(("loss", "bcer"), metrics[lane], strict=True):
            total = math.fsum(values)
            summary[name] = {"sum": total, "count": len(values),
                             "mean": total / len(values) if values else None}
    return {"start_visit_exclusive": start, "end_visit_inclusive": end,
            "loss_definition": NATIVE_LOSS, "bcer_definition": NATIVE_BCER, "lanes": by_lane}
