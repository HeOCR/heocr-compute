import unittest

from heocr_compute.remote.tesseract_telemetry import summarize


class TesseractTelemetryTest(unittest.TestCase):
    def test_attributes_native_errors_to_the_absolute_window_with_skips(self):
        lanes = ["gold", "replay", "replay", "gold", "gold"]
        log = ["not telemetry\n", "HEOCR_VISIT\t3\t0.2\t0.4\n", "HEOCR_VISIT\t4\tskip\n",
               "HEOCR_VISIT\t5\t0.3\t0.5\n"]
        result = summarize(log, lanes, {"gold", "replay", "empty"}, 2, 5)
        self.assertEqual(result["lanes"]["gold"], {
            "attempted": 2, "usable": 1, "skipped": 1,
            "loss": {"sum": 0.3, "count": 1, "mean": 0.3},
            "bcer": {"sum": 0.5, "count": 1, "mean": 0.5},
        })
        self.assertEqual(result["lanes"]["replay"]["bcer"]["sum"], 0.4)
        self.assertEqual(result["lanes"]["empty"]["bcer"], {"sum": 0, "count": 0, "mean": None})

    def test_refuses_rewind_gap_or_nonfinite_error(self):
        for first, second in [("1\t0.1\t0.2", "1\tskip"), ("2\tskip", "3\tskip"),
                              ("1\tnan\t0.2", "2\tskip"), ("1\t0\t2", "2\tskip")]:
            with self.subTest(first=first, second=second), self.assertRaises(ValueError):
                summarize([f"HEOCR_VISIT\t{record}\n" for record in (first, second)],
                          ["gold", "gold"], {"gold"}, 0, 2)
        with self.assertRaises(ValueError):
            summarize([], ["gold", "gold"], {"gold"}, 0, 2)
