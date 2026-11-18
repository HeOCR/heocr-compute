import datetime as dt
import tempfile
import unittest
from pathlib import Path

from heocr_compute.remote import budget
from heocr_compute.remote.ledger import Ledger


def q(**kw):
    args = dict(compute_usd_per_hour=0.96, disk_gb=40, wall_hours=12, balance=100.0, month_spent=0.0)
    args.update(kw)
    return budget.quote(**args)


class BudgetTest(unittest.TestCase):
    def test_cpu_default_is_within_caps(self):
        quote = q()
        self.assertEqual(budget.refusals(quote), [])
        self.assertAlmostEqual(quote.hours, 12.25)
        self.assertGreater(quote.usd_est, 0.96 * 12)
        self.assertLess(quote.usd_est, 25)

    def test_over_run_cap(self):
        r = budget.refusals(q(compute_usd_per_hour=2.5))
        self.assertEqual(len(r), 1)
        self.assertIn("per-run cap", r[0])
        self.assertEqual(budget.refusals(q(compute_usd_per_hour=2.5, run_cap=40)), [])

    def test_over_month_cap(self):
        r = budget.refusals(q(month_spent=140.0))
        self.assertTrue(any("monthly cap" in x for x in r))
        with self.assertRaises(budget.BudgetRefused):
            budget.check(q(month_spent=140.0))

    def test_lower_month_cap_flag(self):
        self.assertTrue(budget.refusals(q(month_cap=10.0)))

    def test_month_cap_cannot_be_raised(self):
        with self.assertRaises(budget.BudgetRefused):
            q(month_cap=151.0)

    def test_over_balance(self):
        r = budget.refusals(q(balance=5.0))
        self.assertTrue(any("balance" in x for x in r))

    def test_no_price_refused(self):
        with self.assertRaises(budget.BudgetRefused):
            q(compute_usd_per_hour=0)

    def test_overpriced(self):
        quote = q()
        self.assertFalse(budget.overpriced(quote, 0.96))
        self.assertFalse(budget.overpriced(quote, 1.0))
        self.assertTrue(budget.overpriced(quote, 1.2))

    def test_billed_rounds_up_to_the_minute(self):
        self.assertAlmostEqual(budget.billed(0.6, 1), 0.01)
        self.assertAlmostEqual(budget.billed(0.6, 61), 0.02)


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = dt.datetime(2026, 11, 18, 12, 0, tzinfo=dt.timezone.utc)
        self.led = Ledger(Path(self.tmp.name) / "spend.jsonl", now=lambda: self.t)

    def tearDown(self):
        self.tmp.cleanup()

    def test_month_to_date_counts_finals_and_open_reservations(self):
        self.led.reserve("a", pod_type="cpu3cx32", usd_est=12.0)
        self.led.final("a", pod_type="cpu3cx32", hours=2.0, usd=1.92, status="ok")
        self.led.reserve("b", pod_type="4090", usd_est=4.0)        # open: counted at its estimate
        m = self.led.month_to_date()
        self.assertEqual(m["month"], "2026-11")
        self.assertAlmostEqual(m["spent"], 1.92)
        self.assertAlmostEqual(m["open_estimate"], 4.0)
        self.assertAlmostEqual(m["total"], 5.92)

    def test_other_months_excluded(self):
        self.t = dt.datetime(2026, 10, 31, 23, 0, tzinfo=dt.timezone.utc)
        self.led.reserve("old", pod_type="x", usd_est=9.0)
        self.led.final("old", pod_type="x", hours=1, usd=9.0, status="ok")
        self.t = dt.datetime(2026, 11, 1, 0, 30, tzinfo=dt.timezone.utc)
        self.assertEqual(self.led.month_to_date()["total"], 0.0)
        self.assertEqual(self.led.month_to_date("2026-10")["total"], 9.0)

    def test_final_fields(self):
        self.led.reserve("a", pod_type="cpu3cx2", usd_est=0.1)
        e = self.led.final("a", pod_type="cpu3cx2", hours=0.1234567, usd=0.0123456, status="ok")
        for k in ("date", "pod_type", "hours", "usd"):
            self.assertIn(k, e)
        self.assertEqual(self.led.events()[-1], e)


if __name__ == "__main__":
    unittest.main()
