"""The money rules, checked before any pod exists.

A run is refused when its worst-case cost (live hourly price x its wall-time limit, plus start-up and
disk) is above the per-run cap, would push the month over the monthly cap, or is more than the account
balance. The monthly cap is the project's budget (set 2026-11-18, $150): a flag can lower it, never raise it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MONTH_CAP_USD = 150.0        # the project's budget, 2026-11-18. Raising it means changing this line, in review.
RUN_CAP_USD = 25.0           # default per-run cap; --run-cap changes it
STARTUP_HOURS = 0.25         # image pull, setup and transfers before the wall-time clock starts counting
DISK_USD_PER_GB_HOUR = 0.0002  # container disk while running (about $0.10-0.14 per GB-month), rounded up
PRICE_DRIFT = 1.05           # a created pod billed more than 5% above the quote is terminated at once


class BudgetRefused(Exception):
    pass


@dataclass(frozen=True)
class Quote:
    compute_usd_per_hour: float  # live catalogue price
    usd_per_hour: float      # compute + disk
    hours: float             # wall-time limit + start-up
    usd_est: float           # worst case
    balance: float
    month_spent: float       # month to date, open reservations included
    run_cap: float
    month_cap: float

    def lines(self) -> list[str]:
        return [
            f"price         ${self.usd_per_hour:.4f}/h (live, compute + disk)",
            f"worst case    ${self.usd_est:.2f} for {self.hours:.2f} h (wall-time limit + start-up)",
            f"per-run cap   ${self.run_cap:.2f}",
            f"month so far  ${self.month_spent:.2f} of ${self.month_cap:.2f}"
            f" (after this run, at worst ${self.month_spent + self.usd_est:.2f})",
            f"balance       ${self.balance:.2f}",
        ]


def hourly(compute_usd_per_hour: float, disk_gb: int) -> float:
    return compute_usd_per_hour + disk_gb * DISK_USD_PER_GB_HOUR


def quote(*, compute_usd_per_hour: float, disk_gb: int, wall_hours: float, balance: float,
          month_spent: float, run_cap: float = RUN_CAP_USD, month_cap: float = MONTH_CAP_USD) -> Quote:
    if month_cap > MONTH_CAP_USD:
        raise BudgetRefused(f"the monthly cap is ${MONTH_CAP_USD:.0f}; a flag can lower it, not raise it")
    if wall_hours <= 0 or not math.isfinite(wall_hours):
        raise BudgetRefused("the wall-time limit must be a positive number of hours")
    if compute_usd_per_hour <= 0:
        raise BudgetRefused("no live price for this pod type")
    per_hour = hourly(compute_usd_per_hour, disk_gb)
    hours = wall_hours + STARTUP_HOURS
    est = math.ceil(per_hour * hours * 100) / 100
    return Quote(compute_usd_per_hour, per_hour, hours, est, balance, month_spent, run_cap, month_cap)


def refusals(q: Quote) -> list[str]:
    out = []
    if q.usd_est > q.run_cap:
        out.append(f"worst case ${q.usd_est:.2f} is over the per-run cap ${q.run_cap:.2f} (--run-cap)")
    if q.month_spent + q.usd_est > q.month_cap:
        out.append(f"month to date ${q.month_spent:.2f} + ${q.usd_est:.2f} would pass the monthly cap"
                   f" ${q.month_cap:.2f}")
    if q.usd_est > q.balance:
        out.append(f"worst case ${q.usd_est:.2f} is more than the account balance ${q.balance:.2f}")
    return out


def check(q: Quote) -> None:
    r = refusals(q)
    if r:
        raise BudgetRefused("; ".join(r))


def overpriced(q: Quote, billed_usd_per_hour: float) -> bool:
    """True when a created pod is billed more than PRICE_DRIFT above the quoted compute price."""
    return billed_usd_per_hour > q.compute_usd_per_hour * PRICE_DRIFT


def billed(usd_per_hour: float, seconds: float) -> float:
    """Cost of ``seconds`` at ``usd_per_hour``, rounded up to the started minute."""
    minutes = max(1, math.ceil(seconds / 60))
    return usd_per_hour * minutes / 60
