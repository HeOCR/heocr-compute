"""heocr-remote: open-data training and engine runs on rented pods, inside a budget.

    heocr-remote profiles
    heocr-remote quote --profile cpu [--hours 12]
    heocr-remote manifest ROOT [--include GLOB ...] [--exclude GLOB ...] -o inputs.json
    heocr-remote run --profile cpu --manifest inputs.json --cmd 'bash in/job.sh' [--dry-run]
    heocr-remote smoke [--dry-run]
    heocr-remote policy [PATH ...]
    heocr-remote spend [--month YYYY-MM]
    heocr-remote sweep [--all] [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from . import budget, manifest, profiles
from .api import RunPod, RunPodError, load_account, load_key
from .policy import Policy, Refused
from .ledger import Ledger, state_dir
from .runner import RunFailed, Runner, Spec, sweep

SMOKE_CAP_USD = 0.50
SMOKE_INPUT = "A tiny invented input for the remote runner's smoke test.\nLine two of it.\n"
SMOKE_JOB = """set -eu
cd /heocr
{ echo "nproc $(nproc)"; uname -m; sha256sum in/input.txt; } > out/smoke.txt
echo '{"smoke": "ok"}' > out/score.json
head -c 65536 /dev/urandom > out/final_checkpoint
head -c 8192 /dev/urandom > out/smoke_1.234_10_100.checkpoint
echo not-returned > out/page.png
"""


def _connect():
    """The RunPod client and this account's ledger, after checking the key is HeOCR's account's."""
    expected = load_account()
    api = RunPod(load_key())
    label = api.check_account(expected)
    return api, Ledger(account=label)


def _profile(a) -> profiles.Profile:
    p = profiles.PROFILES[a.profile]
    if getattr(a, "cpu_flavor", None) or getattr(a, "vcpus", None):
        p = p.sized(a.cpu_flavor, a.vcpus)
    return p.community() if getattr(a, "community", False) else p


def cmd_profiles(a):
    for p in profiles.PROFILES.values():
        print(f"{p.name:6} {p.pod_type:40} image {p.image}, disk {p.disk_gb} GB, default limit {p.default_hours} h")


def cmd_quote(a):
    api, led = _connect()
    p = _profile(a)
    hours = a.hours or p.default_hours
    compute = profiles.live_compute_price(api, p)
    q = budget.quote(compute_usd_per_hour=compute, disk_gb=p.disk_gb, wall_hours=hours, balance=api.balance(),
                     month_spent=led.month_to_date()["total"], run_cap=a.run_cap, month_cap=a.month_cap)
    print(f"{p.name}: {p.pod_type}")
    for line in q.lines():
        print(line)
    r = budget.refusals(q)
    print("verdict       " + ("refused: " + "; ".join(r) if r else "within the caps"))
    return 1 if r else 0


def cmd_manifest(a):
    entries = manifest.build(a.root, Policy.load(), includes=tuple(a.include or ("**",)), excludes=tuple(a.exclude or ()))
    manifest.dump(entries, a.output)
    print(f"{a.output}: {len(entries)} files, {sum(e.size for e in entries) / (1 << 20):.1f} MB")


def _spec(a, p: profiles.Profile, manifest_path, root, cmd, run_cap) -> Spec:
    return Spec(profile=p, manifest_path=Path(manifest_path), root=Path(root), cmd=cmd,
                hours=a.hours or p.default_hours, run_cap=run_cap, month_cap=a.month_cap,
                pulls=tuple(getattr(a, "pull", None) or ("**",)),
                pull_intermediate=getattr(a, "pull_intermediate", False),
                results=Path(a.results).expanduser() if getattr(a, "results", None) else None,
                max_pull_gb=getattr(a, "max_pull_gb", 2.0), max_upload_gb=getattr(a, "max_upload_gb", 20.0),
                dry_run=a.dry_run, yes=a.yes)


def _report(res) -> int:
    print(f"run {res['run_id']}: {res['status']}" + (f" ({res['error']})" if res.get("error") else ""))
    if res.get("deadman"):
        print(f"dead-man's switch: {res['deadman']}" +
              (f", self-removal: {res['self_removal']}" if res.get("self_removal") else ""))
    return 0 if res["status"] in ("ok", "dry_run", "declined") else 1


def cmd_run(a):
    root = Path(a.root) if a.root else Path(a.manifest).resolve().parent
    spec = _spec(a, _profile(a), a.manifest, root, a.cmd, a.run_cap)
    return _report(Runner(*_connect()).run(spec))


def cmd_render(a):
    from .render import launch

    return _report(launch(a, _connect))


def cmd_hf_train(a):
    from .hf_train import launch

    return launch(a, _profile(a), *_connect())


def cmd_collect_hf(a):
    from .hf_train import collect

    return collect(a, *_connect())


def cmd_smoke(a):
    work = Path(tempfile.mkdtemp(prefix="smoke-", dir=state_dir()))
    (work / "input.txt").write_text(SMOKE_INPUT, encoding="utf-8")
    (work / "smoke.sh").write_text(SMOKE_JOB, encoding="utf-8")
    policy = Policy.load().with_root("smoke", work)  # the smoke run's own invented inputs
    manifest.dump(manifest.build(work, policy), work / "inputs.json")
    a.hours = a.hours or profiles.PROFILES["smoke"].default_hours
    a.profile = "smoke"
    spec = _spec(a, profiles.PROFILES["smoke"], work / "inputs.json", work, "bash in/smoke.sh",
                 min(a.run_cap, SMOKE_CAP_USD))
    spec.policy = policy
    spec.prove_deadman = True
    try:
        return _report(Runner(*_connect()).run(spec))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cmd_policy(a):
    from .policy import OPEN_CATEGORIES
    pol = Policy.load()
    for cat, desc in OPEN_CATEGORIES.items():
        print(f"{cat:9} {len(pol.roots.get(cat, []))} roots  ({desc})")
    print(f"denylist  {len(pol.deny_paths)} paths, {len(pol.deny_globs)} globs, {len(pol.deny_hashes)} hashes")
    rc = 0
    for p in a.paths:
        why = pol.deny_reason(p)
        cat = pol.category_of(p)
        verdict = f"refused: {why}" if why else (f"open ({cat})" if cat else "refused: not under an open root")
        rc |= not verdict.startswith("open")
        print(f"{p}: {verdict}")
    return rc


def cmd_spend(a):
    label = load_account()["label"]
    led = Ledger(account=label)
    m = led.month_to_date(a.month)
    usd = lambda x: f"${x:.2f}" if x == 0 or x >= 1 else f"${x:.4f}"
    print(f"{m['month']}, account {label}: {usd(m['spent'])} spent in {len(m['finals'])} runs"
          f" + {usd(m['open_estimate'])} reserved by {len(m['open'])} open runs"
          f" = {usd(m['total'])} of ${budget.MONTH_CAP_USD:.2f} this month")
    for r in m["finals"]:
        f = r["final"]
        print(f"  {f['date']}  {f['pod_type']:40} {f['hours']:7.3f} h  ${f['usd']:7.4f}  {f['status']}")
    for r in m["open"]:
        e = r["reserve"]
        print(f"  {e['date']}  {e['pod_type']:40} open, up to ${e['usd_est']:.2f}")
    if m["others"]:
        print(f"not counted: {len(m['others'])} runs ({usd(m['other_spent'])}) on other or untagged accounts")
    print(f"ledger: {led.path}")


def cmd_sweep(a):
    api, led = _connect()
    killed = sweep(api, led, everything=a.all, dry_run=a.dry_run)
    print(f"sweep: {len(killed)} pods terminated")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="heocr-remote", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def money(p):
        p.add_argument("--hours", type=float, help="wall-time limit (default 12 h CPU, 4 h GPU)")
        p.add_argument("--run-cap", type=float, default=budget.RUN_CAP_USD, help="per-run cap in $ (default 25)")
        p.add_argument("--month-cap", type=float, default=budget.MONTH_CAP_USD,
                       help="monthly cap in $ (default and maximum 150)")

    sub.add_parser("profiles").set_defaults(fn=cmd_profiles)

    q = sub.add_parser("quote", help="live price, worst case and the caps' verdict; spends nothing")
    q.add_argument("--profile", choices=sorted(profiles.PROFILES), required=True)
    q.add_argument("--community", action="store_true", help="GPU only: the community cloud")
    q.add_argument("--cpu-flavor", help="CPU only: another flavour (cpu3c, cpu3g, cpu5c, ...), e.g. when the default has no capacity")
    q.add_argument("--vcpus", type=int, help="CPU only: another vCPU count, a power of two")
    money(q)
    q.set_defaults(fn=cmd_quote)

    m = sub.add_parser("manifest", help="list a folder's files with their sha256")
    m.add_argument("root")
    m.add_argument("--include", action="append")
    m.add_argument("--exclude", action="append")
    m.add_argument("-o", "--output", required=True)
    m.set_defaults(fn=cmd_manifest)

    def run_flags(p):
        money(p)
        p.add_argument("--dry-run", action="store_true", help="check everything, create nothing")
        p.add_argument("--yes", action="store_true", help="do not ask before starting the pod")
        p.add_argument("--results", help="where results go (default ~/heocr-remote/results/RUN_ID)")
        p.add_argument("--max-pull-gb", type=float, default=2.0)

    r = sub.add_parser("run", help="one run on a pod")
    r.add_argument("--profile", choices=["cpu", "gpu"], required=True)
    r.add_argument("--community", action="store_true", help="GPU only: the community cloud")
    r.add_argument("--cpu-flavor", help="CPU only: another flavour (cpu3c, cpu3g, cpu5c, ...), e.g. when the default has no capacity")
    r.add_argument("--vcpus", type=int, help="CPU only: another vCPU count, a power of two")
    r.add_argument("--manifest", required=True)
    r.add_argument("--root", help="folder the manifest's paths are relative to (default: the manifest's)")
    r.add_argument("--cmd", required=True, help="run on the pod in /heocr; inputs in in/, results to out/")
    r.add_argument("--pull", action="append", help="glob of out/ files to bring back (default: all returnable)")
    r.add_argument("--pull-intermediate", action="store_true", help="also bring back intermediate checkpoints")
    r.add_argument("--max-upload-gb", type=float, default=20.0)
    run_flags(r)
    r.set_defaults(fn=cmd_run)

    from . import render

    line = sub.add_parser("render-lines", help="bounded CPU render from pinned public inputs to a private HF dataset")
    line.set_defaults(fn=cmd_render)
    line.add_argument("--hours", type=float, default=render.MAX_HOURS, help="pod wall limit, at most 4 h")
    line.add_argument("--run-cap", type=float, default=render.MAX_COST, help="run cap in USD, at most 1")
    line.add_argument("--month-cap", type=float, default=budget.MONTH_CAP_USD)
    line.add_argument("--dry-run", action="store_true", help="perform preflight and quote without any remote writes")
    line.add_argument("--yes", action="store_true", help="do not prompt to create the pod")
    line.add_argument("--results", help="directory for local transit and receipts")
    line.add_argument("--count", type=int, required=True)
    line.add_argument("--seed", type=int, required=True)
    line.add_argument("--mode", help="pass a mode to a pinned hocrsyngen revision that supports it")
    line.add_argument("--font", action="append", help="governed font ID, repeatable")
    line.add_argument("--code-revision", default=render.CODE, help="full public hocrsyngen commit SHA")
    line.add_argument("--font-manifest-sha256", default=render.FONT_SHA)
    line.add_argument("--dataset", required=True, help="existing private HeOCR HF dataset")

    train = sub.add_parser("hf-train", help="launch autonomous training from approved private HF pins")
    train.set_defaults(fn=cmd_hf_train)
    train.add_argument("--profile", choices=["cpu", "gpu", "smoke"], required=True)
    train.add_argument("--cpu-flavor")
    train.add_argument("--vcpus", type=int)
    train.add_argument("--job", required=True, help="approved private training job specification")
    train.add_argument("--approved-sha256", required=True, help="lead-approved SHA-256 of the exact job bytes")
    train.add_argument("--hf-secret", required=True, help="Runpod managed secret name")
    train.add_argument("--hf-repo-scope", help="restrict all Hub reads and writes to this one private dataset")
    train.add_argument("--registry-auth-id", help="Runpod managed registry credential ID for a private image")
    train.add_argument("--hours", type=float, required=True)
    train.add_argument("--run-cap", type=float, required=True)
    train.add_argument("--month-cap", type=float, default=budget.MONTH_CAP_USD)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--yes", action="store_true")

    collect = sub.add_parser("collect-hf", help="verify and collect an autonomous run after pod teardown")
    collect.set_defaults(fn=cmd_collect_hf)
    collect.add_argument("run_id")
    collect.add_argument("--results", default="~/heocr-remote/results")

    s = sub.add_parser("smoke", help=f"the smallest end-to-end run (capped at ${SMOKE_CAP_USD:.2f})")
    run_flags(s)
    s.set_defaults(fn=cmd_smoke)

    po = sub.add_parser("policy", help="the open roots and local denylist in force; check paths against them")
    po.add_argument("paths", nargs="*")
    po.set_defaults(fn=cmd_policy)

    sp = sub.add_parser("spend", help="month-to-date spend from the ledger")
    sp.add_argument("--month", help="YYYY-MM (default: this month, UTC)")
    sp.set_defaults(fn=cmd_spend)

    sw = sub.add_parser("sweep", help="terminate orphaned heocr pods")
    sw.add_argument("--all", action="store_true", help="every heocr pod, live runs included")
    sw.add_argument("--dry-run", action="store_true")
    sw.set_defaults(fn=cmd_sweep)

    a = ap.parse_args(argv)
    try:
        return a.fn(a) or 0
    except (budget.BudgetRefused, Refused, manifest.ManifestError, RunFailed, RunPodError, ValueError) as e:
        print(f"heocr-remote: refused: {e}" if isinstance(e, (budget.BudgetRefused, Refused))
              else f"heocr-remote: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
