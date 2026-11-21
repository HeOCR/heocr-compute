# heocr-compute

HeOCR's tooling for running open-data training and engine jobs on rented machines, so a laptop stops being the bottleneck. It has one tool, `heocr-remote`, which runs one job on a [RunPod](https://www.runpod.io) pod and keeps it inside a budget:
- the monthly cap is $150, set by the project and lowerable by flag, never raisable;
- the per-run cap is $25 by default.

Every pod is terminated at the end, whether the job succeeded or failed.

Python 3.11 or later. The core runner uses the standard library. The optional line-render publisher and autonomous training collector use the Hugging Face Hub client.

See [autonomous private Hub training](docs/autonomous-hf-training.md) for approved jobs that pull inputs on the pod and return snapshots through the private Hub.

```sh
pip install .
heocr-remote profiles
heocr-remote quote --profile cpu                 # live price, worst case, the caps' verdict; spends nothing
heocr-remote smoke --dry-run                     # the whole check, no pod
heocr-remote smoke                               # the smallest real run, at most $0.50; also proves the pod can remove itself
heocr-remote spend                               # month to date
heocr-remote sweep --dry-run                     # orphaned heocr pods
```

## Profiles

| Profile | Machine | Live price on 2026-11-18 | Default wall-time limit |
|---|---|---|---|
| `cpu` | 32 vCPUs, compute-optimised (`cpu3c`), 64 GB RAM | $0.97/h | 12 h (worst case $11.86) |
| `gpu` | 1 × RTX 4090 (24 GB), secure cloud | $0.90/h | 4 h (worst case $3.84) |
| `gpu --community` | the same on the community cloud | $0.35/h | 4 h (worst case $1.50) |
| `smoke` | 2 vCPUs (`cpu3c`) | $0.06/h | 30 min (worst case $0.05) |

When the default size has no capacity, `--cpu-flavor` and `--vcpus` (a power of two) choose another CPU size for `quote` and `run`; the caps and the live price check are unchanged. The `cpu` profile is sized for 4–8 lstmtraining runs side by side, eight at their usual four OpenMP threads. Prices are never taken from this table: every run reads the live catalogue price first. The images are RunPod's own:
- `runpod/base:1.0.2-ubuntu2404` for the CPU profiles;
- `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` for the GPU profile.

The job installs what it needs. For lstmtraining, it should check that its Tesseract version matches the one the round is compared with.

## A run

```sh
heocr-remote manifest ~/heocr-remote/open/synthetic/r10 -o r10.json   # every file with its sha256
heocr-remote run --profile cpu --manifest r10.json --cmd 'bash in/train.sh' --hours 10
```

On the pod, the inputs are in `/heocr/in`. The job runs in `/heocr`, under `timeout`. The runner writes `/heocr/out/pod.json` (run id, pod id, pod type, hourly price) first, so the results name their machine. The job writes what should come back to `/heocr/out`, and anything else, such as intermediate checkpoints and scratch files, to `/heocr/work`. Its output goes to `/heocr/out/job.log`.

`run` goes through these steps, and stops at the first that fails:

1. **Sweep.** Terminate orphaned heocr pods (below).
2. **Inputs.** Hash every manifest file again; a file that changed since the manifest stops the run. Check every file against the open roots and the local denylist (below).
3. **Money,** under the ledger's lock, so two runs can't both pass:
   - read the live price and the account balance;
   - compute the worst case: price × (wall-time limit + 15 min of start-up), plus disk;
   - refuse when the worst case is over the per-run cap (`--run-cap`), would push the month over the monthly cap (`--month-cap`, at most 150), or is more than the balance;
   - reserve the worst case in the ledger.
4. **Pod.** Create it, then:
   - terminate it at once if it is billed more than 5% above the quote;
   - wait for SSH;
   - arm a dead-man's switch on the pod: after the wall-time limit plus 15 minutes, the pod removes itself with `runpodctl` and RunPod's pod-scoped key, even if this machine slept or lost its network. An SSH session does not inherit the container's environment, so the switch reads the pod id and key from the container's first process. If either is missing, or the image has no `runpodctl`, the run says which, and the sweeper is the only backstop.
5. **Upload** the manifest's files as one tar stream over SSH. On the pod, `sha256sum -c` checks every hash.
6. **Job.** Start it detached on the pod and poll for its exit code with short SSH calls, so a dropped connection doesn't end it. At the wall-time limit it ends as `timeout`.
7. **Results.** Pull back what the job left in `out/`, which must be checkpoints, logs and scores only. By suffix:
   - checkpoints: `.checkpoint`, `_checkpoint`, `.traineddata`, `.pt`, `.safetensors` and similar;
   - logs and scores: `.log`, `.json`, `.tsv`, `.txt`, `.csv`, `.md`.

   lstmtraining's intermediate checkpoints (`name_0.512_120_9000.checkpoint`) stay on the pod, unless `--pull-intermediate` is given. Further limits:
   - `--pull GLOB` narrows the set;
   - `--max-pull-gb` (2 GB by default) and this machine's free disk bound it.

   Every file's sha256 is compared with the pod's. A mismatch is fetched once more, then the run fails. The hashes are written to `SHA256SUMS` next to the results, in `~/heocr-remote/results/RUN_ID` by default.
8. **Terminate** the pod, on success, error, wall-time limit or Ctrl-C (SIGINT, SIGTERM and SIGHUP are trapped), and confirm it is gone. Write the ledger's final record. If termination can't be confirmed, the run says so loudly, and its reservation stays open in the ledger.

Without `--yes`, `run` asks before creating the pod. `--dry-run` does steps 1–3 against the live price and balance, lists orphans without terminating them, and shows the pod it would create. While a run is active on macOS, `caffeinate` keeps the machine from idle sleep.

## CPU line rendering

Use the [bounded line-render template](docs/line-render-pilot.md) to render pinned open training text on a CPU pod and publish the verified artifacts to an existing private HeOCR dataset. The HF credential stays on the machine running `heocr-remote`.

## What may be sent

Only open training inputs go to a pod: synthetic lists, htext and public-domain scans. `policy.py` checks every file, by its path as given and after resolving symlinks, when the manifest is written and again before upload. No flag loosens it.

1. **Open roots, an allowlist.** A file is sent only when it lies under a root listed for one of the open categories: `synthetic`, `htext` or `pd_scans`. Everything else is refused by default, including a symlink that leads out of its root. The categories are fixed in code. Where their roots are on a machine goes in a local file that is never committed:

   ```toml
   # ~/.config/heocr/remote_open_roots.toml
   [roots]
   synthetic = ["~/heocr-remote/open/synthetic"]
   htext = ["~/data/htext"]
   pd_scans = ["~/heocr-remote/open/pd_scans"]
   ```

2. **A local denylist,** also never committed: what must never leave the machine, even from inside an open root. It is required, so without it nothing is sent. One rule per line:

   ```text
   # ~/.config/heocr/remote_denylist.txt
   ~/data/private-set                 a path: it and everything under it
   */reference-texts/*                a glob on the full resolved path, ignoring case
   sha256-names: ~/data/private-set   files named by their sha256: a file with one of those hashes
                                      is refused wherever it sits and whatever it is called
   ```

`heocr-remote policy PATH…` shows what is in force and checks paths against it.

## The ledger

`~/.config/heocr/remote/spend.jsonl` (or `$HEOCR_REMOTE_STATE/spend.jsonl`), outside any repository. JSON lines, one per event:
- `reserve`: the worst case, before a pod exists;
- `pod`: the pod's id and billed hourly price;
- `final`: `date`, `pod_type`, `hours`, `usd`, `status`.

Hours are measured from pod creation to confirmed termination, and dollars are rounded up to the started minute.

Month to date (UTC months) counts every final, plus the worst case of every reservation without one. So a runner that died mid-run can only overstate the month, never understate it. `heocr-remote spend` prints it against the $150 cap, run by run.

**The ledger counts only this runner's spend.** The monthly cap applies to runs started with `heocr-remote`. If the RunPod account is shared with other projects, their pods are not in the ledger and do not count towards the cap. On a shared account, the balance check before each run is the only guard: a run is refused when its worst case is more than the balance left. Keep the balance at what HeOCR may spend, or give HeOCR its own account.

## The sweeper

`heocr-remote sweep` terminates every pod named `heocr-…` that is:
- not in this machine's ledger;
- from a run that already finished;
- past its run's deadline;
- from a run whose runner process on this machine is gone.

It writes the missing final record for each one. Every `run` sweeps first. `--all` also terminates live runs, and `--dry-run` only lists. To sweep regularly, add it to cron or launchd, for example every 30 minutes.

## Credentials and the account

- **API key.** It is read only from HeOCR's own key file, `~/.config/heocr/runpod.toml` (`apikey = "..."`, mode 600), or the file that `HEOCR_RUNPOD_CONFIG` names.
  - No other tool's configuration is read.
  - The `RUNPOD_API_KEY` variable is ignored too, since it may hold another project's key.
  - The key goes only into the `Authorization` header and is never printed or written.
- **Scopes.** The runner needs:
  - REST pods, read and write;
  - the catalogue, read;
  - GraphQL, read: the balance and the account's email;
  - billing, read: optional, for checking the ledger against RunPod's bills.

  The pod removes itself with RunPod's own pod-scoped key, not with this one.
- **Account check.** Before every command that calls RunPod, the runner asks which account the key belongs to, and refuses unless it is the one in a local file that is never committed:

  ```toml
  # ~/.config/heocr/remote_account.toml (or HEOCR_REMOTE_ACCOUNT)
  email = "..."      # the account's login address
  label = "heocr"    # how the ledger tags this account's runs
  ```

  The ledger tags every event with that label. The month's total, and so the cap, counts only the current account's runs. Earlier or other-account runs stay in the file as history, and `spend` reports them apart.
- **SSH** uses HeOCR's own key pair, `~/.config/heocr/remote/ssh/id_ed25519`, made on first use. Its public half is passed to the pod, so it needs no registration with any account.

## Tests

```sh
python -m unittest discover -s tests
```

Offline only: the RunPod API and the pod are fakes, the SSH tar streams run against local folders, and every path is an invented one in a temporary folder.

## Licence

Apache-2.0.
