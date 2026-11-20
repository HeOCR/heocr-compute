# Remote-first compute plan

2026-11-20 · Proposal for lead review; no jobs launched. The goal is to remove open-data work from the Mac's two-slot, AC-aware scheduler while retaining its restricted work locally.

## Boundary first

Remote inputs, dependencies and outputs must be public-domain or openly licensed, with source licences recorded; training inputs must not be noncommercial-licensed. Synthetic does not automatically mean safe: check every source text, font, image, derived artifact, split and manifest. Never send evaluation test material, restricted evaluation repositories, real test-split runs, partner documents or office cases text to a pod. This includes copies, derivatives and reference shingles; deny by default if provenance is uncertain. A public/open dev set is remote-eligible only after confirming it is not test or otherwise restricted. Do decontamination against restricted references locally; a remote-only open-corpus check cannot replace that gate. Never publish development results as performance; performance is not measured on a held-out set yet.

The runner's `policy.py` verifies local paths and hashes against configured open roots and a required denylist; it does not establish licence, split or provenance, and cannot inspect data fetched by a job on the pod. Before a run the lead must approve an explicit source and destination manifest, pinned revisions and hashes, and licence/split checks for **all** inputs including pod-side downloads. Fail closed on ambiguity. No secrets in manifests, commands, images, logs or GitHub. Use only existing runner credential mechanisms; the current RunPod API key is local and the pod-scoped key is for self-termination. There is currently no general HF/GitHub credential forwarding mechanism: authenticated remote uploads are blocked until a separately reviewed extension provides one under that constraint. Public read-only downloads need no secret.

## Inventory and evidence

Scheduler intervals below are elapsed time between the logged AC release and root-process exit, not CPU-hours, verified successful completions, or predicted remote duration. The 30-second polling, pauses, retries and concurrent jobs distort them; time waiting for a slot is excluded. Names alone do not prove data eligibility. Sources: the local power scheduler's `scheduler.sh`, `units.txt`, archived `units.done_*` and `scheduler.log` (18–20 November 2026), and task README/Markdown notes; these local sources are not published as input manifests. The scheduler caps active units at two. The pointed-render task README describes a 74,930-line open-text pool and local CLEAN gate; the scaled-glyph task README describes 300-page pools and a 0.7 GiB batch, illustrating size and the need for rights and leakage checks.

| Job | Classification and observed Mac time | Reason and boundary |
|---|---|---|
| hocrsyngen synthetic **line** rendering | Split; a recent render-labelled unit occupied ~0.36 h (09:11–09:32, 20 Nov); other short attempts may have failed | Render approved open training text/fonts remotely; local restricted CLEAN gate and any eval-derived material stay local. |
| hdocsyngen synthetic **page** rendering | Split; Mac time not isolated in scheduler records reviewed | Open training pages may be remote; evaluation/test pages and any restricted text stay local. Record page count, resolution and artifact size in the pilot. |
| Tesseract LSTM training and lstmf preparation | Split; no reliable Tesseract training duration isolated; scheduler's `lstmf` preparation examples were ~0.03–0.15 h | Open training and open validation only remotely; no test set, restricted reference or office cases. Training is CPU, not CUDA-accelerated by merely selecting a GPU pod. |
| OCR pipeline ablations and decontamination | Split; an explicitly open pipeline unit ran ~0.45 h (19 Nov); other OCR units ~0.27–0.57 h, eligibility unverified | Open dev images and published weights can run remotely after source review; restricted-reference comparisons and real test runs stay local. |
| htext plenary extraction / NER tagging | Split; extraction ~0.28 h; stage 1 released at 05:12 on 20 Nov and had no exit in the reviewed log by 09:52 (>4.6 h elapsed, not a completed runtime) | Open, licensed text only, after checking personal-data and publication rules; locally perform any restricted-reference check. |
| Dataset build, packaging and upload | Split; htext build examples ~0.07–0.11 h; one HF backup ~0.59 h (20 Nov) | Public/open datasets can be built remotely; private, restricted or mixed backups remain local. A backup or upload name does not establish eligibility. |
| Real test-split evaluation, partner documents, office cases | Local-only; timing varies | The material and its text cannot leave the Mac, regardless of licence or apparent synthetic origin. |

## Resource and cost assumptions

[RunPod's pricing page](https://www.runpod.io/pricing), checked **20 November 2026**, lists Secure Cloud RTX 4090 at $0.89/hour, container disk at $0.10/GB-month, volume disk $0.10 running / $0.20 idle per GB-month, and standard network storage under 1 TB at $0.07/GB-month. The runner README records a **historical** 18 November live CPU quote of $0.97/hour for 32 `cpu3c` vCPUs / 64 GB; the public pricing page did not verify today's CPU rate. Treat it as a planning assumption, not a current quote. Before approval, use `heocr-remote quote` to check the live CPU catalogue, availability, disk and caps. GPU rates can change too.

Illustrations below assume actual billed pod lifetime includes setup, transfer and teardown: CPU $0.97/h plus runner's conservative 40 GB disk allowance $0.008/h = **$0.978/h**; GPU $0.89/h plus 60 GB disk allowance $0.012/h = **$0.902/h**. These are estimates, not quotes or measurements; sizes and durations are pilot hypotheses. Runner reserves the configured wall limit plus 15 minutes, not just expected duration, and defaults to $25 per run and $150 per month. Per-run costs include assumed setup/transfer time but not persistent volumes, external service charges or tax. Check actual availability, disk needs and live quote; split workloads if a worst-case reservation exceeds the cap. Record billed time, units completed and actual cost before ranking from experience.

| Remote-eligible portion | Proposed job, data path and changes | Typical-run estimate |
|---|---|---|
| Line rendering | CPU base image `runpod/base:1.0.2-ubuntu2404`; pinned hocrsyngen, fonts and open corpus; 40 GB scratch as a starting point. GitHub supplies pinned code; manifest/SSH supplies small approved inputs; generated images and annotations need a reviewed HF dataset export, checksums and readback. Current output suffix filter excludes image/archive files and default pull is 2 GB, so add a safe bulk-artifact path and job template. | 2 h pod × $0.978 = **$1.96**; batch size and duration unmeasured. |
| Page rendering | Same CPU image, pinned hdocsyngen and approved assets; begin with 40 GB scratch and measure full-resolution output before resizing. Same GitHub/manifest inputs and HF export gap; add page template and dataset output support. | 4 h × $0.978 = **$3.91**; unmeasured. |
| LSTM training | CPU base image, 32 vCPUs / 64 GB, 40 GB scratch initially; pin Tesseract version to the comparison round, limit OpenMP threads and batch up to eight independent four-thread arms only if RAM and input size permit. Open data from HF public snapshots or verified manifest/SSH; published starter model from HF; return final checkpoints and logs through existing SSH output. Add reproducible training template, checkpoint/restart and sizing checks; 20 GB default upload and output limits may constrain inputs/results. | 8 h × $0.978 = **$7.82** for a hypothetical batch of arms, not a measured per-arm price. |
| Open-dev OCR and open-only decontam | CPU base image for Tesseract; GPU image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` and 4090 for GPU-capable engines. Pinned GitHub code and public HF data/models; 40 GB CPU / 60 GB GPU initial disk. Return logs and aggregate files via SSH; keep restricted comparison local. Add pinned engine/ablation templates and source/split checks. | CPU 2 h = **$1.96**; GPU 2 h × $0.902 = **$1.80**. Workload-dependent, unmeasured. |
| Open htext NER/tagging | CPU base for extraction and CPU tagging; GPU image for GPU-compatible tagging only after benchmarking. Public HF/GitHub open corpus, pinned model/code; start with 40 GB CPU disk or 60 GB GPU disk. Logs and small text results via SSH; large datasets require the bulk HF path. Add resumable sharding and safe dataset export; confirm corpus licence and personal-data review. | CPU 6 h = **$5.87**; optional GPU 3 h = **$2.71**, neither a measured speedup nor a committed choice. |
| Open dataset build/upload | CPU base; 40 GB initial disk, sized to input + temporary + output. Fetch public, pinned HF/GitHub sources or manifest/SSH open inputs; stage versioned output on HF only after credential mechanism and readback are reviewed. Add resumable upload, output hash manifest and explicit publication/rights approval. | 2 h × $0.978 = **$1.96**; output size and upload time unmeasured. |

Existing runner images and profiles are described in [README](../README.md). `out/` returns checkpoints, logs and scores only, not arbitrary images or archives; it is not an HF publisher. The runner's `--max-upload-gb` defaults to 20 GB and `--max-pull-gb` to 2 GB. Any new bulk path must preserve fail-closed source checks, hashes, output verification, the budget ledger and termination, and must not silently expand the permitted data. Provision enough ephemeral disk for expanded inputs, scratch and outputs; confirm all desired outputs are durable and hash-verified before termination. No persistent volume is assumed in the estimates.

## Monthly sketch and order

A conservative illustrative allocation, not authorization to run or a guarantee of speed:

| Allocation | Calculation | Monthly USD |
|---|---:|---:|
| Four page and six line render batches | 4 × $3.91 + 6 × $1.96 | $27.40 |
| Four 8-hour LSTM training batches | 4 × $7.82 | $31.28 |
| Eight CPU and four GPU open-dev OCR jobs | 8 × $1.96 + 4 × $1.80 | $22.88 |
| Four CPU NER jobs | 4 × $5.87 | $23.48 |
| Six open builds/uploads | 6 × $1.96 | $11.76 |
| Retries, price changes, storage and other overhead reserve | fixed | $33.20 |
| **Total** | | **$150.00** |

1. **LSTM batches**, if several independent open arms are ready: sharing a 32-vCPU pod could displace several long Mac scheduler slots for one pod-hour. No measured per-dollar ranking yet.
2. **Long open-text NER**: the plenary stage's ongoing interval shows a potentially large slot release; benchmark CPU versus GPU before choosing the latter.
3. **Line/page rendering**: frequent and readily parallelizable, but first solve bulky outputs and confirm source/licence and CLEAN boundaries.
4. **Open-dev OCR**: useful to unblock iterations; the examples are shorter and eligibility must be checked per set.
5. **Builds/uploads**: batch with upstream jobs to avoid setup and transfer overhead; short builds alone may save little.

This ordering is a hypothesis about Mac slot-hours per dollar, not a measured comparison; two simultaneously occupied Mac slots represent two slot-hours per clock hour, not extra elapsed time saved. Use pilot results to revise it. The runner ledger covers only its own account-labelled runs, not all account spend. Reconcile RunPod billing and any other projects, maintain the $150 account-wide ceiling, retain the reserve and use caps that account for already committed charges. A stopped pod's persistent storage can still incur charges; terminate and verify cleanup after durable output readback.

## First approval: a bounded render template

Approve **design and implementation of a CPU line-render template**, followed by a separate lead-approved small run. The template should accept only an explicitly reviewed open, licence-recorded, SHA-pinned corpus, fonts and code revision; fixed seed/count; no restricted material; bounded threads, disk and wall time; a generation manifest, log and checksums. Specify a public/read-only input snapshot and an output destination before execution. Add a tested, approved bulk-output and credential path first, or limit the pilot to small metadata outputs with images deliberately retained only until termination; do not claim a rendered dataset has been saved without verified durable readback. Run the restricted CLEAN check locally. The lead can first review the template, source manifest, dry-run quote, per-run cap and output plan; no pod or spend is authorized by this document.
