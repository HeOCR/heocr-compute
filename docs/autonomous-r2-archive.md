# Autonomous private R2 archive return

Use this `heocr-remote run` output mode for a trusted rendering recipe whose
inputs pass the usual open roots, denylist and manifest hash checks. The
launcher stays connected until it has staged and verified those inputs and
the pod supervisor acknowledges startup. It then returns the run ID. The
pod's strict termination timer and spend reservation remain active.

Install the optional R2 client locally with `pip install '.[r2]'`. Supply a
mode-600 file for an approved private bucket and a read/write credential
scoped to that bucket. The file accepts `R2_ENDPOINT` (the account's HTTPS
S3 endpoint), `R2_BUCKET`, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`. Keep
the file outside the manifest's input root. The launcher retains the bucket
key locally and uses it to sign short-lived URLs for PUT, HEAD and GET on
exactly two objects. It transfers those URLs over SSH stdin to a pod-local
mode-600 file, which the supervisor unlinks before starting the renderer.

```sh
heocr-remote run --profile gpu --root "$OPEN_INPUT_ROOT" \
  --manifest "$INPUT_MANIFEST" --cmd 'bash in/render.sh' \
  --hours 2 --run-cap 3 --r2-archive-credentials "$R2_CREDENTIAL_FILE"

heocr-remote collect-archive "$RUN_ID" \
  --r2-archive-credentials "$R2_CREDENTIAL_FILE" --results "$ARCHIVE_RESULTS"
```

The recipe writes `/heocr/out/output.tar` as an uncompressed tar of regular
files and optional directories, up to 2 GiB. Reserve ten minutes of the
wall limit for the return. No other file from the pod is copied back by
this mode.

Under `runs/<run_id>/` the supervisor PUTs `output.tar`, checks its length
by HEAD, then GETs the bytes to recompute SHA-256. Only after that succeeds
does it PUT and verify `manifest.json` with the run ID, pod ID, size and
digest. The PUTs are create-only. The supervisor invokes pod removal after
the attempt, with the deadline as a backstop. A run with no verified return
may leave an archive without a manifest. The collector requires the exact
pod to be gone before it reads the manifest.

Collection checks the reserved bucket, run and pod, streams the archive and
checks its size and SHA-256 locally. The verified archive and manifest appear
together in the run's result directory. Provider billing settles separately.
Pass the verified data through the applicable CLEAN intake before use.
