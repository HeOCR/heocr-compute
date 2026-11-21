# Kraken candidate runners

This is a candidate pipeline, not an approved evaluation configuration. The
registered PP-OCRv6 asset is the [small model](https://zenodo.org/records/21788405)
(Apache-2.0); the record calls it small but also contains contradictory medium
wording. BiblIA [general](https://zenodo.org/records/5468286),
[Ashkenazi](https://zenodo.org/records/5468478),
[Italian](https://zenodo.org/records/5468573), and
[Sephardi](https://zenodo.org/records/5468665) models each declare CC BY-SA 4.0.
Rights, share-alike and applicable scripts require a decision before admission.
Source and training exposure must be checked before any held-out use.

With Python 3.12, create a dedicated venv and install
`pip install -r comparators/requirements-kraken-macos-arm.txt`. It records the
macOS ARM environment; Linux AMD64 and macOS Intel require separately resolved
and frozen dependencies and have not been tested. Package and system library
hashes are not locked. Download `small.safetensors`, `BiblIA_01.mlmodel`,
`Ashkenazi_01.mlmodel`, `Italian_01.mlmodel`, and `Sephardi_01.mlmodel` from
the corresponding Zenodo records into an asset directory. The runner verifies
weights against the digests in `kraken_runner.py`, and page mode verifies the
Kraken 7.2 bundled BLLA segmentation weights ([model record](https://zenodo.org/records/22879549), Apache-2.0). Kraken itself is Apache-2.0. Do not replace them with a
floating model.

```sh
python comparators/kraken_runner.py --model ppocrv6-small --mode line \
  --assets /path/to/assets --input /path/to/line.png --output /path/to/line.txt
python comparators/kraken_runner.py --model biblia-general --mode page \
  --assets /path/to/assets --input /path/to/page.png --output /path/to/page.txt
```

Line mode treats the complete input as one line. Page mode uses BLLA baseline
segmentation, horizontal right-to-left ordering, and the chosen recognizer.
Both use CPU, one thread, batch size 1, Kraken's logical-order reorder with a
right-to-left base, and UTF-8 text output. These settings are candidates only:
confirm text order, mixed-direction content, blank images and scorer normalization
before fixing a run. Category applicability remains as in the main roster.

— roster[openai/codex#shay-macbook-m4]@heocr/opensource

Smoke tests on macOS ARM used invented images: all five models produced nonempty
line output, and PP-OCRv6 produced nonempty page output on a multi-line image.
A single-line image in page mode produced a valid empty result. These checks
establish plumbing only; they do not establish recognition quality or scoring.
