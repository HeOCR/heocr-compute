# Kraken candidate runners

All four BiblIA v1 models are admitted for historical handwriting comparison.
In the project’s comparison workflow, running unmodified weights for scoring
creates no derivative and does not apply share-alike to the results. Attribute
each model used in the results. Download the weights from upstream and keep
them out of redistributions. The BiblIA dataset is noncommercial and excluded
from HeOCR training.

Eval selects the general and applicable script-specific models for each set,
records the selection in the roster amendment, and checks source exposure
before held-out use. Eval also freezes the runtime and segmentation settings.

| Asset | File | Licence | Attribution for results |
|---|---|---|---|
| BiblIA general v1 | `BiblIA_01.mlmodel` | CC BY-SA 4.0 | Daniel Stökl Ben Ezra, [Medieval Hebrew manuscripts version 1.0](https://doi.org/10.5281/zenodo.5468286) |
| BiblIA Ashkenazi v1 | `Ashkenazi_01.mlmodel` | CC BY-SA 4.0 | Daniel Stökl Ben Ezra, [Medieval Hebrew manuscripts in Ashkenazi bookhand](https://doi.org/10.5281/zenodo.5468478) |
| BiblIA Italian v1 | `Italian_01.mlmodel` | CC BY-SA 4.0 | Daniel Stökl Ben Ezra, [Medieval Hebrew manuscripts in Italian bookhand version 1.0](https://doi.org/10.5281/zenodo.5468573) |
| BiblIA Sephardi v1 | `Sephardi_01.mlmodel` | CC BY-SA 4.0 | Daniel Stökl Ben Ezra, [Medieval Hebrew manuscripts in Sephardi bookhand version 1.0](https://doi.org/10.5281/zenodo.5468665) |
| Kraken PP-OCRv6 small v1 | `small.safetensors` | Apache-2.0 | Benjamin Kiessling, [PP-OCRv6 (small) multilingual handwritten and printed text recognition model](https://doi.org/10.5281/zenodo.21788405) |
| Kraken 7.2 bundled BLLA segmenter | `blla.mlmodel` | Apache-2.0 | Benjamin Kiessling, [General segmentation model for print and handwriting](https://doi.org/10.5281/zenodo.22879549) |

Record the applicable licence, attribution, and SHA-256 from
[`kraken_runner.py`](kraken_runner.py) in the run record. The PP-OCRv6 record
has conflicting size wording. The pinned file is explicitly the small variant.

With Python 3.12, create a dedicated venv and install
`pip install -r comparators/requirements-kraken-macos-arm.txt`. It records the
macOS ARM environment. Linux AMD64 has a separate lock file and
[container recipe](IMAGES.md). macOS Intel remains unverified. The venv recipe
pins package versions. Download `small.safetensors`, `BiblIA_01.mlmodel`,
`Ashkenazi_01.mlmodel`, `Italian_01.mlmodel`, and `Sephardi_01.mlmodel` from
the corresponding Zenodo records into an asset directory. The runner verifies
weights against the digests in `kraken_runner.py`, and page mode verifies the
Kraken 7.2 bundled BLLA segmentation weights. Kraken itself is Apache-2.0.

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
