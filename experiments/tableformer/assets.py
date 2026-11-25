"""Verify or stage the exact local TableFormer artifact family."""

from hashlib import file_digest
from pathlib import Path


BASE_REVISION = "fc0f2d45e2218ea24bce5045f58a389aed16dc23"
UPSTREAM = "a14d633bc9e80a985dab5d6aad686dace72488ba"
BASE_SHA = "2a7d6c924b3cd12fb99a09280ca9c33a89c5d60b93253617d2e088c1a40374d9"
CONFIG_SHA = "984e122ceb8ccf84d84c9d2882f6f2302a44b4f1e577babd6289892c36f3cffd"
MODEL_SUBDIR = Path("docling-project--docling-models/model_artifacts/tableformer/accurate")
LAYOUT_SUBDIR = Path("docling-project--docling-layout-heron")
WEIGHT = "tableformer_accurate.safetensors"
CONFIG = "tm_config.json"
LAYOUT_CONFIGS = {
    "config.json": "fdea30805ce2f5666b147fca941dcdd27ad468e27d6ed21902207d3da056a97d",
    "preprocessor_config.json": "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",
}


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"unexpected file SHA-256: {path.name}")


def stock_assets(root):
    folder = (root / MODEL_SUBDIR).resolve(strict=True)
    config, weights = folder / CONFIG, folder / WEIGHT
    verify(config, CONFIG_SHA)
    verify(weights, BASE_SHA)
    return config, weights


def verify_layout_configs(root):
    for name, expected in LAYOUT_CONFIGS.items():
        verify(root / LAYOUT_SUBDIR / name, expected)


def candidate_session(models, config, weights, expected, scratch):
    from heocr.tableformer import TableFormerSession

    weights = weights.resolve(strict=True)
    verify(weights, expected)
    session = TableFormerSession(models)
    staged = scratch / MODEL_SUBDIR
    staged.mkdir(parents=True)
    (staged / CONFIG).write_bytes(config.read_bytes())
    (staged / WEIGHT).symlink_to(weights)
    (scratch / LAYOUT_SUBDIR).symlink_to((models / LAYOUT_SUBDIR).resolve(strict=True))
    # Keep this alternate pin scoped to the offline pilot session.
    # The package's stock model verification remains in force.
    session._models = scratch
    session._record["weights_sha256"][WEIGHT] = expected
    return session
