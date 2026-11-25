"""Download only the pinned public model files and verify their original bytes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


def verify(directory, pins):
    for model, spec in pins.items():
        for name, expected in spec["files"].items():
            path = directory / model / name
            with path.open("rb") as stream:
                digest = file_digest(stream, "sha256").hexdigest()
            if digest != expected["sha256"] or path.stat().st_size != expected["size"]:
                raise ValueError(f"model pin mismatch: {model}/{name}")


def fetch(directory, pins):
    def download(job):
        model, file, spec = job
        dest = directory / model / file
        dest.parent.mkdir(exist_ok=True, parents=True)
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{file}"
        with urlopen(url, timeout=60) as response, NamedTemporaryFile(dir=dest.parent) as temp:
            shutil.copyfileobj(response, temp)
            temp.flush()
            shutil.copyfile(temp.name, dest)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, [(model, file, spec) for model, spec in pins.items() for file in spec["files"]]))
    verify(directory, pins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pins", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fetch(args.output, json.loads(args.pins.read_bytes()))
