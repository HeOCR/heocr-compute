"""Close a trainer's checkpoint phase log, including quiet intervals."""
from __future__ import annotations

import argparse
from pathlib import Path


def publish_phase_log(phase: Path, destination: Path, start: int, end: int) -> None:
    """Record the completed attempt window and atomically move the closed log."""
    if not 0 <= start < end:
        raise ValueError("invalid trainer phase log window")
    with phase.open("ab") as log:
        log.write(f"Checkpoint visit window ({start}, {end}]\n".encode("ascii"))
    phase.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("start", type=int, help="previous checkpoint visit, exclusive")
    parser.add_argument("end", type=int, help="completed checkpoint visit, inclusive")
    args = parser.parse_args()
    publish_phase_log(args.phase, args.destination, args.start, args.end)


if __name__ == "__main__":
    main()
