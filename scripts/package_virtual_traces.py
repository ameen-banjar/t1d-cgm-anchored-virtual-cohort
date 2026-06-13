#!/usr/bin/env python3
"""Create a checksummed Zenodo archive containing simulator-only traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import tarfile
from pathlib import Path


EXPECTED_TRACES = 180


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.csv")
        if not path.name.lower().startswith("summary")
    )


def _manifest_bytes(root: Path, traces: list[Path]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        ["relative_path", "virtual_id", "scenario", "rows", "bytes", "sha256"]
    )
    for path in traces:
        relative = path.relative_to(root)
        virtual_id = path.name.split("_V0", 1)[0]
        rows = sum(1 for _ in path.open("rb")) - 1
        writer.writerow(
            [relative, virtual_id, path.parent.name, rows, path.stat().st_size, _sha256(path)]
        )
    return output.getvalue().encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("release/virtual-cohort-traces-v0.1.0.tar.gz"),
    )
    args = parser.parse_args()

    root = args.trace_root.expanduser().resolve()
    traces = _trace_files(root)
    if len(traces) != EXPECTED_TRACES:
        raise SystemExit(
            f"Expected {EXPECTED_TRACES} virtual traces, found {len(traces)} in {root}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_bytes(root, traces)
    with tarfile.open(args.output, "w:gz") as archive:
        info = tarfile.TarInfo("virtual_traces/MANIFEST.csv")
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))
        for path in traces:
            archive.add(
                path,
                arcname=Path("virtual_traces") / path.relative_to(root),
                recursive=False,
            )

    print(f"Wrote {args.output} with {len(traces)} simulator-only traces.")


if __name__ == "__main__":
    main()
