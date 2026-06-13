from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import create_demo_data
from .pipeline import run_analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the full cohort analysis")
    run.add_argument("--real-summary", required=True)
    run.add_argument("--virtual-summary", required=True)
    run.add_argument("--real-traces")
    run.add_argument("--virtual-traces")
    run.add_argument("--config", default="configs/analysis.yaml")
    run.add_argument("--output", default="outputs")

    demo = subparsers.add_parser("demo", help="Run the synthetic-data demonstration")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--config", default="configs/analysis.yaml")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "demo":
        data_dir = Path(args.output) / "data"
        real, virtual = create_demo_data(data_dir)
        summary = run_analysis(
            real,
            virtual,
            args.output,
            args.config,
        )
    else:
        summary = run_analysis(
            args.real_summary,
            args.virtual_summary,
            args.output,
            args.config,
            args.real_traces,
            args.virtual_traces,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
