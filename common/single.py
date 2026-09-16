from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_experiment_definition
from .dataset import dataset_layout, validate_dataset_layout
from .dataset_compat import DatasetCompatibilityError, prepare_context_dataset
from .runner import build_context, next_run_dir, resolve_resume_dir, run_experiment


def run_single(experiment_dir: Path, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one comparison experiment.")
    parser.add_argument("--stage", choices=["train", "test", "both"], default="both")
    parser.add_argument("--data", default="data/地大隧道")
    parser.add_argument("--output", default="output")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(experiment_dir).resolve().parent
    definition = load_experiment_definition(Path(experiment_dir).resolve())
    data_root = Path(args.data)
    data_root = data_root if data_root.is_absolute() else root / data_root
    report = validate_dataset_layout(data_root)
    if not report.ok:
        for error in report.errors:
            print(error)
        return 2
    output_root = Path(args.output)
    output_root = output_root if output_root.is_absolute() else root / output_root
    run_dir = resolve_resume_dir(output_root, args.run_dir) if args.run_dir else next_run_dir(output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    context = build_context(root, definition, run_dir, dataset_layout(data_root), args.iterations)
    try:
        context = prepare_context_dataset(context, dry_run=args.dry_run)
    except DatasetCompatibilityError as error:
        print(f"[{definition.id}] blocked: {error}")
        return 2
    result = run_experiment(definition, context, args.stage, args.dry_run)
    print(f"[{definition.id}] {result.get('status', 'unknown')}: {run_dir}")
    return 0 if result.get("status") in {"completed", "skipped", "planned"} else 1

