from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from common.config import discover_experiments
from common.dataset import dataset_layout, validate_dataset_layout
from common.dataset_compat import DatasetCompatibilityError, dataset_compatibility_mode
from common.runner import build_context, next_run_dir, resolve_resume_dir, run_experiment, should_skip_experiment
from common.summary import write_summary


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = Path("data") / "地大隧道"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all 10 3DGS comparison experiments.")
    parser.add_argument("resume_arg", nargs="?", help="Existing run number, e.g. 03, to resume.")
    parser.add_argument("--resume", dest="resume_option", help="Existing output/NN directory or run number.")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--data", default=str(DEFAULT_DATASET), help="Dataset root containing train/ and test/.")
    parser.add_argument("--output", default="output", help="Output root relative to this folder.")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--only", nargs="+", help="Experiment ids or directory names to run.")
    parser.add_argument("--experiments", nargs="+", dest="only_alias", help="Alias of --only.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write commands/status only; never launch algorithms.")
    return parser.parse_args(argv)


def _absolute(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def select_experiments(definitions: Iterable, selectors: list[str] | None) -> list:
    definitions = list(definitions)
    if not selectors:
        return definitions
    by_key = {}
    for definition in definitions:
        by_key[definition.id.upper()] = definition
        by_key[definition.directory.upper()] = definition
    selected = []
    for selector in selectors:
        key = str(selector).upper()
        if key not in by_key:
            known = ", ".join(definition.id for definition in definitions)
            raise ValueError(f"Unknown experiment '{selector}'. Use an id or directory name. Known ids: {known}")
        definition = by_key[key]
        if definition not in selected:
            selected.append(definition)
    return selected


def write_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    dataset_root: Path,
    report,
    definitions: list,
    preserve_existing: bool = False,
) -> Path:
    payload = {
        "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "iterations": args.iterations,
        "dataset_root": str(dataset_root),
        "dataset_report": {"counts": report.counts, "warnings": report.warnings},
        "experiments": [
            {
                **definition.to_dict(),
                "dataset_compat": dataset_compatibility_mode(ROOT, definition.directory),
            }
            for definition in definitions
        ],
    }
    filename = "resume_config.json" if preserve_existing and (run_dir / "run_config.json").is_file() else "run_config.json"
    target = run_dir / filename
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume_arg and args.resume_option:
        print("Use either positional resume value or --resume, not both.", file=sys.stderr)
        return 2
    resume_value = args.resume_arg or args.resume_option
    try:
        definitions = discover_experiments(ROOT)
        selectors = args.only or args.only_alias
        selected = select_experiments(definitions, selectors)
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2

    dataset_root = _absolute(ROOT, args.data)
    report = validate_dataset_layout(dataset_root)
    if not report.ok:
        print("Dataset layout is incomplete:", file=sys.stderr)
        for error in report.errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    if report.warnings:
        for warning in report.warnings:
            print(f"warning: {warning}", file=sys.stderr)

    output_root = _absolute(ROOT, args.output)
    try:
        if resume_value:
            run_dir = resolve_resume_dir(output_root, resume_value)
        else:
            run_dir = next_run_dir(output_root)
            run_dir.mkdir(parents=True, exist_ok=False)
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    try:
        write_run_config(run_dir, args, dataset_root, report, selected, preserve_existing=bool(resume_value))
    except DatasetCompatibilityError as error:
        print(f"Unable to write run configuration: {error}", file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []
    for definition in selected:
        experiment_output = run_dir / definition.directory
        if should_skip_experiment(experiment_output):
            print(f"[skip:{definition.id}] {definition.name} (completed artifacts exist)")
            continue
        print(f"[{definition.id}] {definition.name}")
        try:
            context = build_context(ROOT, definition, run_dir, dataset_layout(dataset_root), args.iterations)
        except DatasetCompatibilityError as error:
            print(f"[{definition.id}] blocked: {error}", file=sys.stderr)
            failures.append((definition.id, "blocked"))
            if not args.continue_on_error:
                print("Stopped at first failure. Resume with this run number to retry missing experiments.", file=sys.stderr)
                break
            continue
        result = run_experiment(definition, context, args.mode, dry_run=args.dry_run)
        status = result.get("status", "unknown")
        print(f"[{definition.id}] {status}")
        if status in {"failed", "blocked"}:
            if result.get("error"):
                print(f"[{definition.id}] error: {result['error']}", file=sys.stderr)
            if result.get("stderr_tail"):
                print(f"[{definition.id}] stderr tail:\n{result['stderr_tail']}", file=sys.stderr)
            failures.append((definition.id, status))
            if not args.continue_on_error:
                print("Stopped at first failure. Resume with this run number to retry missing experiments.", file=sys.stderr)
                break

    rows = write_summary(run_dir, selected)
    (run_dir / "selected_experiments.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Run directory: {run_dir}")
    if failures:
        print("Failed or blocked: " + ", ".join(f"{key}:{value}" for key, value in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
