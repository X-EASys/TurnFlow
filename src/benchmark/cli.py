"""The public TurnFlow command line."""
import argparse
import json
from pathlib import Path

from benchmark import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="turnflow", description="Run the TurnFlow multi-turn benchmark.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Verify all bundled sources and judge/simulator bindings without API calls")
    listing = sub.add_parser("list", help="List bundled source keys and domains")
    listing.add_argument("--tasks", nargs="+", choices=["T1", "T2", "T3", "T4", "T5"])
    execute = sub.add_parser("run", help="Run full dialogues and K1 judgments (paid model calls)")
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--agent-model", help="Override the candidate agent model in the config")
    execute.add_argument("--tasks", nargs="+", choices=["T1", "T2", "T3", "T4", "T5"])
    selection = execute.add_mutually_exclusive_group()
    selection.add_argument("--smoke", type=int, help="Select N sources per task in source-key order")
    selection.add_argument("--source-keys", nargs="+", help="Explicit complete keys from `turnflow list`")
    execute.add_argument("--concurrency", type=int, default=1, help="Concurrent dialogues (default: 1)")
    execute.add_argument("--resume", action="store_true", help="Continue unattempted positions with the same code/data/config")
    execute.add_argument("--prepare-only", action="store_true", help="Write the selection manifest without making API calls")
    summary = sub.add_parser("report", help="Summarize an existing run without API calls")
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            from benchmark.dataset import verify
            result = verify()
        elif args.command == "list":
            from benchmark.dataset import select, source_path
            from benchmark.io import read
            result = [{"source_key": key, "domain": read(source_path(key))["domain"]}
                      for key in select(tasks=args.tasks)]
        elif args.command == "run":
            from benchmark.execution import run
            result = run(args.config, args.output, agent_model=args.agent_model, tasks=args.tasks,
                         keys=args.source_keys, smoke=args.smoke, concurrency=args.concurrency,
                         resume=args.resume, prepare_only=args.prepare_only)
        else:
            from benchmark.execution import report
            result = report(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command in ("run", "report") and (result["counts"]["INVALID"] or result["counts"]["notrun"]):
            if not getattr(args, "prepare_only", False):
                return 2
        return 0
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"turnflow: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
