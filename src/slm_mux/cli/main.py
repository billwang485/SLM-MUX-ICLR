"""SLM-MUX CLI entry point."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="slm-mux",
        description="SLM-MUX: Multi-model orchestration with confidence-based selection",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- collect ---
    collect_parser = subparsers.add_parser(
        "collect", help="Collect single-model samples for benchmarks"
    )
    collect_parser.add_argument("--config", type=str, required=True, help="Config YAML")
    collect_parser.add_argument("--models", type=str, default="", help="Override model list (comma-sep)")
    collect_parser.add_argument("--benchmark", type=str, default="", help="Override benchmark name")
    collect_parser.add_argument("--samples", type=int, default=0, help="Override samples per question")
    collect_parser.add_argument("--output-dir", type=str, default="", help="Override output directory")

    # --- run ---
    run_parser = subparsers.add_parser(
        "run", help="Run live SLM-MUX orchestration"
    )
    run_parser.add_argument("--config", type=str, required=True, help="Config YAML")
    run_parser.add_argument("--models", type=str, default="", help="Override model list (comma-sep)")
    run_parser.add_argument("--samples", type=int, default=0, help="Override samples per model")

    # --- offline ---
    offline_parser = subparsers.add_parser(
        "offline", help="Offline MUX simulation from pre-collected data"
    )
    offline_parser.add_argument("--benchmark", type=str, required=True, choices=["math500", "gpqa", "gsm8k", "ifeval", "humaneval"])
    offline_parser.add_argument("--data-dir", type=str, required=True, help="Dir containing model JSONs")
    offline_parser.add_argument("--models", type=str, default="", help="Comma-sep model IDs (default: auto-detect)")
    offline_parser.add_argument("--samples", type=int, default=5, help="Samples per model to vote")
    offline_parser.add_argument("--trials", type=int, default=10, help="Monte Carlo trials")
    offline_parser.add_argument("--seed", type=int, default=42)
    offline_parser.add_argument("--output-dir", type=str, default="")

    # --- search ---
    search_parser = subparsers.add_parser(
        "search", help="Search for optimal model combinations"
    )
    search_parser.add_argument("--benchmark", type=str, required=True, choices=["math500", "gpqa", "gsm8k", "ifeval", "humaneval"])
    search_parser.add_argument("--data-dir", type=str, required=True, help="Dir containing model JSONs")
    search_parser.add_argument("--models", type=str, default="", help="Comma-sep model IDs (default: auto-detect)")
    search_parser.add_argument("--k-min", type=int, default=2, help="Min models in combo")
    search_parser.add_argument("--k-max", type=int, default=5, help="Max models in combo")
    search_parser.add_argument("--lambda-c", type=float, default=0.5, help="Contradiction penalty weight")
    search_parser.add_argument("--robust-level", type=int, default=2)
    search_parser.add_argument("--output-dir", type=str, default="")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "collect":
        from slm_mux.cli.collect import run_collect
        run_collect(args)
    elif args.command == "run":
        from slm_mux.cli.run import run_mux
        run_mux(args)
    elif args.command == "offline":
        from slm_mux.cli.offline import run_offline
        run_offline(args)
    elif args.command == "search":
        from slm_mux.cli.search import run_search
        run_search(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
