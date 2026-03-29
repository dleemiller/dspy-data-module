"""
CLI for dspy-data: inspect collected traces and build SFT datasets.

Usage:
    # Inspect traces
    dspy-data stats traces.jsonl

    # Build SFT dataset from traces
    dspy-data build-sft traces.jsonl --output sft_dataset/ \
        --min-reward 0.5 --best-of-n --system-prompt "You are..."

    # Build with tool schemas for TRL tool-calling SFT
    dspy-data build-sft traces.jsonl --output sft_dataset/ \
        --tools tools.json --best-of-n
"""

import argparse
import json
import logging

from .loader import collected_stats, extract_tool_calls, filter_collected, load_collected
from .sft import to_hf_dataset, to_sft_examples

logger = logging.getLogger(__name__)


def cmd_stats(args):
    """Show statistics for collected traces."""
    entries = load_collected(args.input)
    stats = collected_stats(entries)
    print(json.dumps(stats, indent=2))

    if args.tool_usage:
        from collections import Counter

        all_tools = Counter()
        traces_with = Counter()
        for e in entries:
            calls = extract_tool_calls(e)
            tools_in_trace = {c["tool_name"] for c in calls}
            for c in calls:
                all_tools[c["tool_name"]] += 1
            for t in tools_in_trace:
                traces_with[t] += 1

        print("\nTool usage:")
        for tool, count in all_tools.most_common():
            pct = traces_with[tool] / len(entries) * 100 if entries else 0
            print(f"  {tool}: {count} calls, {traces_with[tool]}/{len(entries)} traces ({pct:.0f}%)")


def cmd_build_sft(args):
    """Build SFT dataset from collected traces."""
    entries = load_collected(args.input)
    logger.info(f"Loaded {len(entries)} traces from {args.input}")

    # Filter by reward
    if args.min_reward is not None:
        entries = filter_collected(entries, min_reward=args.min_reward, has_output=True)
        logger.info(f"After min_reward={args.min_reward} filter: {len(entries)} traces")

    # Best-of-N selection
    if args.best_of_n:
        entries = _select_best_per_problem(entries, args.problem_key)
        logger.info(f"After best-of-N selection: {len(entries)} traces")

    # Load tool schemas if provided
    tools = None
    if args.tools:
        with open(args.tools) as f:
            tools = json.load(f)
        logger.info(f"Loaded {len(tools)} tool schemas from {args.tools}")

    # Build SFT examples
    sft_kwargs = {
        "system_prompt": args.system_prompt or "",
        "include_thoughts": not args.no_thoughts,
        "tools": tools,
    }

    if args.final_answer_key:
        sft_kwargs["final_answer_key"] = args.final_answer_key

    if args.include_reward:
        sft_kwargs["include_reward"] = True

    # Build SFT examples with metadata columns from inputs
    metadata_keys = [k.strip() for k in args.metadata_keys.split(",")] if args.metadata_keys else []
    sft_kwargs["metadata_keys"] = metadata_keys

    examples = to_sft_examples(entries, **sft_kwargs)
    logger.info(f"Built {len(examples)} SFT examples")

    # Save
    from pathlib import Path

    output_path = Path(args.output)
    if output_path.suffix == ".jsonl":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example, default=str) + "\n")
        logger.info(f"Saved JSONL to {args.output}")
    else:
        # Save as HF dataset directory
        dataset = to_hf_dataset(entries, **sft_kwargs)
        dataset.save_to_disk(args.output)
        logger.info(f"Saved HF dataset to {args.output}")

    # Print summary
    msg_counts = [len(ex["messages"]) for ex in examples]
    print("\nSFT Dataset Summary:")
    print(f"  Examples: {len(examples)}")
    if msg_counts:
        print(
            f"  Messages per example: min={min(msg_counts)}, max={max(msg_counts)}, "
            f"mean={sum(msg_counts)/len(msg_counts):.1f}"
        )
    if tools:
        print(f"  Tools: {len(tools)} schemas included")


def _select_best_per_problem(entries, problem_key="func_name"):
    """Select the highest-reward trace per problem."""
    from collections import defaultdict

    by_problem = defaultdict(list)
    for e in entries:
        key = e.get("inputs", {}).get(problem_key, "unknown")
        by_problem[key].append(e)

    best = []
    for _problem, candidates in sorted(by_problem.items()):
        scored = sorted(candidates, key=lambda e: e.get("reward") or 0.0, reverse=True)
        best.append(scored[0])

    return best


def main():
    parser = argparse.ArgumentParser(prog="dspy-data", description="DSPy data collection utilities")
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show trace statistics")
    p_stats.add_argument("input", help="JSONL file or directory of JSON files")
    p_stats.add_argument("--tool-usage", action="store_true", help="Show per-tool usage breakdown")

    # build-sft
    p_sft = subparsers.add_parser("build-sft", help="Build SFT dataset from traces")
    p_sft.add_argument("input", help="JSONL file or directory of JSON files")
    p_sft.add_argument("--output", "-o", required=True, help="Output path (.jsonl or directory)")
    p_sft.add_argument("--min-reward", type=float, default=None, help="Filter traces by minimum reward")
    p_sft.add_argument("--best-of-n", action="store_true", help="Select best trace per problem")
    p_sft.add_argument(
        "--problem-key", default="func_name", help="Input field to group by for best-of-N"
    )
    p_sft.add_argument("--tools", default=None, help="JSON file with tool schemas for TRL")
    p_sft.add_argument("--system-prompt", default=None, help="System prompt to prepend")
    p_sft.add_argument(
        "--final-answer-key", default=None, help="Output field key (default: cython_code)"
    )
    p_sft.add_argument(
        "--no-thoughts", action="store_true", help="Exclude ReAct thoughts from messages"
    )
    p_sft.add_argument("--include-reward", action="store_true", help="Include reward column in dataset")
    p_sft.add_argument(
        "--metadata-keys",
        default=None,
        help="Comma-separated input field names to include as columns (e.g., func_name,description)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "build-sft":
        cmd_build_sft(args)


if __name__ == "__main__":
    main()
