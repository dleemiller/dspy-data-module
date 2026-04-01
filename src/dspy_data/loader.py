"""
Load and inspect collected traces from Collect output.

Supports both JSON directory output and JSONL file output.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_collected(path: str | Path) -> list[dict]:
    """Load all collected traces from a Collect output path.

    Args:
        path: Directory containing JSON files, or path to a JSONL file.

    Returns:
        List of dicts with keys: inputs, trace, output, reward.
    """
    path = Path(path)

    if path.is_file() and path.suffix == ".jsonl":
        return _load_jsonl(path)
    elif path.is_dir():
        return _load_json_dir(path)
    else:
        raise ValueError(f"Path must be a .jsonl file or directory: {path}")


def _load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {i}: {e}")
    return entries


def _load_json_dir(path: Path) -> list[dict]:
    entries = []
    for json_file in sorted(path.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            entries.append(data)
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping malformed file {json_file.name}: {e}")
    return entries


def filter_collected(
    entries: list[dict],
    *,
    min_reward: float | None = None,
    max_reward: float | None = None,
    has_output: bool | None = None,
) -> list[dict]:
    """Filter collected entries by reward or output presence.

    Args:
        entries: List of collected trace dicts.
        min_reward: Keep entries with reward >= this value.
        max_reward: Keep entries with reward <= this value.
        has_output: If True, keep only entries with non-None output.

    Returns:
        Filtered list of entries.
    """
    result = entries
    if min_reward is not None:
        result = [e for e in result if e.get("reward") is not None and e["reward"] >= min_reward]
    if max_reward is not None:
        result = [e for e in result if e.get("reward") is not None and e["reward"] <= max_reward]
    if has_output is True:
        result = [e for e in result if e.get("output") is not None]
    elif has_output is False:
        result = [e for e in result if e.get("output") is None]
    return result


def extract_tool_calls(entry: dict) -> list[dict]:
    """Extract structured tool calls from a collected entry's trajectory.

    Works with the ReAct trajectory dict format (tool_name_N, tool_args_N,
    observation_N) captured by ScoreAndSaveWrapper.

    Returns:
        List of dicts with keys: thought, tool_name, tool_args, observation.
    """
    trajectory = entry.get("trajectory")
    if not trajectory or not isinstance(trajectory, dict):
        return []

    calls = []
    idx = 0
    while f"tool_name_{idx}" in trajectory:
        call = {
            "thought": trajectory.get(f"thought_{idx}", ""),
            "tool_name": trajectory.get(f"tool_name_{idx}", ""),
            "tool_args": trajectory.get(f"tool_args_{idx}", {}),
            "observation": trajectory.get(f"observation_{idx}", ""),
        }
        reasoning = trajectory.get(f"reasoning_{idx}")
        if reasoning:
            call["reasoning"] = reasoning
        calls.append(call)
        idx += 1
    return calls


def collected_stats(entries: list[dict]) -> dict:
    """Compute summary statistics over collected entries."""
    if not entries:
        return {"count": 0}

    rewards = [e["reward"] for e in entries if e.get("reward") is not None]
    has_output = sum(1 for e in entries if e.get("output") is not None)
    has_trace = sum(1 for e in entries if e.get("trace"))
    has_trajectory = sum(1 for e in entries if e.get("trajectory"))

    # Tool usage stats
    all_tool_calls = []
    tools_per_entry = []
    for e in entries:
        calls = extract_tool_calls(e)
        all_tool_calls.extend(calls)
        tools_per_entry.append(len(calls))

    tool_names = [c["tool_name"] for c in all_tool_calls if c.get("tool_name")]

    stats = {
        "count": len(entries),
        "has_output": has_output,
        "has_trace": has_trace,
        "has_trajectory": has_trajectory,
    }

    if rewards:
        stats.update(
            {
                "reward_count": len(rewards),
                "reward_mean": round(sum(rewards) / len(rewards), 4),
                "reward_min": round(min(rewards), 4),
                "reward_max": round(max(rewards), 4),
            }
        )

    if all_tool_calls:
        stats["tool_calls_total"] = len(all_tool_calls)
        stats["tool_calls_mean"] = round(sum(tools_per_entry) / len(tools_per_entry), 1)
        stats["tools_used"] = dict(sorted({t: tool_names.count(t) for t in set(tool_names)}.items()))

    return stats
