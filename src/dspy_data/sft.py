"""
Convert collected traces to SFT training format.

Converts DSPy ReAct trajectory data into OpenAI-compatible chat messages
with proper tool_calls/tool roles for use with TRL's SFTTrainer.
"""

import logging
import re

from .loader import extract_tool_calls

logger = logging.getLogger(__name__)


def trajectory_to_messages(
    entry: dict,
    *,
    system_prompt: str = "",
    user_prompt_fn=None,
    final_answer_key: str = "cython_code",
    include_thoughts: bool = True,
) -> list[dict]:
    """Convert a collected entry's trajectory to OpenAI tool-calling chat format.

    Transforms the structured trajectory dict (thought_N, tool_name_N,
    tool_args_N, observation_N) into a multi-turn conversation with
    proper assistant tool_calls and tool role messages.

    Args:
        entry: Collected trace dict with trajectory, inputs, and output.
        system_prompt: System message content. If empty, no system message is added.
        user_prompt_fn: Optional callable(inputs) -> str for formatting the user message.
            If None, a default representation of inputs is used.
        final_answer_key: Key in output dict for the final assistant response.
        include_thoughts: If True, include ReAct thoughts as assistant content
            alongside tool calls. If False, tool call messages have no content.

    Returns:
        List of chat messages in OpenAI format with tool_calls.
    """
    messages = []

    # System prompt
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # User message
    inputs = entry.get("inputs", {})
    if user_prompt_fn:
        user_content = user_prompt_fn(inputs)
    else:
        user_content = "\n".join(
            f"{k}: {v}" for k, v in inputs.items() if k not in ("test_cases", "benchmark_args")
        )
    messages.append({"role": "user", "content": user_content})

    # Tool-calling turns from trajectory
    calls = extract_tool_calls(entry)
    for i, call in enumerate(calls):
        # Assistant turn with tool call
        # TRL expects arguments as a dict, not a JSON string
        assistant_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": call["tool_name"],
                        "arguments": call["tool_args"],
                    },
                }
            ],
        }
        if include_thoughts and call.get("thought"):
            assistant_msg["content"] = call["thought"]
        else:
            assistant_msg["content"] = None

        messages.append(assistant_msg)

        # Tool result
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "name": call["tool_name"],
                "content": call.get("observation", ""),
            }
        )

    # Final assistant response
    output = entry.get("output") or {}
    final_answer = output.get(final_answer_key, "")
    if final_answer:
        messages.append({"role": "assistant", "content": final_answer})

    return messages


def extract_messages(entry: dict) -> list[dict]:
    """Extract chat messages from a collected trace entry.

    If the entry has a structured trajectory (from ReAct), converts it to
    proper OpenAI tool-calling format. Otherwise falls back to extracting
    raw messages from the LM trace.

    Args:
        entry: A single collected trace dict.

    Returns:
        List of chat messages in OpenAI format.
    """
    # Prefer structured trajectory conversion
    if entry.get("trajectory"):
        return trajectory_to_messages(entry)

    # Fallback: extract raw messages from LM trace
    messages = []
    trace = entry.get("trace", [])
    for interaction in trace:
        interaction_messages = interaction.get("messages")
        if interaction_messages:
            messages.extend(interaction_messages)

        completion = interaction.get("completion")
        if completion:
            choices = completion.get("choices", [])
            for choice in choices:
                msg = choice.get("message", {})
                if msg:
                    messages.append(msg)

    return messages


def extract_metrics(entry: dict) -> dict:
    """Parse reward metrics from the last evaluate_cython observation.

    Extracts speedup, annotation score, test pass rate, etc. from the
    markdown-formatted tool output.
    """
    calls = extract_tool_calls(entry)
    if not calls:
        return {}

    # Find last observation that has results (not just compilation errors)
    observation = ""
    for call in reversed(calls):
        obs = call.get("observation", "")
        if "## Benchmark" in obs or "## Tests" in obs or "## Annotation" in obs:
            observation = obs
            break

    if not observation:
        return {}

    metrics = {}

    # Speedup
    m = re.search(r"Speedup:\s*([\d.]+)x", observation)
    if m:
        metrics["speedup"] = float(m.group(1))

    # Annotation score
    m = re.search(r"Annotation score:\s*([\d.]+)", observation)
    if m:
        metrics["annotation_score"] = float(m.group(1))

    # Tests
    m = re.search(r"Tests:\s*(\d+)/(\d+)\s*passed", observation)
    if m:
        metrics["tests_passed"] = int(m.group(1))
        metrics["tests_total"] = int(m.group(2))
        metrics["correctness"] = int(m.group(1)) / int(m.group(2)) if int(m.group(2)) > 0 else 0.0

    return metrics


def to_sft_examples(
    entries: list[dict],
    *,
    min_reward: float | None = None,
    include_reward: bool = False,
    tools: list[dict] | None = None,
    system_prompt: str = "",
    user_prompt_fn=None,
    final_answer_key: str = "cython_code",
    include_thoughts: bool = True,
    metadata_keys: list[str] | None = None,
    include_metrics: bool = False,
) -> list[dict]:
    """Convert collected entries to SFT training examples.

    Args:
        entries: List of collected trace dicts.
        min_reward: Only include entries with reward >= this value.
        include_reward: If True, include reward as metadata in each example.
        tools: List of tool JSON schemas for the 'tools' column
            (required by TRL SFTTrainer for tool-calling datasets).
        system_prompt: System prompt for trajectory_to_messages.
        user_prompt_fn: User prompt formatter for trajectory_to_messages.
        final_answer_key: Output field key for final answer.
        include_thoughts: Include ReAct thoughts in assistant messages.
        metadata_keys: Input field names to include as extra columns.
        include_metrics: Parse and include reward breakdown (speedup, annotation, tests).

    Returns:
        List of dicts with 'messages' key (and 'tools', 'reward', metadata if provided).
    """
    examples = []
    for entry in entries:
        reward = entry.get("reward")

        if min_reward is not None and (reward is None or reward < min_reward):
            continue

        if entry.get("trajectory"):
            messages = trajectory_to_messages(
                entry,
                system_prompt=system_prompt,
                user_prompt_fn=user_prompt_fn,
                final_answer_key=final_answer_key,
                include_thoughts=include_thoughts,
            )
        else:
            messages = extract_messages(entry)

        if not messages:
            continue

        example = {"messages": messages}
        if tools is not None:
            example["tools"] = tools
        if include_reward:
            example["reward"] = reward
        if metadata_keys:
            inputs = entry.get("inputs", {})
            for key in metadata_keys:
                example[key] = inputs.get(key, "")
        if include_metrics:
            example.update(extract_metrics(entry))
        examples.append(example)

    return examples


def to_hf_dataset(entries: list[dict], **kwargs):
    """Convert collected entries to a HuggingFace Dataset.

    Args:
        entries: List of collected trace dicts.
        **kwargs: Passed to to_sft_examples().

    Returns:
        datasets.Dataset with 'messages' and optionally 'tools' columns.
    """
    from datasets import Dataset

    sft_examples = to_sft_examples(entries, **kwargs)
    return Dataset.from_list(sft_examples, on_mixed_types="use_json")
