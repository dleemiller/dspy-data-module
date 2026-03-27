"""
Convert collected traces to SFT training format.

Extracts chat messages from DSPy traces for use with TRL's SFTTrainer
or similar frameworks that expect conversation-format training data.
"""

import logging

logger = logging.getLogger(__name__)


def extract_messages(entry: dict) -> list[dict]:
    """Extract chat messages from a collected trace entry.

    DSPy traces contain the full LM interaction history. This function
    extracts the messages array from each interaction, which are already
    in OpenAI chat format.

    Args:
        entry: A single collected trace dict with keys: inputs, trace, output, reward.

    Returns:
        List of chat messages in OpenAI format (role/content dicts).
    """
    messages = []
    trace = entry.get("trace", [])

    for interaction in trace:
        # Each interaction has 'messages' (the input messages) and 'completion' (the response)
        interaction_messages = interaction.get("messages")
        if interaction_messages:
            messages.extend(interaction_messages)

        completion = interaction.get("completion")
        if completion:
            # Extract assistant response from completion
            choices = completion.get("choices", [])
            for choice in choices:
                msg = choice.get("message", {})
                if msg:
                    messages.append(msg)

    return messages


def to_sft_examples(
    entries: list[dict],
    *,
    min_reward: float | None = None,
    include_reward: bool = False,
) -> list[dict]:
    """Convert collected entries to SFT training examples.

    Args:
        entries: List of collected trace dicts.
        min_reward: Only include entries with reward >= this value.
        include_reward: If True, include reward as metadata in each example.

    Returns:
        List of dicts with 'messages' key (and optionally 'reward').
    """
    examples = []
    for entry in entries:
        reward = entry.get("reward")

        if min_reward is not None and (reward is None or reward < min_reward):
            continue

        messages = extract_messages(entry)
        if not messages:
            continue

        example = {"messages": messages}
        if include_reward:
            example["reward"] = reward
        examples.append(example)

    return examples


def to_hf_dataset(entries: list[dict], **kwargs):
    """Convert collected entries to a HuggingFace Dataset.

    Args:
        entries: List of collected trace dicts.
        **kwargs: Passed to to_sft_examples().

    Returns:
        datasets.Dataset with 'messages' column.
    """
    from datasets import Dataset

    sft_examples = to_sft_examples(entries, **kwargs)
    return Dataset.from_list(sft_examples)
