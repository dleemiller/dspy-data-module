# dspy-data

Generate SFT training data from DSPy programs. Collect traces from tool-calling agents, score them with reward functions, and convert to TRL-compatible datasets.

## Installation

```bash
uv pip install -e .
```

## Quick Start

```python
import dspy
from dspy_data import Collect, load_collected, to_sft_examples

# 1. Define your DSPy program
predictor = dspy.ChainOfThought("question -> answer")
lm = dspy.LM("openai/gpt-4o-mini", api_key="...")
dspy.settings.configure(lm=lm)

# 2. Collect traces with rewards
def reward_fn(inputs, prediction):
    return 1.0 if prediction.answer else 0.0

collect = Collect(
    predictor=predictor,
    output_dir="traces.jsonl",
    reward_fn=reward_fn,
    output_format="jsonl",  # or "json" for individual files
    num_threads=8,
)
collect([{"question": "What is 2+2?"}], n=5)

# 3. Load, filter, and convert to SFT format
entries = load_collected("traces.jsonl")
sft_examples = to_sft_examples(entries, min_reward=0.5)
# -> [{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}]
```

## CLI

```bash
# Inspect collected traces
dspy-data stats traces.jsonl --tool-usage

# Build SFT dataset
dspy-data build-sft traces.jsonl -o dataset.jsonl \
    --min-reward 0.8 \
    --best-of-n \
    --system-prompt "You are a helpful assistant." \
    --tools tools.json
```

## Core Components

### Collect

Runs a DSPy program on examples in parallel, capturing full LM interaction traces, ReAct tool-call trajectories, and reward scores.

```python
from dspy_data import Collect

collect = Collect(
    predictor=your_module,       # any dspy.Module
    output_dir="traces.jsonl",   # JSONL file or directory for JSON files
    reward_fn=your_reward_fn,    # fn(inputs, prediction) -> float
    output_format="jsonl",       # "jsonl" (append-only) or "json" (one file per trace)
    num_threads=8,
)

# Generate 5 candidates per example
results = collect(examples, n=5)

# Dry run: format prompts without calling the LM
results = collect(examples, dry_run=True)
```

**Output format** (each JSONL line):
```json
{
    "inputs": {"question": "..."},
    "trace": [{"messages": [...], "completion": {...}}],
    "trajectory": {"thought_0": "...", "tool_name_0": "search", "observation_0": "..."},
    "output": {"answer": "..."},
    "reward": 0.95
}
```

The `trajectory` field is automatically captured from `dspy.ReAct` modules, providing structured tool call data (thought, tool_name, tool_args, observation per step).

### Loading and Filtering

```python
from dspy_data import load_collected, filter_collected, collected_stats

# Load from JSONL file or directory of JSON files
entries = load_collected("traces.jsonl")

# Filter
good = filter_collected(entries, min_reward=0.8, has_output=True)

# Statistics
stats = collected_stats(entries)
# -> {"count": 100, "reward_mean": 0.82, "tool_calls_total": 300,
#     "tools_used": {"search": 150, "calculate": 150}, ...}
```

### Structured Tool Call Extraction

For ReAct agents, extract clean tool call data from the trajectory:

```python
from dspy_data import extract_tool_calls

for entry in entries:
    calls = extract_tool_calls(entry)
    # -> [{"thought": "...", "tool_name": "search",
    #       "tool_args": {...}, "observation": "..."}, ...]
```

### SFT Dataset Conversion

Convert collected traces to TRL's tool-calling SFT format with proper `tool_calls` and `tool` role messages:

```python
from dspy_data import to_sft_examples, to_hf_dataset, trajectory_to_messages

# Convert with options
examples = to_sft_examples(
    entries,
    min_reward=0.5,
    system_prompt="You are a helpful assistant.",
    tools=[{"type": "function", "function": {"name": "search", ...}}],
    include_thoughts=True,   # include ReAct reasoning in assistant messages
    include_reward=False,
)
# -> [{"messages": [...], "tools": [...]}]

# Or get a HuggingFace Dataset directly
dataset = to_hf_dataset(entries, min_reward=0.5, tools=tool_schemas)
```

**Output message format** (TRL tool-calling compatible):
```python
[
    {"role": "system", "content": "You are..."},
    {"role": "user", "content": "Optimize this code..."},
    {"role": "assistant", "content": "Let me compile first.",
     "tool_calls": [{"id": "call_0", "type": "function",
                     "function": {"name": "evaluate", "arguments": {"code": "..."}}}]},
    {"role": "tool", "tool_call_id": "call_0", "name": "evaluate",
     "content": "Compilation successful.\nTests: 4/4 passed\nSpeedup: 10.2x"},
    {"role": "assistant", "content": "Here is the optimized code..."}
]
```

### Format Reward Functions

Validate that model outputs match a DSPy signature's expected structure:

```python
from dspy_data import build_format_reward

reward_fn = build_format_reward(
    dspy.Signature("question -> answer: str, confidence: float"),
    dspy.adapters.JSONAdapter(),
)

reward_fn('{"answer": "Paris", "confidence": 0.95}')  # 1.0
reward_fn('just some text')                             # 0.0
```

## ReAct Agent Workflow

dspy-data is designed to work with DSPy's `ReAct` module for tool-calling agents. The typical workflow:

```python
import dspy
from dspy_data import Collect, load_collected, collected_stats

# Define tools
def search(query: str) -> str:
    """Search the web for information."""
    return web_search(query)

def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# Create ReAct agent
agent = dspy.ReAct("question -> answer", tools=[search, calculate], max_iters=5)

# Collect traces with reward scoring
def reward(inputs, pred):
    return 1.0 if pred.answer == inputs["expected"] else 0.0

collect = Collect(
    predictor=agent,
    output_dir="traces.jsonl",
    reward_fn=reward,
    output_format="jsonl",
    num_threads=4,
)

# Generate 10 candidates per problem, pick best later
collect(problems, n=10)

# Inspect results
entries = load_collected("traces.jsonl")
print(collected_stats(entries))
```

## Best-of-N Selection

Generate multiple candidates per problem and select the best:

```python
from collections import defaultdict
from dspy_data import load_collected

entries = load_collected("traces.jsonl")

# Group by problem, keep highest reward
by_problem = defaultdict(list)
for e in entries:
    by_problem[e["inputs"]["question"]].append(e)

best = [max(candidates, key=lambda e: e.get("reward", 0))
        for candidates in by_problem.values()]
```

Or use the CLI:

```bash
dspy-data build-sft traces.jsonl -o best.jsonl --best-of-n --min-reward 0.5
```

## API Reference

### `Collect`

```python
class Collect(dspy.Module):
    def __init__(self, predictor, output_dir, reward_fn=None,
                 num_threads=8, output_format="json"):
        """
        Args:
            predictor: DSPy module to run.
            output_dir: Directory for JSON files, or file path for JSONL.
            reward_fn: fn(inputs: dict, prediction) -> float.
            num_threads: Parallel execution threads.
            output_format: "json" or "jsonl".
        """

    def forward(self, examples, n=1, *, dry_run=False):
        """
        Args:
            examples: List of input dicts or single dict.
            n: Candidates to generate per example.
            dry_run: Format prompts without calling the LM.
        Returns:
            List of predictions.
        """
```

### `load_collected`

```python
def load_collected(path: str | Path) -> list[dict]:
    """Load traces from a JSONL file or directory of JSON files."""
```

### `filter_collected`

```python
def filter_collected(entries, *, min_reward=None, max_reward=None, has_output=None) -> list[dict]:
    """Filter entries by reward range or output presence."""
```

### `collected_stats`

```python
def collected_stats(entries: list[dict]) -> dict:
    """Summary statistics: count, reward distribution, tool usage breakdown."""
```

### `extract_tool_calls`

```python
def extract_tool_calls(entry: dict) -> list[dict]:
    """Extract structured tool calls from a ReAct trajectory.
    Returns: [{"thought": str, "tool_name": str, "tool_args": dict, "observation": str}]"""
```

### `trajectory_to_messages`

```python
def trajectory_to_messages(entry, *, system_prompt="", user_prompt_fn=None,
                           final_answer_key="cython_code", include_thoughts=True) -> list[dict]:
    """Convert a ReAct trajectory to OpenAI tool-calling chat format."""
```

### `to_sft_examples`

```python
def to_sft_examples(entries, *, min_reward=None, include_reward=False,
                    tools=None, system_prompt="", user_prompt_fn=None,
                    final_answer_key="cython_code", include_thoughts=True) -> list[dict]:
    """Convert entries to SFT training examples with messages and optional tools column."""
```

### `to_hf_dataset`

```python
def to_hf_dataset(entries, **kwargs) -> Dataset:
    """Convert entries to a HuggingFace Dataset. Kwargs passed to to_sft_examples."""
```

### `build_format_reward`

```python
def build_format_reward(signature, adapter=None):
    """Create a reward function that validates structured output format.
    Returns: fn(response_text) -> float (1.0 if valid, 0.0 if not)."""
```

### CLI: `dspy-data stats`

```
dspy-data stats <input> [--tool-usage]
```

### CLI: `dspy-data build-sft`

```
dspy-data build-sft <input> -o <output> [options]

Options:
  --min-reward FLOAT      Filter by minimum reward
  --best-of-n             Select best trace per problem
  --problem-key KEY       Input field to group by (default: func_name)
  --tools FILE            JSON file with tool schemas for TRL
  --system-prompt TEXT     Prepend system message
  --final-answer-key KEY  Output field for final answer
  --no-thoughts           Exclude ReAct thoughts from messages
  --include-reward        Add reward column to dataset
```
