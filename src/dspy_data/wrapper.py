import json
import logging
import threading
import uuid
from copy import deepcopy
from pathlib import Path

import dspy

logger = logging.getLogger(__name__)


def _serialize(value):
    """Make a value JSON-serializable."""
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize(v) for v in value]
    return str(value)


class ScoreAndSaveWrapper(dspy.Module):
    def __init__(self, predictor, output_dir, reward_fn, *, output_format="json"):
        """Wraps a DSPy predictor to capture traces and save results.

        Args:
            predictor: DSPy module to run.
            output_dir: Directory for JSON files, or path for JSONL file.
            reward_fn: Function(inputs, prediction) -> float.
            output_format: "json" for individual files, "jsonl" for append-only JSONL.
        """
        super().__init__()
        self.predictor = predictor
        self.output_dir = Path(output_dir)
        self.reward_fn = reward_fn
        self.output_format = output_format
        self._jsonl_lock = threading.Lock()

        if output_format == "json":
            self.output_dir.mkdir(parents=True, exist_ok=True)
        elif output_format == "jsonl":
            self.output_dir.parent.mkdir(parents=True, exist_ok=True)

    def forward(self, **kwargs):
        thread_local_lm = deepcopy(dspy.settings.lm)
        thread_local_lm.history = []

        prediction = None
        try:
            with dspy.context(lm=thread_local_lm):
                prediction = self.predictor(**kwargs)
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.warning(f"Predictor failed for example {kwargs}: {e}")

        interaction_history = thread_local_lm.history

        simplified_trace = []
        if interaction_history:
            for interaction in interaction_history:
                response = interaction.get("response")
                simplified_trace.append(
                    {
                        "prompt": interaction.get("prompt"),
                        "messages": interaction.get("messages"),
                        "completion": response.to_dict() if response else None,
                    }
                )

        reward = None
        if self.reward_fn and prediction:
            try:
                reward = self.reward_fn(kwargs, prediction)
            except Exception as e:
                logger.warning(f"Reward function failed for example {kwargs}: {e}")

        # Capture structured trajectory from ReAct modules (tool_name_N, tool_args_N, observation_N)
        trajectory = None
        if prediction is not None:
            raw_traj = getattr(prediction, "trajectory", None)
            if isinstance(raw_traj, dict):
                trajectory = {k: _serialize(v) for k, v in raw_traj.items()}

        output_data = {
            "inputs": kwargs,
            "trace": simplified_trace,
            "trajectory": trajectory,
            "output": dict(prediction) if prediction else None,
            "reward": reward,
        }

        self._save(output_data)
        return prediction

    def _save(self, output_data: dict) -> None:
        if self.output_format == "jsonl":
            line = json.dumps(output_data, default=str) + "\n"
            with self._jsonl_lock, open(self.output_dir, "a") as f:
                f.write(line)
            logger.info(f"Appended entry to: {self.output_dir}")
        else:
            file_path = self.output_dir / f"{uuid.uuid4()}.json"
            file_path.write_text(json.dumps(output_data, indent=2))
            logger.info(f"Saved dataset entry to: {file_path}")
