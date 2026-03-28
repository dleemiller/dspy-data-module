import logging
from pathlib import Path

import dspy

from .dry_run_lm import DryRunLM, mock_dspy_lm
from .wrapper import ScoreAndSaveWrapper

logger = logging.getLogger(__name__)


class Collect(dspy.Module):
    def __init__(
        self,
        predictor,
        output_dir,
        reward_fn=None,
        num_threads: int = 8,
        output_format: str = "json",
    ):
        """Initializes the data collector.

        Args:
            predictor: The DSPy predictor module to run.
            output_dir: Directory for JSON files, or file path for JSONL.
            reward_fn: Function(inputs, prediction) -> float.
            num_threads: Number of parallel execution threads.
            output_format: "json" for individual files, "jsonl" for append-only JSONL.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_format = output_format

        if output_format == "json":
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.wrapper = ScoreAndSaveWrapper(
            predictor, output_dir, reward_fn, output_format=output_format
        )
        self.parallel = dspy.Parallel(num_threads=num_threads, provide_traceback=True)

    def forward(self, examples: list[dict] | dict, n: int = 1, *, dry_run: bool = False):
        """Generates a dataset by processing examples in parallel."""
        if isinstance(examples, dict):
            examples = [examples]

        if n > 1 and not dry_run:
            logger.info(f"Generating {n} responses for each of the {len(examples)} unique examples...")
            examples = [ex for ex in examples for _ in range(n)]
        elif n > 1 and dry_run:
            logger.info("In dry_run mode, 'n' is ignored. Generating 1 response per unique example.")

        logger.info(f"Starting dataset generation for {len(examples)} total examples...")

        exec_pairs = [(self.wrapper, ex) for ex in examples]

        predictions = []
        if dry_run:
            with mock_dspy_lm(DryRunLM()):
                predictions = self.parallel(exec_pairs)
        else:
            predictions = self.parallel(exec_pairs)

        logger.info("Dataset generation complete.")
        return [p for p in predictions if p is not None]
