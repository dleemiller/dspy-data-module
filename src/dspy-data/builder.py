import json
import logging
import uuid
from pathlib import Path

import dspy

from .dry_run_lm import mock_dspy_lm, TypeAwareDryRunLM
from .wrapper import ScoreAndSaveWrapper

logger = logging.getLogger(__name__)


class DataBuilder(dspy.Module):
    def __init__(self, predictor, output_dir, reward_fn=None, num_threads: int = 8):
        """
        Initializes the DataBuilder.

        Args:
            predictor (dspy.Module): The DSPy predictor module to run.
            output_dir (str): The directory to save the generated JSON files.
            reward_fn (Callable, optional): A function to score predictions.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wrapper = ScoreAndSaveWrapper(predictor, output_dir, reward_fn)
        self.parallel = dspy.Parallel(num_threads=num_threads)

    def forward(self, examples: list[dict] | dict, n: int = 1, *, dry_run:bool=False):
        """
        Generates a dataset by processing examples in parallel.
        """
        if isinstance(examples, dict):
            examples = [examples]
        
        if n > 1 and not dry_run:
            logger.info(f"Generating {n} responses for each of the {len(examples)} unique examples...")
            examples = [ex for ex in examples for _ in range(n)]
        elif n > 1 and dry_run:
            logger.info("In dry_run mode, 'n' is ignored. Generating 1 response per unique example.")

        logger.info(f"Starting dataset generation for {len(examples)} total examples using {num_threads} threads...")

        exec_pairs = [(self.wrapper, ex) for ex in examples]
        
        
        predictions = []
        if dry_run:
            with mock_dspy_lm(TypeAwareDryRunLM()):
                predictions = self.parallel(exec_pairs)
        else:
            predictions = self.parallel(exec_pairs)
        
        logger.info("Dataset generation complete.")
        return [p for p in predictions if p is not None]
