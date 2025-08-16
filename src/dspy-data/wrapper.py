import logging
from copy import deepcopy

import dspy


logger = logging.getLogger(__name__)

class ScoreAndSaveWrapper(dspy.Module):
    def __init__(self, predictor, output_dir, reward_fn):
        super().__init__()
        self.predictor = predictor
        self.output_dir = output_dir
        self.reward_fn = reward_fn

    def forward(self, **kwargs):
        thread_local_lm = deepcopy(dspy.settings.lm)
        thread_local_lm.history = []

        prediction = None
        try:
            with dspy.context(lm=thread_local_lm):
                prediction = self.predictor(**kwargs)
        except Exception as e:
            logger.info(f"⚠️ Predictor failed for example {kwargs}: {e}")
            pass

        interaction_history = thread_local_lm.history

        simplified_trace = []
        if interaction_history:
            for interaction in interaction_history:
                response = interaction.get("response")
                simplified_trace.append({
                    "prompt": interaction.get("prompt"),
                    "messages": interaction.get("messages"),
                    "completion": response.model_dump() if response else None,
                })

        reward = None
        if self.reward_fn and prediction:
            try:
                reward = self.reward_fn(kwargs, prediction)
            except Exception as e:
                logger.warning(f"Reward function failed for example {kwargs}: {e}")

        output_data = {
            "inputs": kwargs,
            "trace": simplified_trace,
            "output": dict(prediction) if prediction else None,
            "reward": reward,
        }

        file_path = self.output_dir / f"{uuid.uuid4()}.json"
        file_path.write_text(json.dumps(output_data, indent=2))
        logger.info(f"✅ Saved dataset entry to: {file_path}")

        return prediction
