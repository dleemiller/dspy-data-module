import logging

import dspy
from dspy.adapters import Adapter
from dspy.utils.exceptions import AdapterParseError
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def build_format_reward(signature: dspy.Signature, adapter: Adapter = None):
    adapter = adapter or dspy.adapters.ChatAdapter()

    def reward_fn(response):
        try:
            parsed_fields = adapter.parse(signature, response)

            # Check if we got all expected output fields
            expected_fields = set(signature.output_fields.keys())
            parsed_field_names = set(parsed_fields.keys())

            logger.debug(f"  {str(adapter)} parsed fields: {parsed_fields}")

            if expected_fields.issubset(parsed_field_names):
                logger.debug(f"  SUCCESS with {str(adapter)}")
                return 1.0
            else:
                return 0.0

        except (AdapterParseError, ValidationError):
            return 0.0

    return reward_fn
