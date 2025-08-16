from contextlib import contextmanager
from typing import get_origin

import dspy


@contextmanager
def mock_dspy_lm(mock_lm_instance):
    """A context manager to temporarily replace the global dspy.settings.lm."""
    original_lm = dspy.settings.lm
    try:
        dspy.settings.configure(lm=mock_lm_instance)
        yield
    finally:
        dspy.settings.configure(lm=original_lm)


class DryRunLM(dspy.LM):
    """
    A mock LM that generates a valid, typed dummy response and correctly
    populates its history.
    """

    def __init__(self):
        super().__init__(model="dry_run_lm")
        self.history = []

    def _get_default_for_type(self, field_type):
        """Generates a Pydantic-compatible default value for a given type."""
        if field_type is str:
            return "[DRY_RUN]"
        if field_type is int:
            return 0
        if field_type is float:
            return 0.0
        if field_type is bool:
            return False
        origin_type = get_origin(field_type)
        if origin_type is list:
            return []
        if origin_type is dict:
            return {}
        return ""

    def __call__(self, prompt, **kwargs):
        signature = kwargs.get("signature")
        dummy_response_text = ""
        if signature:
            for field in signature.output_fields:
                field_name = field.name
                field_type = field.type
                default_value = self._get_default_for_type(field_type)
                dummy_response_text += f"{field_name.capitalize()}: {str(default_value)}\n"

        self.history.append({"prompt": prompt, "response": dummy_response_text})
        return [{"text": dummy_response_text}]

    def basic_request(self, prompt, **kwargs):
        pass
