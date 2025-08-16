import dspy

from src.dspy_data.rewards import build_format_reward


def test_chat_adapter_successful_parsing():
    """Test reward function with real ChatAdapter and valid response format"""
    signature = dspy.Signature("question -> answer: str, confidence: float")

    chat_adapter = dspy.adapters.ChatAdapter()
    reward_fn = build_format_reward(signature, chat_adapter)

    # Valid ChatAdapter format
    valid_response = "[[ ## answer ## ]]\nParis\n\n[[ ## confidence ## ]]\n0.95"
    result = reward_fn(valid_response)
    assert result == 1.0


def test_chat_adapter_parsing_failure():
    """Test reward function with real ChatAdapter and invalid response format"""
    signature = dspy.Signature("question -> answer: str, confidence: float")

    chat_adapter = dspy.adapters.ChatAdapter()
    reward_fn = build_format_reward(signature, chat_adapter)

    # Invalid format that ChatAdapter can't parse
    invalid_response = "Just some random text without proper formatting"
    result = reward_fn(invalid_response)
    assert result == 0.0


def test_json_adapter_successful_parsing():
    """Test reward function with real JSONAdapter and valid JSON response"""
    signature = dspy.Signature("question -> answer: str, confidence: float, is_certain: bool")

    json_adapter = dspy.adapters.JSONAdapter()
    reward_fn = build_format_reward(signature, json_adapter)

    # Valid JSON response
    json_response = '{"answer": "Paris", "confidence": 0.95, "is_certain": true}'
    result = reward_fn(json_response)
    assert result == 1.0


def test_json_adapter_parsing_failure():
    """Test reward function with real JSONAdapter and invalid JSON"""
    signature = dspy.Signature("question -> answer: str, confidence: float")

    json_adapter = dspy.adapters.JSONAdapter()
    reward_fn = build_format_reward(signature, json_adapter)

    # Invalid JSON
    invalid_json = '{"answer": "Paris", "confidence":}'
    result = reward_fn(invalid_json)
    assert result == 0.0


def test_default_chat_adapter():
    """Test that ChatAdapter is used by default when no adapter provided"""
    signature = dspy.Signature("question -> answer: str")

    reward_fn = build_format_reward(signature)  # No adapter provided

    # Valid ChatAdapter format
    valid_response = "[[ ## answer ## ]]\nParis"
    result = reward_fn(valid_response)
    assert result == 1.0


def test_missing_output_fields():
    """Test behavior when parsed response is missing some expected fields"""
    signature = dspy.Signature("question -> answer: str, confidence: float, extra_field: str")

    json_adapter = dspy.adapters.JSONAdapter()
    reward_fn = build_format_reward(signature, json_adapter)

    # JSON missing the extra_field
    partial_json = '{"answer": "Paris", "confidence": 0.95}'
    result = reward_fn(partial_json)
    assert result == 0.0


def test_extra_output_fields():
    """Test behavior when parsed response has extra fields beyond what's expected"""
    signature = dspy.Signature("question -> answer: str")

    json_adapter = dspy.adapters.JSONAdapter()
    reward_fn = build_format_reward(signature, json_adapter)

    # JSON with extra fields
    extra_json = '{"answer": "Paris", "confidence": 0.95, "extra": "data"}'
    result = reward_fn(extra_json)
    assert result == 1.0
