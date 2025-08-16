import json
from contextlib import contextmanager
from pathlib import Path

import dspy
import pytest

from dspy_data.wrapper import ScoreAndSaveWrapper

# ----- Test helpers -----


class FakeLM:
    def __init__(self):
        self.history = []


@contextmanager
def fake_context(*, lm):
    """Stand-in for dspy.context(lm=...) that swaps dspy.settings.lm."""
    prev = dspy.settings.lm
    dspy.settings.lm = lm
    try:
        yield
    finally:
        dspy.settings.lm = prev


class _Settings(dict):
    """Dict-like AND attribute-accessible settings object."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


@pytest.fixture(autouse=True)
def stub_dspy(monkeypatch):
    """
    Provide a settings object that matches DSPy's expectations:
    - dict-like: supports .get("callbacks", [])
    - attribute access: .lm used throughout your wrapper/tests
    Replace dspy.context with a simple lm-swapping context manager.
    """
    settings = _Settings(lm=FakeLM(), callbacks=[])

    @contextmanager
    def fake_context(*, lm):
        prev = settings.lm
        settings.lm = lm
        try:
            yield
        finally:
            settings.lm = prev

    monkeypatch.setattr(dspy, "settings", settings, raising=False)
    monkeypatch.setattr(dspy, "context", fake_context, raising=False)
    yield


def read_single_json(dirpath: Path) -> dict:
    files = sorted(dirpath.glob("*.json"))
    assert len(files) == 1, f"expected exactly one json file, got {len(files)}"
    return json.loads(files[0].read_text())


# ----- Tests -----


def test_success_writes_file_and_reward(tmp_path):
    # predictor returns a mapping (so dict(prediction) works)
    def predictor(**kwargs):
        # simulate the LM recording an interaction inside the context
        dspy.settings.lm.history.append(
            {"prompt": "p1", "messages": [{"role": "user", "content": "hi"}], "response": None}
        )
        return {"answer": "ok"}

    def reward_fn(inputs, prediction):
        assert prediction["answer"] == "ok"
        # simple numeric reward
        return 0.7

    wrapper = ScoreAndSaveWrapper(predictor=predictor, output_dir=tmp_path, reward_fn=reward_fn)
    out = wrapper(question="What is up?")

    assert out == {"answer": "ok"}
    data = read_single_json(tmp_path)
    assert data["inputs"] == {"question": "What is up?"}
    assert data["output"] == {"answer": "ok"}
    assert data["reward"] == 0.7
    # trace captured
    assert isinstance(data["trace"], list) and len(data["trace"]) == 1
    assert data["trace"][0]["prompt"] == "p1"
    assert data["trace"][0]["messages"][0]["content"] == "hi"


def test_predictor_exception_still_writes_file(tmp_path, caplog):
    def bad_predictor(**kwargs):
        raise RuntimeError("boom")

    def reward_fn(*args, **kwargs):
        # Should not be called when prediction failed
        raise AssertionError("reward_fn must not be called")

    wrapper = ScoreAndSaveWrapper(predictor=bad_predictor, output_dir=tmp_path, reward_fn=reward_fn)
    out = wrapper(x=1)  # no exception should bubble up

    assert out is None
    data = read_single_json(tmp_path)
    assert data["inputs"] == {"x": 1}
    assert data["output"] is None
    assert data["reward"] is None
    # log contains a warning about predictor failure
    assert any("Predictor failed" in rec.message for rec in caplog.records)


def test_reward_exception_logged_and_reward_none(tmp_path, caplog):
    def predictor(**kwargs):
        return {"answer": "ok"}

    def reward_fn(*args, **kwargs):
        raise ValueError("bad reward")

    wrapper = ScoreAndSaveWrapper(predictor=predictor, output_dir=tmp_path, reward_fn=reward_fn)
    out = wrapper(foo="bar")

    assert out == {"answer": "ok"}
    data = read_single_json(tmp_path)
    assert data["output"] == {"answer": "ok"}
    assert data["reward"] is None
    assert any("Reward function failed" in rec.message for rec in caplog.records)


def test_trace_includes_completion_model_dump(tmp_path):
    class Resp:
        def to_dict(self):
            return {"text": "hello"}

    def predictor(**kwargs):
        # Two interactions: one with response, one without
        dspy.settings.lm.history.append(
            {"prompt": "p1", "messages": [{"role": "user", "content": "u1"}], "response": Resp()}
        )
        dspy.settings.lm.history.append(
            {"prompt": "p2", "messages": [{"role": "assistant", "content": "u2"}], "response": None}
        )
        return {"ok": True}

    wrapper = ScoreAndSaveWrapper(predictor=predictor, output_dir=tmp_path, reward_fn=None)
    wrapper(q="x")

    data = read_single_json(tmp_path)
    assert len(data["trace"]) == 2
    # first has a completion from model_dump()
    assert data["trace"][0]["completion"] == {"text": "hello"}
    # second has None for completion
    assert data["trace"][1]["completion"] is None
