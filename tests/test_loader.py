import json

import pytest

from dspy_data.loader import collected_stats, filter_collected, load_collected


@pytest.fixture
def sample_entries():
    return [
        {"inputs": {"q": "a"}, "trace": [], "output": {"answer": "A"}, "reward": 0.8},
        {"inputs": {"q": "b"}, "trace": [], "output": {"answer": "B"}, "reward": 0.3},
        {"inputs": {"q": "c"}, "trace": [], "output": None, "reward": None},
        {"inputs": {"q": "d"}, "trace": [], "output": {"answer": "D"}, "reward": 1.0},
    ]


@pytest.fixture
def json_dir(tmp_path, sample_entries):
    for i, entry in enumerate(sample_entries):
        (tmp_path / f"entry_{i}.json").write_text(json.dumps(entry))
    return tmp_path


@pytest.fixture
def jsonl_file(tmp_path, sample_entries):
    path = tmp_path / "traces.jsonl"
    with open(path, "w") as f:
        for entry in sample_entries:
            f.write(json.dumps(entry) + "\n")
    return path


def test_load_json_dir(json_dir, sample_entries):
    loaded = load_collected(json_dir)
    assert len(loaded) == len(sample_entries)
    assert all(e.get("inputs") for e in loaded)


def test_load_jsonl(jsonl_file, sample_entries):
    loaded = load_collected(jsonl_file)
    assert len(loaded) == len(sample_entries)
    assert loaded[0]["inputs"]["q"] == "a"


def test_load_nonexistent_raises():
    with pytest.raises(ValueError):
        load_collected("/nonexistent/path.txt")


def test_filter_min_reward(sample_entries):
    filtered = filter_collected(sample_entries, min_reward=0.5)
    assert len(filtered) == 2
    assert all(e["reward"] >= 0.5 for e in filtered)


def test_filter_has_output(sample_entries):
    filtered = filter_collected(sample_entries, has_output=True)
    assert len(filtered) == 3
    assert all(e["output"] is not None for e in filtered)


def test_collected_stats(sample_entries):
    stats = collected_stats(sample_entries)
    assert stats["count"] == 4
    assert stats["has_output"] == 3
    assert stats["reward_count"] == 3
    assert stats["reward_mean"] == round((0.8 + 0.3 + 1.0) / 3, 4)


def test_collected_stats_empty():
    assert collected_stats([]) == {"count": 0}
