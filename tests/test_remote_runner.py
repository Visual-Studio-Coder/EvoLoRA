"""RemoteModelRunner: VM-result mapping, fallback, and selection."""

import pytest

from evolora.training.runner import MockModelRunner, RemoteModelRunner, get_runner


def test_get_runner_selects_runner():
    assert isinstance(get_runner("mock"), MockModelRunner)
    assert isinstance(get_runner("vm"), RemoteModelRunner)
    assert isinstance(get_runner("remote"), RemoteModelRunner)
    with pytest.raises(ValueError):
        get_runner("nope")


def test_configured_property():
    assert not RemoteModelRunner(host="", key_path="").configured
    assert not RemoteModelRunner(host="vm", key_path="").configured
    assert RemoteModelRunner(host="vm", key_path="/k").configured


def test_extract_responses_accepts_both_shapes():
    flat = {"a": "x", "b": "y"}
    assert RemoteModelRunner._extract_responses(flat) == flat
    assert RemoteModelRunner._extract_responses({"responses": flat}) == flat
    assert RemoteModelRunner._extract_responses(["not", "a", "dict"]) == {}


@pytest.mark.asyncio
async def test_falls_back_to_mock_when_unconfigured():
    runner = RemoteModelRunner(host="", key_path="")
    prompts = [{"sample_id": "a", "prompt": "p"}]
    out = await runner.run_batch(prompts)
    assert out["a"]  # got a (mock) response so the loop can still run


class _StubVMRunner(RemoteModelRunner):
    """Configured runner whose VM fetch is stubbed (no real SSH)."""

    def __init__(self, results: dict):
        super().__init__(host="vm", user="u", key_path="/k")
        self._results = results

    async def _read_results(self) -> dict:
        return self._results


@pytest.mark.asyncio
async def test_maps_vm_results_to_requested_prompts():
    runner = _StubVMRunner({"a": '{"x":1}', "b": '{"x":2}'})
    prompts = [
        {"sample_id": "a", "prompt": "p"},
        {"sample_id": "b", "prompt": "q"},
        {"sample_id": "c", "prompt": "r"},  # not produced by the VM
    ]
    out = await runner.run_batch(prompts)
    assert out["a"] == '{"x":1}'
    assert out["b"] == '{"x":2}'
    assert out["c"] == ""  # missing on the VM -> empty (scores 0, not a crash)
