"""create_evals is a tool the model calls — it now generates the eval examples."""

from evolora.agent.tools import TOOLS


def test_all_three_are_model_tools():
    names = {t["function"]["name"] for t in TOOLS}
    assert {"create_evals", "add_training_examples", "start_training_model"} <= names


def test_create_evals_tool_generates_eval_examples():
    ce = next(t["function"] for t in TOOLS if t["function"]["name"] == "create_evals")
    props = ce["parameters"]["properties"]
    assert "criteria" in props
    assert "eval_examples" in props
    item = props["eval_examples"]["items"]["properties"]
    assert "prompt" in item and "expected_output" in item
