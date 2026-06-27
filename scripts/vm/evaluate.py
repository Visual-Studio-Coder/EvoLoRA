"""Run the trained LoRA adapter on EvoLoRA eval prompts inside the GPU VM."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel

MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 512
ADAPTER_PATH = "lora_model"
EVALS_PATH = Path("data/evals.json")
RESULTS_PATH = Path("generations/results.json")

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def main() -> None:
    evals = json.loads(EVALS_PATH.read_text(encoding="utf-8"))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    results: dict[str, str] = {}
    for item in evals:
        prompt = PROMPT_TEMPLATE.format(
            instruction=item.get("instruction", ""),
            input=item.get("input", ""),
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(_model_device(model))
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results[str(item["sample_id"])] = _extract_response(decoded)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(results)} generations to {RESULTS_PATH}")


def _model_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "cuda" if torch.cuda.is_available() else "cpu"


def _extract_response(decoded: str) -> str:
    marker = "### Response:"
    if marker in decoded:
        return decoded.rsplit(marker, 1)[-1].strip()
    return decoded.strip()


if __name__ == "__main__":
    main()
