import json
import os

from unsloth import FastLanguageModel

# ====================== CONFIG ======================
config_path = "config.json"
evals_path = "data/evals.json"
default_model_name = "unsloth/Phi-3-mini-4k-instruct"
max_seq_length = 2048
load_in_4bit = True
# ====================================================

if not os.path.exists(evals_path):
    print(f"Error: {evals_path} not found.")
    exit(1)

config = {}
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)

model_name = config.get("base_model_id") or default_model_name

print(f"Loading base model for baseline: {model_name}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)

with open(evals_path) as f:
    evals = json.load(f)

print("Running baseline inference...")
for index, item in enumerate(evals, start=1):
    prompt = f"""### Input:
{item["input"]}

### Response:
"""

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    actual_response = decoded.split("### Response:")[-1].strip()

    item["actual"] = actual_response
    item["score"] = None
    print(f"Baseline eval {index}/{len(evals)} complete")

with open(evals_path, "w") as f:
    json.dump(evals, f, indent=4)

print(f"Baseline complete. Outputs saved to {evals_path}")
