from unsloth import FastLanguageModel
import torch
import sys

# ====================== CONFIG ======================
max_seq_length = 2048
load_in_4bit = True
# ====================================================

if len(sys.argv) < 2:
    print("Error: Please provide an input prompt.")
    print("Usage: python chat.py \"Your prompt here\" [model_dir]")
    exit(1)

user_input = sys.argv[1]
# Optional 2nd arg selects which model to load: a trained adapter dir
# (e.g. lora_model or adapters/<name>) or a base model name. Defaults to lora_model.
model_dir = sys.argv[2] if len(sys.argv) > 2 else "lora_model"

# 1. Load the fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model) # 2x faster inference

# 2. Format prompt to match the template
prompt = f"""### Input:
{user_input}

### Response:
"""

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

# 3. Generate response
outputs = model.generate(
    **inputs, 
    max_new_tokens=128, 
    use_cache=True,
    pad_token_id=tokenizer.eos_token_id
)

# 4. Decode and print only the response
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
response = decoded.split("### Response:")[-1].strip()

# Marker lets the caller extract just the response from unsloth's noisy stdout.
print("<<<EVOLORA_RESPONSE>>>", flush=True)
print(response)
