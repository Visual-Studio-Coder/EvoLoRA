from unsloth import FastLanguageModel
import torch
import sys

# ====================== CONFIG ======================
model_dir = "lora_model"
max_seq_length = 2048
load_in_4bit = True
# ====================================================

if len(sys.argv) < 2:
    print("Error: Please provide an input prompt.")
    print("Usage: python chat.py \"Your prompt here\"")
    exit(1)

user_input = sys.argv[1]

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

print(response)
