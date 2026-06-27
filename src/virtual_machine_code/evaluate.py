from unsloth import FastLanguageModel
import torch
import json
import os

# ====================== CONFIG ======================
evals_path = "data/evals.json"
model_dir = "lora_model"
max_seq_length = 2048
load_in_4bit = True
# ====================================================

if not os.path.exists(evals_path):
    print(f"Error: {evals_path} not found.")
    exit(1)

# 1. Load the fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model) # 2x faster inference

# 2. Read the current evaluation cases
with open(evals_path, "r") as f:
    evals = json.load(f)

# 3. Run inference on each test case
print("Running evaluation inference...")
for item in evals:
    # Simplified prompt template matching the new training template
    prompt = f"""### Input:
{item["input"]}

### Response:
"""
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generate response
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128, 
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and clean output
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract only the generated response (after "### Response:")
    actual_response = decoded.split("### Response:")[-1].strip()
    
    # Save the output and initialize score
    item["actual"] = actual_response
    item["score"] = None  # Local agent will fill this in later

# 4. Save the updated JSON back to the file
with open(evals_path, "w") as f:
    json.dump(evals, f, indent=4)

print(f"Evaluation complete. Outputs saved to {evals_path}")
