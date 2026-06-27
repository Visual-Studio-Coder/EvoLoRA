# Unsloth MUST be imported before trl/transformers/peft/datasets so its patches to the
# tokenization pipeline apply. Otherwise datasets tries to dill-pickle the map function and
# hits unsloth's torch-config object -> "cannot pickle 'ConfigModuleInstance'".
from unsloth import FastLanguageModel  # isort: skip  # noqa: E402

import json

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# ====================== LOAD CONFIG ======================
with open("config.json") as f:
    config = json.load(f)

learning_rate = config["learning_rate"]
lora_rank = config["lora_rank"]
lora_alpha = config["lora_alpha"]
num_train_epochs = config["num_train_epochs"]
per_device_train_batch_size = config["per_device_train_batch_size"]

model_name = "unsloth/Phi-3-mini-4k-instruct"
max_seq_length = 2048
load_in_4bit = True
gradient_accumulation_steps = 4
data_file = "data/training_data.jsonl"
# =========================================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_alpha,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

dataset = load_dataset("json", data_files=data_file, split="train")

def formatting_prompts_func(examples):
    texts = []
    for input_text, output in zip(
        examples["input"], examples["output"]
    ):
        text = f"""### Input:
{input_text}

### Response:
{output}"""
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# trl 0.24: dataset_text_field and max_length live on SFTConfig (not SFTTrainer),
# and max_seq_length was renamed to max_length.
training_args = SFTConfig(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=5,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
    save_strategy="no",
    dataset_text_field="text",
    max_length=max_seq_length,
    # Single-process tokenization: trl's default num_proc=30 forks workers that must
    # pickle the map fn + module globals, which include unsloth's patched torch config
    # (ConfigModuleInstance) — unpicklable -> crash. num_proc=1 keeps it in-process.
    dataset_num_proc=1,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

print("Starting training...")
trainer.train()

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("Training complete. Model saved to 'lora_model'")
