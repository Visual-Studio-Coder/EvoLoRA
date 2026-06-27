from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json

# ====================== LOAD CONFIG ======================
with open("config.json", "r") as f:
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
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

training_args = TrainingArguments(
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
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
)

print("Starting training...")
trainer.train()

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("Training complete. Model saved to 'lora_model'")
