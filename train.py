import os
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from PIL import Image

# -----------------------------
# 1. Config
# -----------------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
TRAIN_FILE = "training_ready.jsonl"
VAL_FILE = "validation_ready.jsonl"
OUTPUT_DIR = "./qwen-vl-lora"

# -----------------------------
# 2. Load processor + model
# -----------------------------
processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)

# -----------------------------
# 3. LoRA setup
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# -----------------------------
# 4. Load dataset
# -----------------------------
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

# -----------------------------
# 5. Preprocessing
# -----------------------------
def preprocess(example):
    messages = example["messages"]

    system_msg = messages[0]["content"][0]["text"]

    images = []
    user_texts = []
    for c in messages[1]["content"]:
        if c["type"] == "text":
            user_texts.append(c["text"])
        elif c["type"] == "image":
            # Load actual image file
            img_path = c["image"]
            if os.path.exists(img_path):
                images.append(Image.open(img_path).convert("RGB"))

    user_msg = " ".join(user_texts)
    assistant_msg = messages[2]["content"][0]["text"]

    chat = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {"role": "user", "content": [{"type": "text", "text": user_msg}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]},
    ]

    tokenized = processor.apply_chat_template(
        chat,
        images=images if images else None,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    result = {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0],
    }
    if "pixel_values" in tokenized:
        result["pixel_values"] = tokenized["pixel_values"][0]
    return result

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# -----------------------------
# 6. Data collator
# -----------------------------
data_collator = DataCollatorForSeq2Seq(processor.tokenizer, model=model)

# -----------------------------
# 7. Training setup
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
)

# -----------------------------
# 8. Train
# -----------------------------
if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)