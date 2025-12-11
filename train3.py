import os
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Seq2SeqTrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from typing import List, Dict

SCRATCH = os.environ.get("SCRATCH", "/tmp")
MODEL_NAME = "/pscratch/sd/j/justiny/huggingface/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
TOKENIZED_PATH = os.path.join(SCRATCH, "tokenized_dataset_patched")
OUTPUT_DIR = os.path.join(SCRATCH, "qwen_vl_lora_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load tokenized dataset
# -----------------------------
print("==> Loading tokenized dataset...")
dataset = load_from_disk(TOKENIZED_PATH)
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
print(f"Loaded dataset: train={len(train_dataset)}, val={len(eval_dataset)}")

# -----------------------------
# Load processor
# -----------------------------
print("==> Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0)

# -----------------------------
# Load model (no device_map="auto" for DDP)
# -----------------------------
print("==> Loading base model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
)

# -----------------------------
# Apply LoRA
# -----------------------------
print("==> Applying LoRA...")
lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    # Keep your original targets; change if you later decide to tune different modules
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# data collator
# -----------------------------
class MyDataCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def _to_tensor(self, x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)

    def __call__(self, features):
        # --- Token tensors ---
        input_ids = [self._to_tensor(f["input_ids"]) for f in features]
        attention_mask = [self._to_tensor(f["attention_mask"]) for f in features]

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0,
        )

        # --- Pixel values ---
        pixel_values = []
        for f in features:
            pv = f["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv, dtype=torch.float32)
            if pv.ndim == 4 and pv.shape[0] == 1:
                pv = pv[0]
            pixel_values.append(pv)
        pixel_values = torch.stack(pixel_values)

        batch_size = len(features)
        image_grid_thw = torch.tensor(
            [[1, 28, 28]] * batch_size,
            dtype=torch.long
        )

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": batch_input_ids.clone(),
        }



data_collator = MyDataCollator(pad_token_id=pad_token_id)

# -----------------------------
# Training Arguments
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=100,
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    num_train_epochs=2,
    ddp_find_unused_parameters=False,
    report_to="none",
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# -----------------------------
# Start training
# -----------------------------
if __name__ == "__main__":
    print("==> Starting training...")
    trainer.train()
    print("==> Training finished!")
