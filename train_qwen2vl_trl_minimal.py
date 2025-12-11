#!/usr/bin/env python3
"""
Minimal, official-style TRL finetuning for Qwen2-VL-7B.
NO custom collator. TRL infers vision-language model automatically.
Designed for Perlmutter (A100, bf16, DDP).
"""

import os
import argparse
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a casual but honest reviewer describing your experience at a restaurant including quality of food, service, and environment."
)

USER_TEMPLATE = (
    "You are visiting this restaurant looking for a great overall experience. You have an appropriate rating in mind matching the aura of the photos. "
    "Write a vivid Google review that matches the aura from the photos."
)


# ---------------------------------------------------------------------
# FORMATTER
# ---------------------------------------------------------------------

import hashlib
from PIL import Image
import os


LOCAL_ROOT = "/pscratch/sd/a/abhipa/finalproject/googlelocal_shards"


class Formatter:
    def __init__(self, max_images=3):
        self.max_images = max_images

    def _sha256_path(self, url):
        """Convert an image URL into local (dir, path) based on SHA-256."""
        if not isinstance(url, str):
            return None

        sha = hashlib.sha256(url.encode("utf-8")).hexdigest()  # 64-char hex
        subdir = sha[:2]                                       # first 2 chars
        # Adjust extension if needed (.jpg, .png, or no extension)
        filename = sha + ".jpg"

        full_path = os.path.join(LOCAL_ROOT, subdir, filename)
        return full_path

    def _load_local_image(self, url):
        """Given a URL string, compute sha256, map to local file, load."""
        img_path = self._sha256_path(url)
        if img_path is None:
            return None
        if not os.path.exists(img_path):
            return None
        try:
            return Image.open(img_path).convert("RGB")
        except:
            return None

    def __call__(self, sample):
        msgs = sample["messages"]

        # assistant text
        try:
            assistant_text = msgs[2]["content"][0]["text"]
            if not assistant_text or not isinstance(assistant_text, str):
                return None
        except:
            return None

        # Your original user template
        user_prompt = USER_TEMPLATE

        # ------------------------------------------------------------
        # LOAD IMAGES FROM *LOCAL SHA256 PATHS*
        # ------------------------------------------------------------
        images = []
        for m in msgs:
            for c in m["content"]:
                if c.get("type") == "image":
                    url = c.get("image")
                    img = self._load_local_image(url)
                    if img:
                        images.append(img)

        # Require ALL images for this sample
        if len(images) == 0:
            return None

        images = images[: self.max_images]

        # ------------------------------------------------------------
        # Final messages structure for Qwen2-VL
        # ------------------------------------------------------------
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": (
                    [{"type": "text", "text": user_prompt}]
                    + [{"type": "image", "image": img} for img in images]
                ),
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

        return {"messages": messages, "images": images}

# ---------------------------------------------------------------------
# DATASET WRAPPER
# ---------------------------------------------------------------------

class GoogleLocalDataset(Dataset):
    def __init__(self, raw_ds, formatter, max_valid_samples=None, split_name="train"):
        """
        raw_ds            : HuggingFace raw parquet Dataset
        formatter         : Formatter() object
        max_valid_samples : Stop after collecting this many valid samples
        split_name        : "train" or "eval" (for printing)
        """
        if max_valid_samples is None:
            max_valid_samples = len(raw_ds)

        self.data = []
        checked = 0

        print(f"[INFO] Building {split_name} dataset... target={max_valid_samples}")

        for sample in raw_ds:
            checked += 1
            formatted = formatter(sample)
            if formatted:
                self.data.append(formatted)

            if checked % 5000 == 0:
                print(f"[DEBUG] {split_name}: checked={checked}, valid={len(self.data)}")

            if len(self.data) >= max_valid_samples:
                break

        print(f"[INFO] Final {split_name} samples: {len(self.data)} (checked {checked})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------

def load_model_4bit(local_rank, load_in_4bit=True):
    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
        trust_remote_code=True,
    )
    return model


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--eval-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--local_rank", type=int, default=-1,
                    help="Local rank for distributed training")

    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Before formatter")
    formatter = Formatter(args.max_images)
    print("After formatter")

    train_raw = load_dataset("parquet", data_files={"train": args.train_parquet})["train"]
    eval_raw  = load_dataset("parquet", data_files={"eval":  args.eval_parquet})["eval"]

    train_ds = GoogleLocalDataset(
        train_raw,
        formatter,
        max_valid_samples=70000,
        split_name="train"
    )

    eval_ds = GoogleLocalDataset(
        eval_raw,
        formatter,
        max_valid_samples=1000,
        split_name="eval"
    )
    print(f"Train samples: {len(train_ds)} | Eval samples: {len(eval_ds)}")
    model = load_model_4bit(local_rank, args.load_in_4bit)

    # LoRA config (lightweight, safe)
    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # TRL SFT settings
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=1,

        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,

        bf16=True,
        max_grad_norm=1.0,
        warmup_ratio=0.03,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},

        max_length=None,
        dataset_text_field=None,

        report_to="wandb",
        push_to_hub=False,
        deepspeed="ds_zero3.json"
    )
    
    # ---------------------------------------------------------------------
    # DEBUG: PRINT 2 SAMPLES FROM TRAIN/EVAL
    # ---------------------------------------------------------------------



    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora,
        processing_class=processor,   # IMPORTANT: this activates VLM auto-collator
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
