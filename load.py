import os
import torch
from datasets import load_dataset
from transformers import AutoProcessor
import time

SCRATCH = os.environ["SCRATCH"]

# -----------------------------
# Force HF caches to $SCRATCH and offline mode
# -----------------------------
os.environ["HF_HOME"] = os.path.join(SCRATCH, "hf_home")                  # transformers cache
os.environ["HF_DATASETS_CACHE"] = os.path.join(SCRATCH, "hf_datasets_cache")  # datasets cache
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_ALLOW_MULTIPLE_PROCESSES"] = "1"  # avoid lock blocking

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "/pscratch/sd/j/justiny/huggingface/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
TRAIN_FILE = "training_filtered.jsonl"
VAL_FILE = "validation_filtered.jsonl"
TOKENIZED_PATH = os.path.join(SCRATCH, "tokenized_dataset")

# -----------------------------
# Load processor
# -----------------------------
print("==> Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("==> Processor loaded")

# -----------------------------
# Load raw dataset with retry (to avoid lock hangs)
# -----------------------------
print("==> Loading raw dataset...")

dataset = None
retry_count = 0
while dataset is None and retry_count < 5:
    try:
        dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    except Exception as e:
        retry_count += 1
        print(f"  Warning: dataset load failed on attempt {retry_count}. Retrying in 5s...")
        time.sleep(5)

if dataset is None:
    raise RuntimeError("Failed to load dataset after multiple retries!")

print(f"==> Dataset loaded. Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

def preprocess(example):
    from PIL import Image

    # Take first image if any
    first_image_path = None
    for msg in example["messages"]:
        if msg["role"] == "user":
            for content in msg["content"]:
                if content["type"] == "image":
                    first_image_path = content["image"]
                    break
            break

    # Prepare text with <image> token if there is an image
    new_messages = []
    for msg in example["messages"]:
        new_msg = {"role": msg["role"], "content": []}
        for c in msg["content"]:
            if c["type"] == "text":
                new_msg["content"].append(c)
        new_messages.append(new_msg)

    # If we have an image, add a placeholder <image> token in the user message
    if first_image_path:
        for msg in new_messages:
            if msg["role"] == "user":
                msg["content"].append({"type": "text", "text": processor.image_token})
                break

    # Apply chat template to get text input
    tokenized = processor.apply_chat_template(
        new_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    result = {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0],
    }

    # Process image if it exists
    if first_image_path:
        img = Image.open(first_image_path).convert("RGB")
        pixel_values = processor(
            text=[processor.image_token],  # include placeholder token
            images=img,
            return_tensors="pt"
        )["pixel_values"][0]
        result["pixel_values"] = pixel_values

    return result

# -----------------------------
# Tokenize dataset
# -----------------------------
print("==> Tokenizing dataset (this may take a while)...")
tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing dataset..."
)

# -----------------------------
# Save tokenized dataset to scratch
# -----------------------------
print(f"==> Saving tokenized dataset to {TOKENIZED_PATH}")
tokenized_dataset.save_to_disk(TOKENIZED_PATH)
print("==> Tokenized dataset saved")
