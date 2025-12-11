from datasets import load_from_disk
import os

SCRATCH = os.environ["SCRATCH"]
TOKENIZED_PATH = os.path.join(SCRATCH, "tokenized_dataset")

print("Loading dataset...")
ds = load_from_disk(TOKENIZED_PATH)

def add_grid(example):
    # Qwen2.5-VL expects a list of tuples
    example["image_grid_thw"] = [(1, 28, 28)]
    return example

print("Patching train...")
ds["train"] = ds["train"].map(add_grid)

print("Patching validation...")
ds["validation"] = ds["validation"].map(add_grid)

patched_path = TOKENIZED_PATH + "_patched"
print("Saving patched dataset...")
ds.save_to_disk(patched_path)

print("Saved to:", patched_path)
