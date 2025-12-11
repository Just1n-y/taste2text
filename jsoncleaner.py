import json

INPUT_FILE = "validation_ready_clean.jsonl"
OUTPUT_FILE = "validation_filtered.jsonl"

def has_local_image(example):
    for msg in example["messages"]:
        for c in msg["content"]:
            if c["type"] == "image" and not c["image"].startswith("https://lh5.googleusercontent.com"):
                return True
    return False

with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    kept = 0
    removed = 0
    for line in f_in:
        example = json.loads(line)
        if has_local_image(example):
            f_out.write(json.dumps(example) + "\n")
            kept += 1
        else:
            removed += 1

print(f"Filtered dataset. Kept: {kept}, Removed: {removed}")
