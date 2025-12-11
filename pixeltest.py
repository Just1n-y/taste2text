from datasets import load_from_disk
import os

SCRATCH = os.environ["SCRATCH"]
ds = load_from_disk(os.path.join(SCRATCH, "tokenized_dataset"))

x = ds["train"][0]

print("pixel_values shape:", 
      None if x["pixel_values"] is None else 
      (len(x["pixel_values"]), len(x["pixel_values"][0]), len(x["pixel_values"][0][0])) )

