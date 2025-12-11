Taste2Text

Taste2Text is a multimodal model that generates Google-style restaurant reviews directly from an image. The project fine-tunes Qwen2.5-VL-7B using LoRA adapters on a filtered set of Google Maps image–review pairs.

Overview

- Built a restaurant-only multimodal dataset by filtering a large Google Maps review corpus

- Downloaded and validated thousands of images with a parallelized HPC pipeline

- Stored data in Parquet and hashed image files for efficient loading

- Fine-tuned Qwen2.5-VL-7B using LoRA + DeepSpeed ZeRO-3 on 8 A100 GPUs


Results

The fine-tuned model produces realistic, multi-sentence reviews from a single restaurant or food photo. Training was stable, and qualitative outputs closely match real Google Maps reviews. Training was run on Perlmutter supercomputer.

Team: Abhi Patel, Kai Gardner, Justin Yang
