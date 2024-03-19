from datasets import load_dataset

cv_13 = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train", cache_dir="/media/sinclair/M1/huggingface/datasets")
