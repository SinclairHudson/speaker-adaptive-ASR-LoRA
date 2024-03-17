from datasets import load_dataset

dataset = load_dataset("librispeech_asr", split="test.clean", cache_dir="/media/sinclair/M1/huggingface/datasets")

breakpoint()
