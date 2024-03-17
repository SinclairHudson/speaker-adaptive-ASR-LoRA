## This script trains a single lora on a subset of librispeech (by speaker ID)
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from peft import get_peft_model, LoraConfig


# initialize the first model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# model definition
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
lora_config = LoraConfig(init_lora_weights="gaussian", target_modules=["k_proj", "q_proj", "v_proj", "out_proj"])
peft_model = get_peft_model(model, lora_config)

train_dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "train.clean", split="validation")


