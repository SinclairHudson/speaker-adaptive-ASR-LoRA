## This script trains a single lora on a subset of librispeech (by speaker ID)
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from peft import get_peft_model, LoraConfig
from dataclasses import dataclass
from tqdm import tqdm

# initialize the first model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# model definition
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# only target the transformer layers, in the feature extractor
# TODO the head probably needs to be trained as well
lora_config = LoraConfig(init_lora_weights="gaussian", target_modules=["k_proj", "q_proj", "v_proj", "out_proj"])
peft_model = get_peft_model(model, lora_config)

train_dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "train.clean", split="validation")

@dataclass
class Hyperparams:
    learning_rate: float = 3e-4
    batch_size: int = 16
    num_epochs: int = 10

hyperparams = Hyperparams()

optim = torch.optim.Adam(peft_model.parameters(), lr=hyperparams.learning_rate)

for epoch in range(hyperparams.num_epochs):
    for i in tqdm(range(0, len(train_dataset), hyperparams.batch_size)):
        batch = train_dataset[i:i+hyperparams.batch_size]
        input_values = processor(batch["speech"], return_tensors="pt", padding="longest")
        with torch.no_grad():
            peft_model(input_values.input_values)
        loss = peft_model(input_values.input_values).loss
        loss.backward()
        optim.step()
        optim.zero_grad()

