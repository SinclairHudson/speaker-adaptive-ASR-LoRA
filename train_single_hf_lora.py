## This script trains a single lora on a subset of librispeech (by speaker ID)
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from peft import get_peft_model, LoraConfig
from dataclasses import dataclass
from tqdm import tqdm
from hyperparams import Hyperparams

def train_single_lora(dataset, lora_config: LoraConfig, base_model: Wav2Vec2ForCTC, hyperparams: Hyperparams):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    peft_model = get_peft_model(base_model, lora_config)
    optim = torch.optim.Adam(peft_model.parameters(), lr=hyperparams.learning_rate)

    # training loop
    for epoch in range(hyperparams.num_epochs):
        for i in tqdm(range(0, len(dataset), hyperparams.batch_size)):
            batch = dataset[i:i+hyperparams.batch_size]
            # TODO need to get this training loop working, collating
            input_values = processor(batch, return_tensors="pt", padding="longest")
            loss = peft_model(input_values.input_values).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
    return peft_model
