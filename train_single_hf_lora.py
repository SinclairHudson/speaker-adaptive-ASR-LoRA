## This script trains a single lora on a subset of librispeech (by speaker ID)
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, DataCollatorCTCWithPadding
from peft import get_peft_model, LoraConfig
from dataclasses import dataclass
from tqdm import tqdm
from hyperparams import Hyperparams



def train_single_lora(dataset, lora_config: LoraConfig, base_model: Wav2Vec2ForCTC, hyperparams: Hyperparams):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


    # def collate(batch):
        # audio_array = [x["array"] for x in batch["audio"]]
        # batch_collated = processor(audio_array, sampling_rate=16_000, return_tensors="pt",
                                   # padding=True, text=batch["text"])

        # return batch_collated

    breakpoint()
    peft_model = get_peft_model(base_model, lora_config)
    optim = torch.optim.Adam(peft_model.parameters(), lr=hyperparams.learning_rate)
    torch.set_num_threads(1)  # required for map, otherwise it hangs
    # dataset_prepped = dataset.map(collate, batch_size=hyperparams.batch_size, batched=True, num_proc=4)

    # training loop
    for epoch in range(hyperparams.num_epochs):
        for i in tqdm(range(0, len(dataset), hyperparams.batch_size)):
            batch = collator(dataset[i:i + hyperparams.batch_size])
            loss = peft_model(input_values=batch.input_values,
                              labels=batch.labels).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
    return peft_model
