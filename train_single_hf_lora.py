## This script trains a single lora on a subset of librispeech (by speaker ID)
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from peft import get_peft_model, LoraConfig, PeftModel
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from hyperparams import Hyperparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_single_lora(name: str, dataset, lora_config: LoraConfig, base_model: Wav2Vec2ForCTC, hyperparams: Hyperparams):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def collate(batch):
        # this is needed because librispeech sucks and has multiple levels to their data dicts
        audio_array = [x["array"] for x in batch["audio"]]
        batch_collated_audio = processor(audio_array, sampling_rate=16_000, return_tensors="pt",
                                   padding=True)
        with processor.as_target_processor():
            # this envrionment literally changes the processor to be the tokenizer instead
            batch_collated_text = processor(batch["text"], return_tensors="pt", padding=True, return_attention_mask=True)

        labels = batch_collated_text["input_ids"].masked_fill(batch_collated_text.attention_mask.ne(1), -100)
        batch_collated_audio["labels"] = labels
        inputs = {k: v.to(device) for k, v in batch_collated_audio.items()}
        return inputs

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.to(device)
    # TODO can we make sure that we're only optimizing the trainable parameters?
    optim = torch.optim.Adam(peft_model.parameters(), lr=hyperparams.learning_rate)

    # split dataset into 10% validation and 90% training, shuffling first
    # shuffling crashes my computer
    # dataset = dataset.shuffle(seed=42)
    val_size = len(dataset) // 10

    # training loop
    for epoch in range(hyperparams.num_epochs):
        accum_loss = 0
        # I know this splitting is scuffed but it's the only way to not run out of memory
        for i in tqdm(range(0, len(dataset) - val_size, hyperparams.batch_size)):
            inputs = collate(dataset[i:i + hyperparams.batch_size])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model_output = peft_model(**inputs)
            model_output.loss.backward()
            optim.step()
            optim.zero_grad()
            accum_loss += model_output.loss.item()
        print(f"Epoch {epoch} average_loss: {accum_loss/len(dataset)}")

        # validation loop
        accum_loss = 0
        true_val_size = 0
        print("Validation")
        with torch.no_grad():
            for i in range(len(dataset) - val_size, len(dataset), hyperparams.batch_size):
                true_val_size += hyperparams.batch_size
                inputs = collate(dataset[i:i + hyperparams.batch_size])
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model_output = peft_model(**inputs)
                accum_loss += model_output.loss.item()
        print("Validation loss: ", accum_loss/true_val_size)
    print("Finished training")
    # save the peft model
    peft_model.save_pretrained(f"peft_models/{name}")
    # loading it looks like this
    # peft_model = PeftModel.from_pretrained(base_model, f"peft_models/{name}")

def train_boring():
    from transformers import AutoProcessor

    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    model.to(device)
    def collate(batch):
        # this is needed because librispeech sucks and has multiple levels to their data dicts
        audio_array = [x["array"] for x in batch["audio"]]
        batch_collated_audio = processor(audio_array, sampling_rate=16_000, return_tensors="pt",
                                   padding=True)
        with processor.as_target_processor():
            # this envrionment literally changes the processor to be the tokenizer instead
            batch_collated_text = processor(batch["text"], return_tensors="pt", padding=True, return_attention_mask=True)

        labels = batch_collated_text["input_ids"].masked_fill(batch_collated_text.attention_mask.ne(1), -100)
        batch_collated_audio["labels"] = labels
        inputs = {k: v.to(device) for k, v in batch_collated_audio.items()}
        return inputs
    # audio file is decoded on the fly
    hyperparams = Hyperparams()
    for i in tqdm(range(0, len(dataset), hyperparams.batch_size)):
        inputs = collate(dataset[i:i + hyperparams.batch_size])
        loss = model(**inputs).loss
        print(loss)


def example_script():
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from datasets import load_dataset
    import torch

    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# audio file is decoded on the fly
    for i in range(len(dataset)):
        inputs = processor(dataset[i]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # transcribe speech
        transcription = processor.batch_decode(predicted_ids)
        transcription[0]

        inputs["labels"] = processor(text=dataset[i]["text"], return_tensors="pt").input_ids

        # compute loss
        loss = model(**inputs).loss
        print(round(loss.item(), 2))

if __name__ == "__main__":
    # example_script()
    train_boring()
