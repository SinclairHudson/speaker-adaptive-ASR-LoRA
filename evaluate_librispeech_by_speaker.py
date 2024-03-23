from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import soundfile as sf
from tqdm import tqdm
import numpy as np
import torch
from evaluate import load
wer_metric = load("wer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_split = "train.clean.100"
full_dataset = load_dataset('librispeech_asr', split=dataset_split, cache_dir="/media/sinclair/M1/huggingface/datasets")
# full_dataset = full_dataset.remove_columns(["chapter_id", "audio"])  # too much to store in memory, so drop these and load audio on the fly.

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)

dataset_by_speaker = {}

# print("splitting the dataset by speaker")
# for item in tqdm(full_dataset):
    # if item['speaker_id'] in dataset_by_speaker:
        # dataset_by_speaker[item['speaker_id']].append(item)
    # else:
        # dataset_by_speaker[item['speaker_id']] = [item]

# Now we have a dictionary of speakers, with each speaker having a list of items

# compute WER for each speaker

# for speaker_id, speaker_dataset in tqdm(dataset_by_speaker.items()):
    # total_wer = 0
    # total_tokens = 0
    # print("evaluating on speaker", speaker_id)
    # for item in tqdm(speaker_dataset):
        # audio_input, sample_rate = sf.read(item["file"])
        # input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

        # prediction = model(input_values).logits

        # wer = wer_metric.compute(predictions=prediction, references=item["sentence"])

predictions_by_speaker = {}

with torch.no_grad():
    for item in tqdm(full_dataset):
        input_values = processor(item["audio"]["array"],
                                 sampling_rate=item["audio"]["sampling_rate"],
                                 return_tensors="pt").input_values

        prediction = model(input_values.to(device))
        pred_ids = np.argmax(prediction.logits.cpu(), axis=-1)
        pred_str = processor.batch_decode(pred_ids)[0]

        if item['speaker_id'] in predictions_by_speaker:
            predictions_by_speaker[item['speaker_id']][0].append(pred_str)
            predictions_by_speaker[item['speaker_id']][1].append(item['text'])
        else:
            predictions_by_speaker[item['speaker_id']] = [[pred_str], [item['text']]]

for speaker in predictions_by_speaker:
    wer = wer_metric.compute(predictions=predictions_by_speaker[speaker][0],
                             references=predictions_by_speaker[speaker][1])
    print("Speaker ", speaker, "WER ", wer)

