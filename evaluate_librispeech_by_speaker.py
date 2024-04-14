from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import soundfile as sf
from tqdm import tqdm
import numpy as np
import torch
from evaluate import load
import matplotlib.pyplot as plt
wer_metric = load("wer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_split = "test.clean"
# for dataset_split in ['train.clean.100', 'train.clean.360', 'train.other.500', 'validation.clean', 'validation.other', 'test.clean', 'test.other']:
full_dataset = load_dataset('librispeech_asr', split=dataset_split, cache_dir="/media/sinclair/M1/huggingface/datasets")
# full_dataset = full_dataset.remove_columns(["chapter_id", "audio"])  # too much to store in memory, so drop these and load audio on the fly.
# filtered_dataset = full_dataset.filter(lambda x: x['speaker_id'] == 6930)  # only evaluate on one speaker

# breakpoint()
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)

def evaluate_per_speaker():
    speaker_ids = np.load("data/librispeech_test_clean_speaker_ids.npy")

    data = np.zeros((0, 3))
    for speaker_id in tqdm(speaker_ids):
        speaker_dataset = full_dataset.filter(lambda x: x['speaker_id'] == speaker_id)
        preds = []
        gts = []
        with torch.no_grad():
            for item in tqdm(speaker_dataset):
                input_values = processor(item["audio"]["array"],
                                         sampling_rate=item["audio"]["sampling_rate"],
                                         return_tensors="pt").input_values

                prediction = model(input_values.to(device))
                pred_ids = np.argmax(prediction.logits.cpu(), axis=-1)
                pred_str = processor.batch_decode(pred_ids)[0]
                preds.append(pred_str)
                gts.append(item["text"])
        wer = wer_metric.compute(predictions=preds, references=gts)
        data = np.vstack((data, np.array([speaker_id, len(speaker_dataset), wer])))


    np.save("data/librispeech_test_clean_by_speaker.npy", data)


# plot data on two axes as a scatter_plot
def plot_num_instances_by_wer():
    data = np.load("data/librispeech_test_clean_by_speaker.npy")
    print(data[:, 0])
    plt.scatter(data[:, 1], data[:, 2])
    plt.xlabel("Number of data instances")
    plt.ylabel("Word Error Rate")
    plt.title("Librispeech test.clean by speaker")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # evaluate_per_speaker()
    plot_num_instances_by_wer()

