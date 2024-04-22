from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import soundfile as sf
from tqdm import tqdm
import numpy as np
import torch
from evaluate import load
import matplotlib.pyplot as plt
from cluster_speakers import compute_speaker_embedding
wer_metric = load("wer")
from cluster_model import ClusterModel
from attention_lora import AttentionLoRA
from peft import LoraConfig
import wespeaker

import matplotlib

font = {
    'size'   : 20
        }

matplotlib.rc('font', **font)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_split = "test.clean"
# for dataset_split in ['train.clean.100', 'train.clean.360', 'train.other.500', 'validation.clean', 'validation.other', 'test.clean', 'test.other']:
full_dataset = load_dataset('librispeech_asr', split=dataset_split, cache_dir="/media/sinclair/One Touch/huggingface/datasets")
# full_dataset = full_dataset.remove_columns(["chapter_id", "audio"])  # too much to store in memory, so drop these and load audio on the fly.
# filtered_dataset = full_dataset.filter(lambda x: x['speaker_id'] == 6930)  # only evaluate on one speaker

# breakpoint()

@torch.no_grad()
def evaluate_attentionLoRA_unconditional():
    preds = []
    gts = []
    base_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    lora_config = LoraConfig(init_lora_weights="gaussian", target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "projection"])

    select_net = wespeaker.load_model("english")
    select_net = select_net.model.to(device)
    model = AttentionLoRA(K=4, base_model=base_model, selector_network=select_net,
                          lora_config=lora_config, hidden_dim=256).to(device)
    model.load_state_dict(torch.load("attention_lora.pt"))

    for item in tqdm(full_dataset):
        input_dict = processor(item["audio"]["array"],
                                 sampling_rate=item["audio"]["sampling_rate"],
                                 text=item["text"],
                                 return_tensors="pt")
        prediction = model(input_dict.to(device))
        pred_ids = np.argmax(prediction.logits.cpu(), axis=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        preds.append(pred_str)
        gts.append(item["text"])
    wer = wer_metric.compute(predictions=preds, references=gts)
    print("over test set, base model averages WER: ", wer)

@torch.no_grad()
def evaluate_base_model_unconditional():
    preds = []
    gts = []
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    for item in tqdm(full_dataset):
        input_values = processor(item["audio"]["array"],
                                 sampling_rate=item["audio"]["sampling_rate"],
                                 return_tensors="pt").input_values

        prediction = model(input_values.to(device))
        pred_ids = np.argmax(prediction.logits.cpu(), axis=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        preds.append(pred_str)
        gts.append(item["text"])
    wer = wer_metric.compute(predictions=preds, references=gts)
    print("over test set, base model averages WER: ", wer)

@torch.no_grad()
def evaluate_cluster_model_unconditional(K=4):
    model = ClusterModel(K=K, device=device)
    preds = []
    gts = []
    with torch.no_grad():
        for item in tqdm(full_dataset):
            pred_str = model(item)
            preds.append(pred_str)
            gts.append(item["text"])
    wer = wer_metric.compute(predictions=preds, references=gts)
    print("over test set, cluster model averages WER: ", wer)

@torch.no_grad()
def evaluate_cluster_model_per_speaker(K=4):
    speaker_ids = np.load("data/librispeech_test_clean_speaker_ids.npy")

    data = np.zeros((0, 3))
    model = ClusterModel(K=K, device=device)
    for speaker_id in tqdm(speaker_ids):
        speaker_dataset = full_dataset.filter(lambda x: x['speaker_id'] == speaker_id)
        preds = []
        gts = []
        with torch.no_grad():
            for item in tqdm(speaker_dataset):
                pred_str = model(item)
                preds.append(pred_str)
                gts.append(item["text"])
        wer = wer_metric.compute(predictions=preds, references=gts)
        data = np.vstack((data, np.array([speaker_id, len(speaker_dataset), wer])))

    np.save(f"data/librispeech_test_clean_cluster_model_by_speaker_K={K}.npy", data)

@torch.no_grad()
def evaluate_per_speaker():
    speaker_ids = np.load("data/librispeech_test_clean_speaker_ids.npy")

    data = np.zeros((0, 3))
    for speaker_id in tqdm(speaker_ids):
        speaker_dataset = full_dataset.filter(lambda x: x['speaker_id'] == speaker_id)
        preds = []
        gts = []
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
    plt.scatter(data[:, 1], data[:, 2])
    # plot average wer for stock model
    plt.axhline(y=0.03383673158855752, color='b', linestyle='-')

    # cluster_model_data = np.load("data/librispeech_test_clean_cluster_model_by_speaker.npy")
    # plt.scatter(cluster_model_data[:, 1], cluster_model_data[:, 2], color='red')
    # plot average wer for cluster model
    # plt.axhline(y=0.03585286062081558, color='r', linestyle='-')


    plt.xlabel("Number of data instances")
    plt.ylabel("Word Error Rate (WER)")
    plt.title("LibriSpeech test.clean by Speaker")


    plt.tight_layout()
    plt.show()

def plot_changes(K=4):
    data = np.load("data/librispeech_test_clean_by_speaker.npy")
    cluster_model_data = np.load(f"data/librispeech_test_clean_cluster_model_by_speaker_K={4}.npy")
    assert np.all(data[:, :2] == cluster_model_data[:, :2])
    plt.scatter(data[:, 1], cluster_model_data[:, 2] - data[:, 2])

    data = np.load("data/librispeech_test_clean_by_speaker.npy")
    cluster_model_data = np.load(f"data/librispeech_test_clean_cluster_model_by_speaker_K={8}.npy")
    assert np.all(data[:, :2] == cluster_model_data[:, :2])
    plt.scatter(data[:, 1], cluster_model_data[:, 2] - data[:, 2])
    # plot zero
    plt.axhline(y=0, color='b', linestyle='-')

    plt.xlabel("Number of data instances")
    plt.ylabel("Word Error Rate change")
    plt.title(f"Librispeech test.clean performance change between base model and ClusterModels, per-speaker")

    plt.tight_layout()
    plt.show()

@torch.no_grad()
def assess_speaker_purity():
    K=4
    speaker_ids = np.load("data/librispeech_test_clean_speaker_ids.npy")
    cluster_centres = np.load("data/kmeans_cluster_centers_train.clean.100.npy")

    cluster_distributions = np.zeros((len(speaker_ids), K))
    for speaker_ind, speaker_id in enumerate(tqdm(speaker_ids)):
        speaker_dataset = full_dataset.filter(lambda x: x['speaker_id'] == speaker_id)
        for item in tqdm(speaker_dataset):
            audio = item["audio"]["array"]
            audio = torch.Tensor(audio).unsqueeze(0)
            embedding = compute_speaker_embedding(audio).numpy()
            embedding_expanded = np.stack([embedding] * K, axis=0)
            # L2 distance, could try cosine?
            distances = np.linalg.norm(cluster_centres - embedding_expanded, axis=1)
            cluster = np.argmin(distances)
            cluster_distributions[speaker_ind, cluster] += 1

        cluster_distributions[speaker_ind] /= np.sum(cluster_distributions[speaker_ind])
        print(cluster_distributions[speaker_ind])

    np.save("data/librispeech_test_clean_speaker_purity.npy", cluster_distributions)

def compute_average_purity():
    purity = np.load("data/librispeech_test_clean_speaker_purity.npy")
    print(np.mean(np.max(purity, axis=1)))

if __name__ == "__main__":
    # evaluate_per_speaker()
    # assess_speaker_purity()
    # compute_average_purity()
    # evaluate_cluster_model_per_speaker()
    # plot_num_instances_by_wer()
    # evaluate_cluster_model_unconditional(K=8)

    # evaluate_cluster_model_per_speaker(K=8)
    # evaluate_base_model_unconditional()
    plot_changes(K=8)
    # plot_num_instances_by_wer()

