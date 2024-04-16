from dataclasses import dataclass
from tqdm import tqdm
from datasets import load_dataset
from sklearn.cluster import KMeans
import numpy as np
from train_single_hf_lora import train_single_lora
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from peft import LoraConfig
from hyperparams import Hyperparams

K=4

hyperparams = Hyperparams()
dataset_split = "train.clean.100"
full_dataset = load_dataset('librispeech_asr', split=dataset_split, cache_dir="/media/sinclair/M1/huggingface/datasets")

base_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

kmeans_assignment = np.load(f"data/kmeans_assignment_{dataset_split}.npy")

for k in range(K):
    cluster_dataset = full_dataset.filter(lambda x, indice: kmeans_assignment[indice] == k, with_indices=True)
    print(f"Cluster {k} has {len(cluster_dataset)} samples")

    lora_config = LoraConfig(init_lora_weights="gaussian", target_modules=["k_proj", "q_proj", "v_proj", "out_proj"])
    print(f"training a single lora on cluster {k}")
    trained_peft_model = train_single_lora(cluster_dataset, lora_config, base_model, hyperparams)




