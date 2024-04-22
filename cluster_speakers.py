import wespeaker
import torchaudio.compliance.kaldi as kaldi
import torch
from datasets import load_dataset
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np

speaker_embedding_model = wespeaker.load_model("english")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
speaker_embedding_model.model = speaker_embedding_model.model.to(device)


# this function take nfrom wespeaker
def compute_fbank(wavform, sample_rate=16000, num_mel_bins=80, frame_length=25,
                     frame_shift=10, cmn=True):
       feat = kaldi.fbank(wavform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          sample_frequency=sample_rate)
       if cmn:
           feat = feat - torch.mean(feat, 0)
       return feat

@torch.no_grad()
def compute_speaker_embedding(waveform):
    """ waveform must be of 16000 sample rate"""
    feats = compute_fbank(waveform)
    feats = feats.unsqueeze(0)
    feats = feats.to(device)
    speaker_embedding_model.model.eval()
    outputs = speaker_embedding_model.model(feats)
    outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
    embedding = outputs[0].to(torch.device('cpu'))
    return embedding

def compute_all_speaker_embeddings():
    # TODO could batch this?
    dataset_split = "train.clean.100"
    full_dataset = load_dataset('librispeech_asr', split=dataset_split, cache_dir="/media/sinclair/One Touch/huggingface/datasets")
    full_dataset_embeddings = np.zeros((0, 256))
    for item in tqdm(full_dataset):
        audio = item["audio"]["array"]
        audio = torch.Tensor(audio).unsqueeze(0)
        embedding = compute_speaker_embedding(audio)
        full_dataset_embeddings = np.vstack((full_dataset_embeddings, embedding))
        # in wespeaker, these are similar based on cosine similarity
    np.save(f"data/full_dataset_embeddings_{dataset_split}.npy", full_dataset_embeddings)

def cluster_speakers():
    dataset_split = "train.clean.100"
    K = 8
    embeddings = np.load(f"data/full_dataset_embeddings_{dataset_split}.npy")
    kmeans = KMeans(n_clusters=K, random_state=0).fit(embeddings)
    np.save(f"data/kmeans_assignment_{dataset_split}_K={K}.npy", kmeans.labels_)
    np.save(f"data/kmeans_cluster_centers_{dataset_split}_K={K}.npy", kmeans.cluster_centers_)

if __name__ == "__main__":
    cluster_speakers()
    # compute_all_speaker_embeddings()
    # pass
