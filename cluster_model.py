from peft import get_peft_model, LoraConfig, PeftModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from cluster_speakers import compute_speaker_embedding
import numpy as np
import torch

class ClusterModel():
    def __init__(self, K, device):
        self.device = device
        self.base_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        # load all the peft models
        self.peft_models = []
        for k in range(K):
            peft_model = PeftModel.from_pretrained(self.base_model, f"peft_models/lora_cluster_{k}_of_{K}")
            self.peft_models.append(peft_model.to(device))

        self.cluster_centers = np.load(f"data/kmeans_cluster_centers_train.clean.100_K={K}.npy")
        assert len(self.cluster_centers) == K
        print(f"ClusterModel initialized with {K} clusters, on device {device}")

    @torch.no_grad()
    def __call__(self, raw_input_dict):
        audio = raw_input_dict["audio"]["array"]
        audio = torch.Tensor(audio).unsqueeze(0).to(self.device)
        embedding = compute_speaker_embedding(audio)
        # find the closest cluster centre by L2 distance:
        distances = np.linalg.norm(self.cluster_centers - embedding.cpu().numpy(), axis=1)
        cluster_id = np.argmin(distances)
        peft_model = self.peft_models[cluster_id]
        input_values = self.processor(raw_input_dict["audio"]["array"],
                                 sampling_rate=raw_input_dict["audio"]["sampling_rate"],
                                 return_tensors="pt").input_values
        prediction = peft_model(input_values.to(self.device))
        pred_ids = np.argmax(prediction.logits.cpu(), axis=-1)
        pred_str = self.processor.batch_decode(pred_ids)[0]
        return pred_str


