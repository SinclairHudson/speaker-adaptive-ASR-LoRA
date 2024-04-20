import torch
import wespeaker
from tqdm import tqdm
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from attention_lora import AttentionLoRA
from peft import LoraConfig
from hyperparams import AttentionLoraTrainingParams


def train_attention_lora():
    hp = AttentionLoraTrainingParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_split = "train.clean.100"
    training_dataset = load_dataset('librispeech_asr', split=dataset_split, cache_dir="/media/sinclair/One Touch/huggingface/datasets")

    base_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    lora_config = LoraConfig(init_lora_weights="gaussian", target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "projection"])

    select_net = wespeaker.load_model("english")
    select_net = select_net.model.to(device)
    model = AttentionLoRA(K=4, base_model=base_model, selector_network=select_net,
                          lora_config=lora_config, hidden_dim=256).to(device)


    selector_params = [param for name, param in model.named_parameters() if "selector_network" in name]
    selector_optim = torch.optim.Adam(selector_params, lr=hp.selector_learning_rate)

    cloned_parameters = [param for name, param in model.named_parameters() if "clone" in name]
    cloned_optim = torch.optim.Adam(cloned_parameters, lr=hp.lora_clones_learning_rate)
    for item in tqdm(training_dataset):

        input_dict = processor(item["audio"]["array"],
                                 sampling_rate=item["audio"]["sampling_rate"],
                                 text=item["text"],
                                 return_tensors="pt")

        loss = model(input_dict.to(device)).loss
        loss += model.compute_pair_contrastive_loss()  # add loss so that keys don't collapse
        loss.backward()
        breakpoint()
        cloned_optim.step()
        selector_optim.step()
        model.zero_grad()  # zeros more than just the cloned parameters

if __name__ == "__main__":
    train_attention_lora()
