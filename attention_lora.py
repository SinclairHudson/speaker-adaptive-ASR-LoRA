import torch
from peft import get_peft_model, LoraConfig


"""
this is a module that sort of acts like a single LoRA, but in reality is K loras, fused using attention.
"""



class AttentionLoRA(torch.nn.Module):
    def __init__(self, K, base_model, selector_network, lora_config, hidden_dim):
        super(AttentionLoRA, self).__init__()
        self.K = K  # number of loras, technically.
        self.selector_network = selector_network
        self.loraconfig = loraconfig
        self.hidden_dim = hidden_dim

        self.template_peft_model = get_peft_model(base_model, lora_config)
        self.keys = torch.nn.Parameter(torch.randn(K, hidden_dim))

        param_list = [(name, p) for (name, p) in self.template_peft_model.named_parameters() if p.requires_grad]
        param_dict = dict(param_list)
        # cloning the parameters of the lora model K times
        param_clones_dict = {}
        for name, p in param_dict:
            param_clones_dict[name] = p.clone().repeat(K)
                # they all have an extra K dimrension




    def merge_loras(self, x):
        # x is batched audio
        queries = self.selector_network(x)
        attention_weights = torch.nn.functional.softmax(torch.matmul(queries, self.keys.T) / torch.sqrt(self.hidden_dim), dim=1)
        for name, p in param_clones_dict:
            # reduce by the attention weights, and then slam it into the actual model
            # this is going to be a [K] x [K x ? x ? ...] matmul
            param_dict[name] = attention_weights * p




    def forward(self, x):
        # x: (seq_len, audio_length)

