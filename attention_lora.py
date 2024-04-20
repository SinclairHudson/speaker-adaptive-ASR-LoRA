import torch
from peft import get_peft_model, LoraConfig
from torch import nn
from cluster_speakers import compute_fbank
from math import sqrt


"""
this is a module that sort of acts like a single LoRA, but in reality is K loras, fused using attention.
"""

class AttentionLoRA(torch.nn.Module):
    def __init__(self, K, base_model, selector_network, lora_config, hidden_dim):
        super(AttentionLoRA, self).__init__()
        self.K = K  # number of loras, technically.
        self.selector_network = selector_network
        self.lora_config = lora_config
        self.hidden_dim = hidden_dim

        self.template_peft_model = get_peft_model(base_model, lora_config)
        self.keys = torch.nn.Parameter(torch.randn(K, hidden_dim))

        param_list = [(name, p) for (name, p) in self.template_peft_model.named_parameters() if p.requires_grad]
        self.param_dict = dict(param_list)
        # cloning the parameters of the lora model K times
        for param_name in self.param_dict:
            para = self.param_dict[param_name]
            size = para.size()
            safe_name = param_name.replace(".", "-")
            self.register_parameter(f"{safe_name}_dupes", nn.Parameter(para.clone().repeat([K] + [1] * len(size))))
            # self.param_clones_dict[param_name] = )
                # they all have an extra K dimrension
        self.materialized_clones = []


    def merge_loras(self, x):
        # x is batched audio
        fbank = compute_fbank(x)
        outputs = self.selector_network(fbank.unsqueeze(0))
        queries = outputs[-1] if isinstance(outputs, tuple) else outputs
        attention_weights = torch.nn.functional.softmax(torch.matmul(queries, self.keys.T) / sqrt(self.hidden_dim), dim=1)
        sd = self.template_peft_model.state_dict()
        for name in self.param_dict:
            dupes_name = name.replace(".", "-") + "_dupes"
            # reduce by the attention weights, and then slam it into the actual model
            # this is going to be a [K] x [K x ? x ? ...] matmul
            # breakpoint()
            K_para = self.get_parameter(clone_name)
            materialized_clone = torch.sum(attention_weights * K_para.permute(1, 2, 0), dim=-1)

            sd[name].data = materialized_clone
            self.materialized_clones.append(materialized_clone)
            # self.param_dict[name] = torch.sum(attention_weights * K_para.permute(1, 2, 0), dim=-1)

        # ok so the plan is to save the materialized_clone for the backward pass, then clone over the grad that the template
        # peft model gets. after that. call torch.autograd.backward(materialized_clones, cloned_grads)

    def continue_gradients(self):
        pass


    def compute_pair_contrastive_loss(self):
        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(self.keys, self.keys.t())  # K x K matrix
        # Compute pairwise loss matrix
        pairwise_loss_matrix = 0.5 * (1 - similarity_matrix)**2
        # Exclude diagonal elements (self-similarity)
        pairwise_loss_matrix.fill_diagonal_(0)
        # Compute total loss
        total_loss = torch.sum(pairwise_loss_matrix)
        return total_loss


    # batch_size is going to be limited to one!!! unfortunately.
    # this is because the lora needs to be set for a single forward pass.
    # need a custom lora implementation for the batch..
    def forward(self, x):
        self.merge_loras(x.input_values)
        return self.template_peft_model(**x)
