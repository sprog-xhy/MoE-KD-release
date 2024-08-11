import math
import os.path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from termcolor import colored
from transformers import (
    AutoTokenizer, AutoConfig
)

from dschat.utils.utils import load_state_dict_into_model


def color_print(text, color='blue', flush=False):
    print(colored(text, color), flush=flush)


def count_parameters(model):
    trainable_params = 0
    frozen_params = 0

    for param in model.parameters():
        if param.requires_grad:
            trainable_params += torch.numel(param)
        else:
            frozen_params += torch.numel(param)
    total_params = trainable_params + frozen_params
    color_print(
        f"Model params\n"
        f"|---> trainable_params: {trainable_params / (10 ** 6):.2f}M\n"
        f"|---> frozen_params: {frozen_params / (10 ** 6):.2f}M\n"
        f"|---> total_params: {total_params / (10 ** 6):.2f}M\n"
    )
    return trainable_params, frozen_params, total_params


class LoraLinear(nn.Module):
    def __init__(self, input_dim, output_dim, r):
        super().__init__()
        self.r = r
        self.lora_A = nn.Parameter(torch.zeros(self.r, input_dim))
        self.lora_B = nn.Parameter(torch.zeros(output_dim, self.r))
        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(1))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(1))

    def forward(self, x):
        return x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)


class SparseExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp_lora_r = config.mlp_lora_r
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = LoraLinear(self.hidden_dim, self.ffn_dim, self.mlp_lora_r)
        self.w2 = LoraLinear(self.ffn_dim, self.hidden_dim, self.mlp_lora_r)
        self.w3 = LoraLinear(self.hidden_dim, self.ffn_dim, self.mlp_lora_r)
        # self.act_fn = nn.SiLU()
        self.act_fn = nn.LeakyReLU()

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MoeMLP(nn.Module):
    def __init__(self, config, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([SparseExpertMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[-1]

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # return final_hidden_states, router_logits
        return final_hidden_states, routing_weights


class CombinedMoEMLP(nn.Module):
    def __init__(self, config, dense_mlp, num_experts):
        super().__init__()
        self.config = config
        self.dense_mlp = dense_mlp
        self.moe_mlp = MoeMLP(config, num_experts)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]):
        dense_mlp_output = self.dense_mlp(hidden_states)
        sparse_mlp_output, routing_weights = self.moe_mlp(hidden_states)
        output = dense_mlp_output + self.config.mlp_lora_alpha * sparse_mlp_output
        return output


class GPT2MoELoraModelForCausal(transformers.GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if not hasattr(self.config, 'intermediate_size'):
            self.config.intermediate_size = self.config.n_inner if self.config.n_inner is not None else 4 * self.config.hidden_size

    def inject_moe_lora(self, layer_ids: list = (), nums_of_experts: list = ()):
        for layer_id, num_of_expert in zip(layer_ids, nums_of_experts):
            dense_mlp = self.transformer.h[layer_id].mlp
            self.transformer.h[layer_id].mlp = CombinedMoEMLP(self.config, dense_mlp, num_experts=num_of_expert)

    def freeze_model(self, train_mode: str = "only_moe"):
        if train_mode == "only_moe":
            for n, p in self.named_parameters():
                check = any([x in n for x in ['mlp.moe_mlp']])
                p.requires_grad = True if check else False
        elif train_mode == "full":
            self.requires_grad_(True)
        else:
            raise ValueError


def create_moe_lora_model(model_path,
                          num_experts_per_tok=2,
                          mlp_lora_alpha=1,
                          mlp_lora_r=64,
                          moe_layer_ids: list = (),
                          nums_of_experts: list = (),
                          train_mode: str = "full",
                          load_from_moe=False
                          ):
    config = AutoConfig.from_pretrained(model_path)
    config.num_experts_per_tok = num_experts_per_tok
    config.mlp_lora_alpha = mlp_lora_alpha
    config.mlp_lora_r = mlp_lora_r

    if len(moe_layer_ids) != len(nums_of_experts):
        raise ValueError(f"Check your layer_ids={moe_layer_ids}, nums_of_experts={nums_of_experts}")
    if len(moe_layer_ids) > 0 and hasattr(config, "moe_layer_ids"):
        raise NotImplementedError("You must initialize your MoE model from dense causal model.")
    config.moe_layer_ids = moe_layer_ids if len(moe_layer_ids) > 0 else config.moe_layer_ids if hasattr(config, "moe_layer_ids") else []
    config.nums_of_experts = nums_of_experts if len(nums_of_experts) > 0 else config.nums_of_experts if hasattr(config, "nums_of_experts") else []
    if load_from_moe:
        model = GPT2MoELoraModelForCausal(config=config)
        model.inject_moe_lora(config.moe_layer_ids, config.nums_of_experts)
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        error_info = load_state_dict_into_model(model, state_dict)
    else:
        model = GPT2MoELoraModelForCausal.from_pretrained(model_path, config=config, device_map='auto')
        if len(moe_layer_ids) > 0:
            model.inject_moe_lora(config.moe_layer_ids, config.nums_of_experts)
    model.freeze_model(train_mode)
    model = model.cuda()
    return model


def main():
    model_path = "./debug_save_moe"

    config = AutoConfig.from_pretrained(model_path)
    model = create_moe_lora_model(model_path,
                                  train_mode="only_moe",
                                  # moe_layer_ids=[10, 11],
                                  # nums_of_experts=[8, 8],
                                  load_from_moe=True
                                  )
    color_print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text = ['Today is not a bad day!']
    embedding = tokenizer(text, return_tensors='pt')
    model.eval()
    output = model(
        input_ids=embedding.input_ids.cuda(),
        attention_mask=embedding.attention_mask.cuda(),
        labels=embedding.input_ids.cuda(),
    )
    color_print(output.loss, 'green')


if __name__ == "__main__":
    main()
