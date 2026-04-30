# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaDA2MoE model."""

import math
import os
import json
import numpy as np
from typing import List, Callable, Optional, Tuple, Union
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache

try:
    from transformers.masking_utils import create_bidirectional_mask
except ImportError:
    # Fallback for transformers < 4.51
    def create_bidirectional_mask(config, inputs_embeds, attention_mask=None, **kwargs):
        """Create a bidirectional (non-causal) attention mask. If already 4D, pass through."""
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        if attention_mask is not None and attention_mask.dim() == 2:
            # Expand 2D (batch, seq) -> 4D (batch, 1, 1, seq) for SDPA
            extended = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
            extended = (1.0 - extended) * torch.finfo(inputs_embeds.dtype).min
            return extended
        return attention_mask


from transformers.modeling_outputs import (
    MoeModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

try:
    from transformers.modeling_rope_utils import dynamic_rope_update
except ImportError:
    # Fallback for transformers < 4.51: no-op decorator (default rope doesn't need dynamic update)
    def dynamic_rope_update(func):
        return func


from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

try:
    from transformers.processing_utils import Unpack
    from transformers.utils import TransformersKwargs
except ImportError:
    # Fallback for transformers < 4.51
    from typing import Any

    TransformersKwargs = Any

    def Unpack(x):
        return x  # type: ignore


from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_llada2uni_moe import LLaDA2MoeConfig
from transformers.generation.utils import GenerationMixin


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LLaDA2MoeConfig"

_CACHE_HAS_LAYERS = hasattr(DynamicCache(), "layers") and not hasattr(
    DynamicCache(), "key_cache"
)


def _cache_num_layers(cache):
    if _CACHE_HAS_LAYERS:
        return len(cache.layers)
    return len(cache.key_cache)


def _cache_get_keys(cache, layer_idx):
    if _CACHE_HAS_LAYERS:
        return cache.layers[layer_idx].keys
    return cache.key_cache[layer_idx]


def _cache_get_values(cache, layer_idx):
    if _CACHE_HAS_LAYERS:
        return cache.layers[layer_idx].values
    return cache.value_cache[layer_idx]


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _compute_confidence_scores(logits, x0, mask_index, remasking, *, opt_softmax=False):
    if remasking == "random":
        scores = torch.full(x0.shape, -np.inf, device=x0.device, dtype=logits.dtype)
        if mask_index.any():
            scores[mask_index] = torch.rand_like(
                scores[mask_index].to(torch.float32)
            ).to(logits.dtype)
        return scores
    if remasking not in ("low_confidence", "top_k_margin", "neg_entropy"):
        raise NotImplementedError(
            f"Remasking strategy '{remasking}' is not implemented."
        )
    if opt_softmax:
        masked_logits = logits[mask_index]
        scores = torch.full(x0.shape, -np.inf, device=x0.device, dtype=logits.dtype)
        if masked_logits.numel() == 0:
            return scores
        p = F.softmax(masked_logits.to(torch.float32), dim=-1).to(logits.dtype)
        if remasking == "low_confidence":
            chosen = x0[mask_index].unsqueeze(-1)
            masked_scores = torch.gather(p, dim=-1, index=chosen).squeeze(-1)
        elif remasking == "top_k_margin":
            if p.shape[-1] < 2:
                masked_scores = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)
            else:
                sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                masked_scores = sorted_probs[..., 0] - sorted_probs[..., 1]
        else:
            epsilon = 1e-10
            entropy = -torch.sum(p * torch.log(p + epsilon), dim=-1)
            max_entropy = float(np.log(p.shape[-1])) if p.shape[-1] > 1 else 1.0
            masked_scores = 1.0 - (entropy / max_entropy)
        scores[mask_index] = masked_scores
        return scores
    p = F.softmax(logits.to(torch.float32), dim=-1).to(logits.dtype)
    if remasking == "low_confidence":
        scores_all = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "top_k_margin":
        if p.shape[-1] < 2:
            scores_all = torch.zeros_like(p[..., 0])
        else:
            sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
            scores_all = sorted_probs[..., 0] - sorted_probs[..., 1]
    else:
        epsilon = 1e-10
        entropy = -torch.sum(p * torch.log(p + epsilon), dim=-1)
        max_entropy = float(np.log(p.shape[-1])) if p.shape[-1] > 1 else 1.0
        scores_all = 1.0 - (entropy / max_entropy)
    return torch.where(mask_index, scores_all, torch.full_like(scores_all, -np.inf))


def get_transfer_index_bd_adaptive(
    logits,
    mask_index,
    x,
    block_end,
    temperature,
    top_p,
    top_k,
    remasking,
    *,
    steps_left,
    minimal_topk=1,
    opt_softmax=False,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    if top_p is not None and top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits_with_noise, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        mask = torch.zeros_like(logits_with_noise, dtype=torch.bool)
        mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits_with_noise = logits_with_noise.masked_fill(
            mask, torch.finfo(logits_with_noise.dtype).min
        )
    if top_k is not None:
        top_k_val = min(top_k, logits_with_noise.size(-1))
        indices_to_remove = (
            logits_with_noise
            < torch.topk(logits_with_noise, top_k_val)[0][..., -1, None]
        )
        logits_with_noise = logits_with_noise.masked_fill(
            indices_to_remove, torch.finfo(logits_with_noise.dtype).min
        )
    x0 = torch.argmax(logits_with_noise, dim=-1)
    confidence = _compute_confidence_scores(
        logits, x0, mask_index, remasking, opt_softmax=opt_softmax
    )
    if block_end is not None:
        confidence[:, block_end:] = -np.inf
    x0 = torch.where(mask_index, x0, x)
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    steps_left = max(1, int(steps_left))
    mask_counts = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        m = int(mask_counts[j].item())
        if m <= 0:
            continue
        target_k = int(math.ceil(m / float(steps_left)))
        target_k = max(int(minimal_topk), target_k)
        target_k = min(target_k, m)
        _, select_index = torch.topk(confidence[j], k=target_k)
        transfer_index[j, select_index] = True
    return x0, transfer_index


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class LLaDA2MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LLaDA2MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LLaDA2MoeRMSNorm)


class LLaDA2MoeRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for register_buffer

    def __init__(self, config: LLaDA2MoeConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: LLaDA2MoeConfig = None,
        device=None,
        seq_len: int = None,
    ):
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class LLaDA2MoeMLP(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LLaDA2MoeGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.routed_scaling_factor = config.routed_scaling_factor

        self.register_buffer("expert_bias", torch.zeros(self.num_experts))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(
        self,
        scores: torch.Tensor,
    ):
        num_tokens, _ = scores.size()
        # Organize the experts into groups
        group_scores = (
            scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask the experts based on selection groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)

        return probs, top_indices

    def forward(self, hidden_states):
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32)
        )

        scores = torch.sigmoid(logits.float()).type_as(logits)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = (
            scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
            if self.top_k > 1
            else scores
        )
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, logits


class LLaDA2MoeSparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self._setup_experts()
        self.gate = LLaDA2MoeGate(config)
        if config.num_shared_experts is not None:
            self.shared_experts = LLaDA2MoeMLP(
                config=config,
                intermediate_size=config.moe_intermediate_size
                * config.num_shared_experts,
            )

    def _setup_experts(self):
        self.experts = nn.ModuleList(
            [
                LLaDA2MoeMLP(
                    config=self.config,
                    intermediate_size=self.config.moe_intermediate_size,
                )
                for _ in range(self.config.num_experts)
            ]
        )

    def forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight, router_logits = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(bsz, seq_len, h)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(
                bsz, seq_len, h
            )
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y, (
            router_logits.view(bsz, seq_len, -1),
            topk_idx.view(bsz, seq_len, -1),
        )

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens_tensor in enumerate(tokens_per_expert):
            num_tokens = num_tokens_tensor.item()
            if num_tokens == 0:
                continue
            end_idx = start_idx + num_tokens
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out.to(x.device))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->LLaDA2Moe
class LLaDA2MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LLaDA2MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        partial_rotary_factor = (
            config.partial_rotary_factor
            if hasattr(config, "partial_rotary_factor")
            else 1.0
        )
        self.rope_dim = int(self.head_dim * partial_rotary_factor)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )

        if self.config.use_qk_norm:
            self.query_layernorm = LLaDA2MoeRMSNorm(
                self.head_dim, eps=config.rms_norm_eps
            )
            self.key_layernorm = LLaDA2MoeRMSNorm(
                self.head_dim, eps=config.rms_norm_eps
            )
        self.dense = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias
        )
        self.sliding_window = getattr(config, "sliding_window", None)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]

        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(
            bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)

        return attn_output, attn_weights, past_key_value


class LLaDA2MoeDecoderLayer(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attention = LLaDA2MoeAttention(config=config, layer_idx=layer_idx)

        self.mlp = (
            LLaDA2MoeSparseMoeBlock(config)
            if (
                config.num_experts is not None
                and layer_idx >= config.first_k_dense_replace
            )
            else LLaDA2MoeMLP(config=config, intermediate_size=config.intermediate_size)
        )
        self.input_layernorm = LLaDA2MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LLaDA2MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


LLADA2MOE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`LLaDA2MoeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaDA2Moe Model outputting raw hidden-states without any specific head on top.",
    LLADA2MOE_START_DOCSTRING,
)
class LLaDA2MoePreTrainedModel(PreTrainedModel):
    config_class = LLaDA2MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLaDA2MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, LLaDA2MoeGate):
            nn.init.normal_(module.weight, mean=0.0, std=std)


LLADA2MOE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.
            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.
            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaDA2Moe Model outputting raw hidden-states without any specific head on top.",
    LLADA2MOE_START_DOCSTRING,
)
class LLaDA2MoeModel(LLaDA2MoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LLaDA2MoeDecoderLayer`]
    Args:
        config: LLaDA2MoeConfig
    """

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LLaDA2MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flex_attention = config._attn_implementation == "flex_attention"
        self.norm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LLaDA2MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @add_start_docstrings_to_model_forward(LLADA2MOE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
            position_ids = position_ids.unsqueeze(0)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_logits,
                ]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class LLaDA2MoeModelLM(LLaDA2MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.model = LLaDA2MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLADA2MOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer
        >>> model = LLaDA2MoeForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            **kwargs,
        )

        loss = None
        aux_loss = None
        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if labels is not None:
            # LLaDA2.0 will use same label position logits
            shift_logits = logits
            shift_labels = labels
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        token_type_ids=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = (
                    past_key_values.get_max_length()
                    if hasattr(past_key_values, "get_max_length")
                    else past_key_values.get_max_cache_shape()
                )
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    @staticmethod
    def _top_k_logits(logits, k):
        if k is None or k <= 0:
            return logits
        else:
            values, _ = torch.topk(logits, k)
            min_values = values[..., -1, None]
            return torch.where(
                logits < min_values, torch.full_like(logits, float("-inf")), logits
            )

    @staticmethod
    def _top_p_logits(logits, p):
        if p is None or p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(
            torch.full_like(logits, False, dtype=torch.bool),
            -1,
            sorted_indices,
            sorted_mask,
        )
        return logits.masked_fill(mask_indices, float("-inf"))

    def _sample_with_temperature_topk_topp(
        self, logits, temperature=1.0, top_k=0, top_p=1.0
    ):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        if temperature == 0.0:
            token = torch.argmax(logits, dim=-1, keepdim=True)
            probs = F.softmax(logits, dim=-1)
            token_prob = torch.gather(probs, -1, token)
            return token.view(*orig_shape), token_prob.view(*orig_shape)

        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return token.view(*orig_shape), token_prob.view(*orig_shape)

    @staticmethod
    def _get_num_transfer_tokens(block_length, steps):
        if steps == 0:
            return torch.tensor([], dtype=torch.int64)
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.full((steps,), base, dtype=torch.int64)
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens

    # ================================================================
    # Sprint acceleration helpers
    # ================================================================

    @staticmethod
    def _ensure_dynamic_cache(past_key_values):
        if isinstance(past_key_values, DynamicCache):
            return past_key_values
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(past_key_values)
        cache = DynamicCache()
        for layer_kv in past_key_values:
            k, v = layer_kv[0], layer_kv[1]
            cache.update(k, v, len(cache))
        return cache

    @staticmethod
    def _sprint_compute_prefix_confidence(logits, prefix_len):
        if prefix_len <= 0:
            return None
        prefix_logits = logits[:, :prefix_len, :].to(torch.float32)
        max_logits = prefix_logits.max(dim=-1).values
        log_z = torch.logsumexp(prefix_logits, dim=-1)
        return torch.exp(max_logits - log_z)

    @staticmethod
    def _sprint_shallow_copy_cache(cache):
        if cache is None:
            return None
        cache_copy = DynamicCache()
        for layer_idx in range(_cache_num_layers(cache)):
            cache_copy.update(
                _cache_get_keys(cache, layer_idx),
                _cache_get_values(cache, layer_idx),
                layer_idx,
            )
        return cache_copy

    def _sprint_prune_cache(
        self,
        past_key_values,
        query_block,
        prefix_len,
        block_length,
        keep_ratio=0.5,
        token_confidence=None,
        confidence_alpha=1.0,
        valid_prefix_mask=None,
        prefix_ids=None,
        image_token_offset=None,
        image_keep_ratio=None,
        text_keep_ratio=None,
    ):
        if image_token_offset is None:
            image_token_offset = getattr(self.config, "image_token_offset", 157184)
        pruned_cache = DynamicCache()
        n_layers = _cache_num_layers(past_key_values)
        alpha = float(max(0.0, min(1.0, confidence_alpha)))

        pin_mask = None
        n_pinned = 0
        if prefix_ids is not None and prefix_len > 0:
            ids = prefix_ids[0, :prefix_len]
            is_text = ids < image_token_offset
            is_image = ~is_text
            pin_mask = torch.zeros(prefix_len, dtype=torch.bool, device=ids.device)
            if text_keep_ratio is None or text_keep_ratio >= 1.0:
                pin_mask |= is_text
            if image_keep_ratio is not None and image_keep_ratio >= 1.0:
                pin_mask |= is_image
            n_pinned = int(pin_mask.sum().item())

        for layer_idx in range(n_layers):
            k_full = _cache_get_keys(past_key_values, layer_idx)
            v_full = _cache_get_values(past_key_values, layer_idx)
            k_prefix = k_full[:, :, :prefix_len, :]
            v_prefix = v_full[:, :, :prefix_len, :]

            valid_mask = (
                valid_prefix_mask[:, :prefix_len].to(torch.bool)
                if valid_prefix_mask is not None
                else None
            )

            if prefix_len == 0:
                pruned_cache.update(k_prefix, v_prefix, layer_idx)
                continue

            if (keep_ratio >= 1.0 or n_pinned >= prefix_len) and valid_mask is None:
                pruned_cache.update(k_prefix, v_prefix, layer_idx)
                continue

            importance = k_prefix.norm(dim=-1).mean(dim=1)
            if valid_mask is not None:
                importance = importance.masked_fill(~valid_mask, float("-inf"))
            if pin_mask is not None:
                importance = importance.masked_fill(
                    pin_mask.unsqueeze(0), float("+inf")
                )

            if (
                token_confidence is not None
                and alpha < 1.0
                and token_confidence.shape[-1] == prefix_len
            ):
                if valid_mask is None:
                    importance_mean = importance.mean(dim=-1, keepdim=True).clamp_min(
                        1e-6
                    )
                    normalized_importance = importance / importance_mean
                else:
                    valid_float = valid_mask.to(importance.dtype)
                    masked_sum = importance.masked_fill(~valid_mask, 0.0).sum(
                        dim=-1, keepdim=True
                    )
                    valid_count = valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
                    importance_mean = (masked_sum / valid_count).clamp_min(1e-6)
                    normalized_importance = torch.where(
                        valid_mask,
                        importance / importance_mean,
                        torch.zeros_like(importance),
                    )
                confidence = token_confidence.to(normalized_importance.dtype)
                importance = alpha * normalized_importance + (1.0 - alpha) * confidence
                if valid_mask is not None:
                    importance = importance.masked_fill(~valid_mask, float("-inf"))

            base_keep_num = (
                prefix_len
                if keep_ratio >= 1.0
                else max(1, int(prefix_len * keep_ratio))
            )
            base_keep_num = max(base_keep_num, n_pinned)
            if valid_mask is not None:
                max_keep = int(valid_mask.sum(dim=-1).min().item())
                if max_keep <= 0:
                    pruned_cache.update(
                        k_prefix[:, :, :0, :], v_prefix[:, :, :0, :], layer_idx
                    )
                    continue
                keep_num = min(base_keep_num, max_keep)
            else:
                keep_num = base_keep_num

            _, keep_indices = torch.topk(importance, k=keep_num, dim=-1)
            keep_indices, _ = keep_indices.sort(dim=-1)
            n_kv_heads = k_prefix.size(1)
            idx_exp = keep_indices.unsqueeze(1).expand(-1, n_kv_heads, -1)
            k_pruned = torch.gather(
                k_prefix, 2, idx_exp.unsqueeze(-1).expand(-1, -1, -1, k_prefix.size(-1))
            )
            v_pruned = torch.gather(
                v_prefix, 2, idx_exp.unsqueeze(-1).expand(-1, -1, -1, v_prefix.size(-1))
            )
            pruned_cache.update(k_pruned, v_pruned, layer_idx)

        return pruned_cache

    @staticmethod
    def _split_cache_by_batch(cache):
        cond_cache = DynamicCache()
        uncond_cache = DynamicCache()
        for layer_idx in range(_cache_num_layers(cache)):
            k = _cache_get_keys(cache, layer_idx)
            v = _cache_get_values(cache, layer_idx)
            cond_cache.update(k[0:1], v[0:1], layer_idx)
            uncond_cache.update(k[1:2], v[1:2], layer_idx)
        return cond_cache, uncond_cache

    # ================================================================
    # Block-diffusion generation methods
    # ================================================================

    @torch.no_grad()
    def generate_bd(
        self,
        data: Optional[dict] = None,
        temperature: float = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = True,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        eos_id: int = 156892,
        mask_id: int = 156895,
        use_sprint: bool = False,
        remasking: str = "low_confidence",
        keep_ratio: float = 0.7,
        cache_warmup_steps: int = 2,
        confidence_alpha: float = 0.5,
        image_keep_ratio: Optional[float] = None,
        text_keep_ratio: Optional[float] = None,
        show_progress: bool = False,
    ):
        r"""
        Generate **text** tokens using block-wise iterative refinement (block diffusion).

        The method creates a full-length template filled with ``mask_id``, then processes it
        block-by-block from left to right. Within each block, ``steps`` denoising iterations
        progressively replace ``mask_id`` tokens with real tokens based on model confidence.
        A block-diagonal causal attention mask ensures each block can attend to all preceding
        blocks but not future ones.

        Args:
            data (`dict`):
                Must contain ``"input_ids"`` — a ``(1, prompt_length)`` tensor of prompt tokens.
            temperature (`float`, *optional*, defaults to 0.0):
                Sampling temperature. 0.0 means greedy decoding.
            block_length (`int`, *optional*, defaults to 32):
                Number of tokens per generation block.
            steps (`int`, *optional*, defaults to 32):
                Denoising iterations per block. Capped at ``gen_length // minimal_topk``.
            gen_length (`int`, *optional*, defaults to 2048):
                Maximum number of tokens to generate (excluding the prompt).
            top_p (`float`, *optional*):
                Nucleus-sampling probability cutoff.
            top_k (`int`, *optional*):
                Top-k filtering count.
            eos_early_stop (`bool`, *optional*, defaults to True):
                Stop as soon as an ``eos_id`` token is confirmed.
            minimal_topk (`int`, *optional*, defaults to 1):
                Lower-bounds the number of tokens transferred per step; also caps ``steps``.
            threshold (`float`, *optional*, defaults to 0.95):
                Confidence threshold — a sampled token is accepted only when its probability
                exceeds this value; otherwise the top-confidence tokens are chosen.
            eos_id (`int`, *optional*, defaults to 156892):
                End-of-sequence token ID.
            mask_id (`int`, *optional*, defaults to 156895):
                Placeholder token ID for positions yet to be generated.
            use_sprint (`bool`, *optional*, defaults to False):
                Enable Sprint acceleration via KV cache pruning. When True, the prefix KV
                cache is computed once during warmup steps and then pruned for reuse.
            remasking (`str`, *optional*, defaults to ``"low_confidence"``):
                Token remasking strategy used in Sprint mode. One of ``"low_confidence"``,
                ``"random"``, ``"neg_entropy"``, ``"top_k_margin"``.
            keep_ratio (`float`, *optional*, defaults to 0.7):
                Fraction of prefix KV cache entries to retain after pruning (Sprint mode).
            cache_warmup_steps (`int`, *optional*, defaults to 2):
                Number of full forward passes before switching to cached Sprint mode.
            confidence_alpha (`float`, *optional*, defaults to 0.5):
                Blending weight between KV importance and token confidence for pruning.
            image_keep_ratio (`float`, *optional*, defaults to None):
                Fraction of image-token KV entries to retain during pruning. ``1.0``
                pins all image tokens. ``None`` falls back to global ``keep_ratio``.
            text_keep_ratio (`float`, *optional*, defaults to None):
                Fraction of text-token KV entries to retain during pruning. ``1.0``
                pins all text tokens (default legacy behavior when ``None``).

        Returns:
            `torch.Tensor` of shape ``(1, output_length)``: generated token IDs (prompt + generated),
            truncated at the first ``eos_id``.
        """
        steps = min(steps, gen_length // minimal_topk)
        input_ids = data["input_ids"]

        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
        ).bool()

        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()

        prefill_blocks = prompt_length // block_length
        denoising_steps_per_block = steps
        num_transfer_tokens_schedule = self._get_num_transfer_tokens(
            block_length, denoising_steps_per_block
        )

        num_gen_blocks = num_blocks - prefill_blocks
        pbar = (
            tqdm(total=num_gen_blocks, desc="Generating text blocks", unit="block")
            if show_progress
            else None
        )

        for num_block in range(prefill_blocks, num_blocks):
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    block=f"{num_block - prefill_blocks + 1}/{num_gen_blocks}"
                )
            current_window_end = (num_block + 1) * block_length
            prefix_len = current_window_end - block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[
                :, :, :current_window_end, :current_window_end
            ]
            cur_position_ids = position_ids[:, :current_window_end]

            pruned_cache = None

            for step in range(denoising_steps_per_block):
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if active_block_mask.sum() == 0:
                    break

                use_cache_this_step = use_sprint and (step == cache_warmup_steps - 1)
                use_pruned = (
                    use_sprint and (step >= cache_warmup_steps) and (prefix_len > 0)
                )

                if use_pruned and pruned_cache is not None:
                    pruned_prefix_len = _cache_get_keys(pruned_cache, 0).shape[2]
                    prefix_attn = torch.ones(
                        1,
                        1,
                        block_length,
                        pruned_prefix_len,
                        dtype=torch.bool,
                        device=self.device,
                    )
                    block_self_attn = cur_attn_mask[
                        :, :, -block_length:, -block_length:
                    ]
                    sprint_attn_mask = torch.cat([prefix_attn, block_self_attn], dim=-1)

                    logits = self.forward(
                        cur_x[:, -block_length:],
                        attention_mask=sprint_attn_mask,
                        position_ids=position_ids[:, prefix_len:current_window_end],
                        past_key_values=self._sprint_shallow_copy_cache(pruned_cache),
                        use_cache=False,
                    ).logits
                    active_logits = logits[:, :, :]
                else:
                    outputs = self.forward(
                        cur_x,
                        attention_mask=cur_attn_mask,
                        position_ids=cur_position_ids,
                        use_cache=use_cache_this_step,
                    )
                    logits = outputs.logits
                    active_logits = logits[:, -block_length:, :]

                    if use_cache_this_step and outputs.past_key_values is not None:
                        prefix_confidence = self._sprint_compute_prefix_confidence(
                            logits, prefix_len
                        )
                        pruned_cache = self._sprint_prune_cache(
                            self._ensure_dynamic_cache(outputs.past_key_values),
                            None,
                            prefix_len,
                            block_length,
                            keep_ratio,
                            prefix_confidence,
                            confidence_alpha,
                            prefix_ids=cur_x[:, :prefix_len],
                            image_keep_ratio=image_keep_ratio,
                            text_keep_ratio=text_keep_ratio,
                        )
                        del outputs
                        torch.cuda.empty_cache()

                if use_sprint:
                    x0, transfer_index = get_transfer_index_bd_adaptive(
                        active_logits,
                        active_block_mask,
                        cur_x[:, -block_length:],
                        block_end=block_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        remasking=remasking,
                        steps_left=int(denoising_steps_per_block - step),
                        minimal_topk=int(minimal_topk),
                        opt_softmax=True,
                    )
                    probs = F.softmax(active_logits.float(), dim=-1)
                    max_probs = probs.max(dim=-1).values
                    high_conf = (max_probs > threshold) & active_block_mask
                    transfer_index = transfer_index | high_conf
                else:
                    x0, x0_p = self._sample_with_temperature_topk_topp(
                        active_logits, temperature=temperature, top_k=top_k, top_p=top_p
                    )

                    num_to_transfer = num_transfer_tokens_schedule[step].item()
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                    high_conf_mask = confidence[0] > threshold

                    if high_conf_mask.sum().item() >= num_to_transfer:
                        transfer_index[0] = high_conf_mask
                    else:
                        _, idx = torch.topk(
                            confidence[0],
                            k=min(num_to_transfer, active_block_mask.sum().item()),
                        )
                        transfer_index[0, idx] = True

                if transfer_index.any():
                    cur_x[:, -block_length:][transfer_index] = x0[transfer_index]
                if eos_early_stop and (x0[transfer_index] == eos_id).any():
                    eos_pos_in_x = (cur_x[0] == eos_id).nonzero(as_tuple=True)
                    if len(eos_pos_in_x[0]) > 0:
                        eos_pos = eos_pos_in_x[0][0].item()
                        if (cur_x[0, prompt_length:eos_pos] != mask_id).all():
                            if pbar is not None:
                                pbar.close()
                            return x[:, :total_length][:, : eos_pos + 1]

            x[:, :current_window_end] = cur_x
            if (
                eos_id is not None
                and (x[0, prompt_length:current_window_end] == eos_id).any()
            ):
                break

        if pbar is not None:
            pbar.close()
        generated_answer = x[:, : prompt_length + gen_length]
        mask_positions = (generated_answer[0][input_ids.shape[1] :] == eos_id).nonzero(
            as_tuple=True
        )[0]
        if len(mask_positions) > 0:
            first_mask_position = mask_positions[0].item()
        else:
            first_mask_position = gen_length
        return generated_answer[:, : input_ids.shape[1] + first_mask_position + 1]

    @torch.no_grad()
    def generate_bd_image(
        self,
        data: Optional[dict] = None,
        temperature: float = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = True,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        eos_id: int = 156892,
        mask_id: int = 156895,
        cfg_scale: float = 1.0,
        text_vocab_size: int = None,
        cfg_rescale: float = 0.7,
        mode: str = "Normal",
        cfg_text_scale: float = 0.0,
        cfg_image_scale: float = 0.0,
        use_sprint: bool = False,
        remasking: str = "low_confidence",
        keep_ratio: float = 0.7,
        cache_warmup_steps: int = 2,
        confidence_alpha: float = 0.5,
        image_keep_ratio: Optional[float] = None,
        text_keep_ratio: Optional[float] = None,
        stability_probe: bool = False,
        stability_plot_path: Optional[str] = None,
        stability_json_path: Optional[str] = None,
        stability_layer_index: int = -2,
        stability_window: int = 4,
        stability_attn_threshold: float = 0.90,
        stability_conf_threshold: float = 0.70,
        stability_anchor_patience: int = 2,
    ):
        r"""
        Generate **discrete image tokens** using block diffusion with classifier-free guidance (CFG).

        Supports two CFG modes selected by ``mode``:

        * **Simple CFG** (``mode="Normal"``, ``cfg_scale != 1.0``):
          Two-way guidance — conditional vs. unconditional (``data["uncond_ids"]``).
          Formula: ``logits = uncond + cfg_scale * (cond - uncond)``

        * **Editing CFG** (``mode="editing"``, ``cfg_text_scale > 0`` or ``cfg_image_scale > 0``):
          Three-way guidance — full condition / no-text condition / no-image condition.
          Requires ``data["uncond_text"]`` and ``data["uncond_img"]``.
          Formula: ``logits = no_text + cfg_text * (full - no_text) + cfg_image * (no_text - no_img)``

        Text-vocabulary logits (indices ``< text_vocab_size``) are forced to ``-inf`` so that
        only discrete image tokens can be sampled.

        When ``use_sprint=True``, Sprint acceleration is enabled: the prefix KV cache is
        computed during warmup steps, pruned by importance, then reused for subsequent
        denoising steps to reduce computation. Sprint is supported for Simple CFG and
        no-CFG modes; Editing CFG automatically falls back to baseline.

        Args:
            data (`dict`):
                Must contain ``"input_ids"`` (``(1, prompt_length)`` tensor).
                For simple CFG: also ``"uncond_ids"`` (list of unconditional token IDs).
                For editing CFG: also ``"uncond_text"`` and ``"uncond_img"`` (lists of token IDs).
            temperature (`float`, *optional*, defaults to 0.0):
                Sampling temperature. 0.0 means greedy.
            block_length (`int`, *optional*, defaults to 32):
                Tokens per generation block.
            steps (`int`, *optional*, defaults to 32):
                Denoising iterations per block.
            gen_length (`int`, *optional*, defaults to 2048):
                Maximum tokens to generate (excluding prompt).
            top_p (`float`, *optional*): Nucleus-sampling cutoff.
            top_k (`int`, *optional*): Top-k filtering count.
            eos_early_stop (`bool`, *optional*, defaults to True):
                Stop at the first confirmed ``eos_id``.
            minimal_topk (`int`, *optional*, defaults to 1):
                Minimum tokens transferred per step; also caps ``steps``.
            threshold (`float`, *optional*, defaults to 0.95):
                Confidence threshold for accepting a sampled token.
            eos_id (`int`, *optional*, defaults to 156892): End-of-sequence token ID.
            mask_id (`int`, *optional*, defaults to 156895): Mask placeholder token ID.
            cfg_scale (`float`, *optional*, defaults to 1.0):
                Simple CFG strength. 1.0 disables CFG.
            text_vocab_size (`int`, *optional*):
                Boundary index — logits below this are clamped to ``-inf`` (text tokens).
                Defaults to ``config.image_token_offset``.
            cfg_rescale (`float`, *optional*, defaults to 0.7):
                Rescale factor to prevent logit-variance explosion after CFG extrapolation.
                0.0 disables rescaling.
            mode (`str`, *optional*, defaults to ``"Normal"``):
                ``"Normal"`` for text-to-image generation; ``"editing"`` for three-way editing CFG.
            cfg_text_scale (`float`, *optional*, defaults to 0.0):
                Text-guidance strength in editing mode.
            cfg_image_scale (`float`, *optional*, defaults to 0.0):
                Image-guidance strength in editing mode.
            use_sprint (`bool`, *optional*, defaults to False):
                Enable Sprint acceleration via KV cache pruning.
            remasking (`str`, *optional*, defaults to ``"low_confidence"``):
                Token remasking strategy for Sprint adaptive sampling.
            keep_ratio (`float`, *optional*, defaults to 0.7):
                Fraction of prefix KV cache to retain after pruning (Sprint mode).
            cache_warmup_steps (`int`, *optional*, defaults to 2):
                Full forward passes before switching to cached Sprint mode.
            confidence_alpha (`float`, *optional*, defaults to 0.5):
                Blending weight between KV importance and token confidence for pruning.

        Returns:
            `torch.Tensor` of shape ``(1, output_length)``: generated token sequence
            (prompt + image tokens), truncated at ``eos_id`` or ``gen_length``.
        """
        if text_vocab_size is None:
            text_vocab_size = getattr(self.config, "image_token_offset", 157184)
        steps = min(steps, gen_length // minimal_topk)
        input_ids = data["input_ids"]

        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
        ).bool()

        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()

        denoising_steps_per_block = steps
        num_transfer_tokens_schedule = self._get_num_transfer_tokens(
            block_length, denoising_steps_per_block
        )

        use_editing_cfg = mode == "editing" and (
            cfg_text_scale > 0 or cfg_image_scale > 0
        )
        use_simple_cfg = (cfg_scale != 1.0) and not use_editing_cfg

        sprint_active = use_sprint and not use_editing_cfg
        if stability_probe and sprint_active:
            logger.warning(
                "stability_probe=True 时自动禁用 Sprint，以获得稳定一致的全序列注意力图。"
            )
            sprint_active = False

        if stability_probe and stability_anchor_patience < 1:
            stability_anchor_patience = 1

        stability_records = []
        attn_hist_by_pos = {}
        runlen_by_pos = {}
        anchor_step_by_pos = {}
        probe_global_step = 0

        def _finalize_stability_probe():
            if not stability_probe or len(stability_records) == 0:
                return
            if stability_plot_path is None and stability_json_path is None:
                return

            json_path = stability_json_path
            plot_path = stability_plot_path
            if plot_path is not None:
                os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
                if json_path is None:
                    base, _ = os.path.splitext(plot_path)
                    json_path = f"{base}.json"
            if json_path is not None:
                os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

            payload = {
                "config": {
                    "stability_layer_index": int(stability_layer_index),
                    "stability_window": int(stability_window),
                    "stability_attn_threshold": float(stability_attn_threshold),
                    "stability_conf_threshold": float(stability_conf_threshold),
                    "stability_anchor_patience": int(stability_anchor_patience),
                },
                "anchor_step_by_pos": {
                    str(k): int(v) for k, v in anchor_step_by_pos.items()
                },
                "records": stability_records,
            }
            if json_path is not None:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)

            if plot_path is None:
                return
            try:
                import matplotlib.pyplot as plt

                recs = [r for r in stability_records if r["is_active"]]
                if len(recs) == 0:
                    logger.warning("stability_probe 没有可用 active 记录，跳过画图。")
                    return

                token_ids = sorted({r["token_abs"] for r in recs})
                max_step = max(r["global_step"] for r in recs)
                indegree_avg = {}
                for tid in token_ids:
                    vals = [r["indegree"] for r in recs if r["token_abs"] == tid]
                    indegree_avg[tid] = float(np.mean(vals)) if len(vals) else 0.0
                anchor_vals = {tid: anchor_step_by_pos.get(tid, max_step + 1) for tid in token_ids}
                sorted_tokens = sorted(token_ids, key=lambda t: indegree_avg[t], reverse=True)

                top_n = min(64, len(sorted_tokens))
                heat_tokens = sorted_tokens[:top_n]
                heat = np.full((top_n, max_step + 1), np.nan, dtype=np.float32)
                for i, tid in enumerate(heat_tokens):
                    rows = [r for r in recs if r["token_abs"] == tid]
                    for row in rows:
                        heat[i, row["global_step"]] = row["attn_stability"]

                xs = np.array([indegree_avg[t] for t in sorted_tokens], dtype=np.float32)
                ys = np.array([anchor_vals[t] for t in sorted_tokens], dtype=np.float32)

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                axes[0].scatter(xs, ys, alpha=0.7, s=14)
                if len(xs) >= 2 and np.unique(xs).shape[0] > 1:
                    coef = np.polyfit(xs, ys, 1)
                    line_x = np.linspace(xs.min(), xs.max(), 100)
                    line_y = coef[0] * line_x + coef[1]
                    axes[0].plot(line_x, line_y, linewidth=2)
                axes[0].set_title("Structurality vs Anchor Step")
                axes[0].set_xlabel("Mean In-degree (attention column mean)")
                axes[0].set_ylabel("Anchor Global Step (smaller=earlier)")

                im = axes[1].imshow(
                    heat,
                    aspect="auto",
                    interpolation="nearest",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                )
                axes[1].set_title(f"Attention Stability Heatmap (Top-{top_n} tokens)")
                axes[1].set_xlabel("Global Step")
                axes[1].set_ylabel("Token (sorted by structurality)")
                fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(plot_path, dpi=180)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"stability_probe 画图失败: {e}")

        def _return_with_probe(ret_tensor):
            _finalize_stability_probe()
            return ret_tensor

        def _build_uncond_inputs(uncond_token_list):
            uncond_len = len(uncond_token_list)
            pad_len = prompt_length - uncond_len
            uncond_input = torch.full(
                (1, prompt_length), mask_id, dtype=torch.long, device=self.device
            )
            if pad_len >= 0:
                uncond_input[0, -uncond_len:] = torch.tensor(
                    uncond_token_list, dtype=torch.long, device=self.device
                )
            else:
                uncond_input[0, :] = torch.tensor(
                    uncond_token_list[-prompt_length:],
                    dtype=torch.long,
                    device=self.device,
                )
                pad_len = 0
            attn_mask = block_diffusion_attention_mask.clone()
            if pad_len > 0:
                attn_mask[:, :, :, :pad_len] = False
            base_pos = torch.arange(total_length - pad_len, device=self.device)
            pad_pos = torch.zeros(pad_len, dtype=torch.long, device=self.device)
            pos_ids = torch.cat([pad_pos, base_pos]).unsqueeze(0)
            return uncond_input, attn_mask, pos_ids

        if use_simple_cfg:
            uncond_ids = (
                data["uncond_ids"]
                if isinstance(data.get("uncond_ids"), list)
                else data["uncond_ids"]
            )
            uncond_input, uncond_attn_mask, uncond_pos_ids = _build_uncond_inputs(
                uncond_ids
            )

        if use_editing_cfg:
            uncond_text_input, uncond_text_attn_mask, uncond_text_pos_ids = (
                _build_uncond_inputs(data["uncond_text"])
            )
            uncond_img_input, uncond_img_attn_mask, uncond_img_pos_ids = (
                _build_uncond_inputs(data["uncond_img"])
            )

        prefill_blocks_img = prompt_length // block_length
        num_gen_blocks_img = num_blocks - prefill_blocks_img
        pbar = tqdm(
            total=num_gen_blocks_img, desc="Generating image blocks", unit="block"
        )

        for num_block in range(prefill_blocks_img, num_blocks):
            pbar.update(1)
            pbar.set_postfix(
                block=f"{num_block - prefill_blocks_img + 1}/{num_gen_blocks_img}"
            )
            current_window_end = (num_block + 1) * block_length
            prefix_len = current_window_end - block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[
                :, :, :current_window_end, :current_window_end
            ]
            cur_position_ids = position_ids[:, :current_window_end]

            pruned_cond_cache = None
            pruned_uncond_cache = None
            pruned_nocfg_cache = None

            for step in range(denoising_steps_per_block):
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if active_block_mask.sum() == 0:
                    break

                use_cache_this_step = sprint_active and (step == cache_warmup_steps - 1)
                use_pruned = (
                    sprint_active and (step >= cache_warmup_steps) and (prefix_len > 0)
                )

                if use_editing_cfg:
                    cur_uncond_text_x = cur_x.clone()
                    cur_uncond_text_x[:, :prompt_length] = uncond_text_input
                    cur_uncond_img_x = cur_x.clone()
                    cur_uncond_img_x[:, :prompt_length] = uncond_img_input

                    combined_x = torch.cat(
                        [cur_x, cur_uncond_text_x, cur_uncond_img_x], dim=0
                    )
                    combined_pos = torch.cat(
                        [
                            cur_position_ids,
                            uncond_text_pos_ids[:, :current_window_end],
                            uncond_img_pos_ids[:, :current_window_end],
                        ],
                        dim=0,
                    )
                    combined_mask = torch.cat(
                        [
                            cur_attn_mask,
                            uncond_text_attn_mask[
                                :, :, :current_window_end, :current_window_end
                            ],
                            uncond_img_attn_mask[
                                :, :, :current_window_end, :current_window_end
                            ],
                        ],
                        dim=0,
                    )

                    outputs = self.forward(
                        combined_x,
                        attention_mask=combined_mask,
                        position_ids=combined_pos,
                        output_attentions=stability_probe,
                    )
                    logits_all = outputs.logits
                    logits_full, logits_no_text, logits_no_img = logits_all.chunk(
                        3, dim=0
                    )
                    probe_attn_2d = None
                    if stability_probe and outputs.attentions is not None:
                        n_layers = len(outputs.attentions)
                        layer_idx = (
                            stability_layer_index
                            if stability_layer_index >= 0
                            else n_layers + stability_layer_index
                        )
                        layer_idx = max(0, min(n_layers - 1, layer_idx))
                        probe_attn_2d = (
                            outputs.attentions[layer_idx][0]
                            .mean(dim=0)
                            .detach()
                            .to(torch.float32)
                            .cpu()
                        )

                    active_full = logits_full[:, -block_length:, :]
                    active_no_text = logits_no_text[:, -block_length:, :]
                    active_no_img = logits_no_img[:, -block_length:, :]

                    active_logits = (
                        active_no_text
                        + cfg_text_scale * (active_full - active_no_text)
                        + cfg_image_scale * (active_no_text - active_no_img)
                    )

                    if cfg_rescale > 0:
                        std_cond = active_full.std(dim=-1, keepdim=True)
                        std_cfg = active_logits.std(dim=-1, keepdim=True)
                        rescaled = active_logits * (std_cond / (std_cfg + 1e-6))
                        active_logits = (
                            cfg_rescale * rescaled + (1.0 - cfg_rescale) * active_logits
                        )

                elif use_simple_cfg:
                    if use_pruned and pruned_cond_cache is not None:
                        pruned_prefix_len_c = _cache_get_keys(
                            pruned_cond_cache, 0
                        ).shape[2]
                        pruned_prefix_len_u = _cache_get_keys(
                            pruned_uncond_cache, 0
                        ).shape[2]

                        block_self_attn = cur_attn_mask[
                            :, :, -block_length:, -block_length:
                        ]

                        prefix_attn_c = torch.ones(
                            1,
                            1,
                            block_length,
                            pruned_prefix_len_c,
                            dtype=torch.bool,
                            device=self.device,
                        )
                        sprint_attn_c = torch.cat(
                            [prefix_attn_c, block_self_attn], dim=-1
                        )
                        logits_cond = self.forward(
                            cur_x[:, -block_length:],
                            attention_mask=sprint_attn_c,
                            position_ids=position_ids[:, prefix_len:current_window_end],
                            past_key_values=self._sprint_shallow_copy_cache(
                                pruned_cond_cache
                            ),
                            use_cache=False,
                        ).logits

                        cur_uncond_x = cur_x.clone()
                        cur_uncond_x[:, :prompt_length] = uncond_input
                        prefix_attn_u = torch.ones(
                            1,
                            1,
                            block_length,
                            pruned_prefix_len_u,
                            dtype=torch.bool,
                            device=self.device,
                        )
                        sprint_attn_u = torch.cat(
                            [prefix_attn_u, block_self_attn], dim=-1
                        )
                        logits_uncond = self.forward(
                            cur_uncond_x[:, -block_length:],
                            attention_mask=sprint_attn_u,
                            position_ids=uncond_pos_ids[
                                :, prefix_len:current_window_end
                            ],
                            past_key_values=self._sprint_shallow_copy_cache(
                                pruned_uncond_cache
                            ),
                            use_cache=False,
                        ).logits

                        active_logits_cond = logits_cond[:, :, :]
                        active_logits_uncond = logits_uncond[:, :, :]
                        active_logits = active_logits_uncond + cfg_scale * (
                            active_logits_cond - active_logits_uncond
                        )

                    else:
                        cur_uncond_x = cur_x.clone()
                        cur_uncond_x[:, :prompt_length] = uncond_input
                        combined_x = torch.cat([cur_x, cur_uncond_x], dim=0)
                        combined_pos = torch.cat(
                            [cur_position_ids, uncond_pos_ids[:, :current_window_end]],
                            dim=0,
                        )
                        combined_mask = torch.cat(
                            [
                                cur_attn_mask,
                                uncond_attn_mask[
                                    :, :, :current_window_end, :current_window_end
                                ],
                            ],
                            dim=0,
                        )

                        if use_cache_this_step:
                            outputs = self.forward(
                                combined_x,
                                attention_mask=combined_mask,
                                position_ids=combined_pos,
                                use_cache=True,
                                output_attentions=stability_probe,
                            )
                            logits_all = outputs.logits
                            if outputs.past_key_values is not None:
                                full_cache = self._ensure_dynamic_cache(
                                    outputs.past_key_values
                                )
                                cond_cache, uncond_cache = self._split_cache_by_batch(
                                    full_cache
                                )
                                cond_conf = self._sprint_compute_prefix_confidence(
                                    logits_all[0:1], prefix_len
                                )
                                uncond_conf = self._sprint_compute_prefix_confidence(
                                    logits_all[1:2], prefix_len
                                )
                                pruned_cond_cache = self._sprint_prune_cache(
                                    cond_cache,
                                    None,
                                    prefix_len,
                                    block_length,
                                    keep_ratio,
                                    cond_conf,
                                    confidence_alpha,
                                    prefix_ids=cur_x[:, :prefix_len],
                                    image_keep_ratio=image_keep_ratio,
                                    text_keep_ratio=text_keep_ratio,
                                )
                                pruned_uncond_cache = self._sprint_prune_cache(
                                    uncond_cache,
                                    None,
                                    prefix_len,
                                    block_length,
                                    keep_ratio,
                                    uncond_conf,
                                    confidence_alpha,
                                    prefix_ids=cur_uncond_x[:, :prefix_len],
                                    image_keep_ratio=image_keep_ratio,
                                    text_keep_ratio=text_keep_ratio,
                                )
                                del full_cache, cond_cache, uncond_cache
                                torch.cuda.empty_cache()
                        else:
                            outputs = self.forward(
                                combined_x,
                                attention_mask=combined_mask,
                                position_ids=combined_pos,
                                output_attentions=stability_probe,
                            )
                            logits_all = outputs.logits

                        logits_cond, logits_uncond = logits_all.chunk(2, dim=0)
                        active_logits_cond = logits_cond[:, -block_length:, :]
                        active_logits_uncond = logits_uncond[:, -block_length:, :]
                        active_logits = active_logits_uncond + cfg_scale * (
                            active_logits_cond - active_logits_uncond
                        )
                        probe_attn_2d = None
                        if stability_probe and outputs.attentions is not None:
                            n_layers = len(outputs.attentions)
                            layer_idx = (
                                stability_layer_index
                                if stability_layer_index >= 0
                                else n_layers + stability_layer_index
                            )
                            layer_idx = max(0, min(n_layers - 1, layer_idx))
                            probe_attn_2d = (
                                outputs.attentions[layer_idx][0]
                                .mean(dim=0)
                                .detach()
                                .to(torch.float32)
                                .cpu()
                            )

                    if cfg_rescale > 0:
                        if use_pruned and pruned_cond_cache is not None:
                            std_cond = active_logits_cond.std(dim=-1, keepdim=True)
                        else:
                            std_cond = active_logits_cond.std(dim=-1, keepdim=True)
                        std_cfg = active_logits.std(dim=-1, keepdim=True)
                        rescaled = active_logits * (std_cond / (std_cfg + 1e-6))
                        active_logits = (
                            cfg_rescale * rescaled + (1.0 - cfg_rescale) * active_logits
                        )

                else:
                    if use_pruned and pruned_nocfg_cache is not None:
                        pruned_prefix_len = _cache_get_keys(
                            pruned_nocfg_cache, 0
                        ).shape[2]
                        prefix_attn = torch.ones(
                            1,
                            1,
                            block_length,
                            pruned_prefix_len,
                            dtype=torch.bool,
                            device=self.device,
                        )
                        block_self_attn = cur_attn_mask[
                            :, :, -block_length:, -block_length:
                        ]
                        sprint_attn_mask = torch.cat(
                            [prefix_attn, block_self_attn], dim=-1
                        )

                        logits = self.forward(
                            cur_x[:, -block_length:],
                            attention_mask=sprint_attn_mask,
                            position_ids=position_ids[:, prefix_len:current_window_end],
                            past_key_values=self._sprint_shallow_copy_cache(
                                pruned_nocfg_cache
                            ),
                            use_cache=False,
                        ).logits
                        active_logits = logits[:, :, :]
                    else:
                        outputs = self.forward(
                            cur_x,
                            attention_mask=cur_attn_mask,
                            position_ids=cur_position_ids,
                            use_cache=use_cache_this_step,
                            output_attentions=stability_probe,
                        )
                        logits = outputs.logits
                        active_logits = logits[:, -block_length:, :]
                        probe_attn_2d = None
                        if stability_probe and outputs.attentions is not None:
                            n_layers = len(outputs.attentions)
                            layer_idx = (
                                stability_layer_index
                                if stability_layer_index >= 0
                                else n_layers + stability_layer_index
                            )
                            layer_idx = max(0, min(n_layers - 1, layer_idx))
                            probe_attn_2d = (
                                outputs.attentions[layer_idx][0]
                                .mean(dim=0)
                                .detach()
                                .to(torch.float32)
                                .cpu()
                            )

                        if use_cache_this_step and outputs.past_key_values is not None:
                            prefix_confidence = self._sprint_compute_prefix_confidence(
                                logits, prefix_len
                            )
                            pruned_nocfg_cache = self._sprint_prune_cache(
                                self._ensure_dynamic_cache(outputs.past_key_values),
                                None,
                                prefix_len,
                                block_length,
                                keep_ratio,
                                prefix_confidence,
                                confidence_alpha,
                                prefix_ids=cur_x[:, :prefix_len],
                                image_keep_ratio=image_keep_ratio,
                                text_keep_ratio=text_keep_ratio,
                            )
                            del outputs
                            torch.cuda.empty_cache()

                # Force image-only tokens
                active_logits[:, :, :text_vocab_size] = float("-inf")

                if stability_probe:
                    probs_probe = F.softmax(active_logits[0].to(torch.float32), dim=-1)
                    conf_probe = probs_probe.max(dim=-1).values
                    pred_probe = probs_probe.argmax(dim=-1)
                    entropy_probe = -(
                        probs_probe * torch.log(probs_probe.clamp_min(1e-8))
                    ).sum(dim=-1)

                    for j in range(block_length):
                        abs_pos = int(prefix_len + j)
                        is_active = bool(active_block_mask[0, j].item())
                        attn_stability = np.nan
                        indegree = np.nan
                        if probe_attn_2d is not None:
                            key_idx = abs_pos
                            if 0 <= key_idx < probe_attn_2d.shape[1]:
                                inbound = probe_attn_2d[:, key_idx]
                                indegree = float(inbound.mean().item())
                                hist = attn_hist_by_pos.get(abs_pos, [])
                                if len(hist) > 0:
                                    hist_mean = torch.stack(
                                        hist[-max(1, int(stability_window)) :], dim=0
                                    ).mean(dim=0)
                                    attn_stability = float(
                                        F.cosine_similarity(
                                            inbound.unsqueeze(0),
                                            hist_mean.unsqueeze(0),
                                            dim=-1,
                                        ).item()
                                    )
                                hist.append(inbound)
                                if len(hist) > max(1, int(stability_window)):
                                    hist = hist[-max(1, int(stability_window)) :]
                                attn_hist_by_pos[abs_pos] = hist

                        conf_val = float(conf_probe[j].item())
                        stable_now = (
                            (not np.isnan(attn_stability))
                            and (attn_stability >= stability_attn_threshold)
                            and (conf_val >= stability_conf_threshold)
                        )
                        runlen = runlen_by_pos.get(abs_pos, 0)
                        runlen = runlen + 1 if stable_now else 0
                        runlen_by_pos[abs_pos] = runlen
                        if (
                            abs_pos not in anchor_step_by_pos
                            and runlen >= int(stability_anchor_patience)
                        ):
                            anchor_step_by_pos[abs_pos] = int(probe_global_step)

                        stability_records.append(
                            {
                                "block_id": int(num_block),
                                "step_in_block": int(step),
                                "global_step": int(probe_global_step),
                                "token_local": int(j),
                                "token_abs": abs_pos,
                                "is_active": is_active,
                                "pred_id": int(pred_probe[j].item()),
                                "confidence": conf_val,
                                "entropy": float(entropy_probe[j].item()),
                                "indegree": float(indegree)
                                if not np.isnan(indegree)
                                else None,
                                "attn_stability": float(attn_stability)
                                if not np.isnan(attn_stability)
                                else None,
                            }
                        )
                    probe_global_step += 1

                if sprint_active:
                    x0, transfer_index = get_transfer_index_bd_adaptive(
                        active_logits,
                        active_block_mask,
                        cur_x[:, -block_length:],
                        block_end=block_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        remasking=remasking,
                        steps_left=int(denoising_steps_per_block - step),
                        minimal_topk=int(minimal_topk),
                        opt_softmax=True,
                    )
                    probs = F.softmax(active_logits.float(), dim=-1)
                    max_probs = probs.max(dim=-1).values
                    high_conf = (max_probs > threshold) & active_block_mask
                    transfer_index = transfer_index | high_conf
                else:
                    x0, x0_p = self._sample_with_temperature_topk_topp(
                        active_logits, temperature=temperature, top_k=top_k, top_p=top_p
                    )

                    num_to_transfer = num_transfer_tokens_schedule[step].item()
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                    high_conf_mask = confidence[0] > threshold

                    if high_conf_mask.sum().item() >= num_to_transfer:
                        transfer_index[0] = high_conf_mask
                    else:
                        _, idx = torch.topk(
                            confidence[0],
                            k=min(num_to_transfer, active_block_mask.sum().item()),
                        )
                        transfer_index[0, idx] = True

                if transfer_index.any():
                    cur_x[:, -block_length:][transfer_index] = x0[transfer_index]

                if eos_early_stop and (x0[transfer_index] == eos_id).any():
                    eos_pos = (cur_x[0] == eos_id).nonzero(as_tuple=True)[0]
                    if (
                        len(eos_pos) > 0
                        and (cur_x[0, prompt_length : eos_pos[0]] != mask_id).all()
                    ):
                        pbar.close()
                        return _return_with_probe(
                            x[:, :current_window_end][:, : eos_pos[0] + 1]
                        )

            x[:, :current_window_end] = cur_x
            if (x[0, prompt_length:current_window_end] == eos_id).any():
                break

        pbar.close()
        return _return_with_probe(x[:, : prompt_length + gen_length])

    # ================================================================
    # Chat template helpers
    # ================================================================

    def _get_tokenizer(self, tokenizer=None):
        tok = tokenizer or getattr(self, "tokenizer", None)
        assert tok, "Provide a tokenizer or set model.tokenizer"
        return tok

    def _get_special_tokens(self, tok, image_h=None, image_w=None):
        """Return commonly used special token id lists."""
        tokens = {
            "soi": tok("<|image|>").input_ids,
            "eoi": tok("<|/image|>").input_ids,
            "boi": tok("<boi>").input_ids,
        }
        if image_h is not None:
            tokens["h"] = tok(f"<|reserved_token_{image_h}|>").input_ids
        if image_w is not None:
            tokens["w"] = tok(f"<|reserved_token_{image_w}|>").input_ids
        return tokens

    def _build_chat(self, tok, system, user_content_ids):
        """Build: <role>SYSTEM</role> {system} <role>HUMAN</role> {user} <role>ASSISTANT</role>"""
        sys_ids = tok(f"<role>SYSTEM</role> {system} <role>HUMAN</role>").input_ids
        asst_ids = tok("<role>ASSISTANT</role>").input_ids
        return sys_ids, user_content_ids, asst_ids

    def _build_image_header(self, sp):
        """Build: <soi> <h> <w> <boi>"""
        return sp["soi"] + sp["h"] + sp["w"] + sp["boi"]

    # ================================================================
    # High-level API
    # ================================================================

    @torch.no_grad()
    def generate_image(
        self,
        prompt,
        tokenizer=None,
        image_h=1024,
        image_w=1024,
        steps=16,
        block_length=32,
        cfg_scale=4.0,
        gen_length=1088,
        use_sprint=False,
        remasking="low_confidence",
        keep_ratio=0.7,
        cache_warmup_steps=2,
        confidence_alpha=0.5,
        image_keep_ratio=None,
        text_keep_ratio=None,
        mode="normal",
        thinking_steps=32,
        thinking_gen_length=4096,
        thinking_temperature=0.0,
        thinking_top_p=None,
        thinking_top_k=None,
        stability_probe: bool = False,
        stability_plot_path: Optional[str] = None,
        stability_json_path: Optional[str] = None,
        stability_layer_index: int = -2,
        stability_window: int = 4,
        stability_attn_threshold: float = 0.90,
        stability_conf_threshold: float = 0.70,
        stability_anchor_patience: int = 2,
    ):
        r"""
        Text-to-image generation. Returns dict with token_ids, h, w.

        When ``mode="thinking"``, the model first generates a chain-of-thought
        reasoning trace (including the image header ``<|image|><h><w><boi>``)
        via :meth:`generate_bd`, then uses the full thinking output as the
        prefix for :meth:`generate_bd_image` to produce the image tokens.
        The returned dict includes an extra ``"thinking"`` key with the
        decoded thinking text.

        Args:
            mode (`str`, *optional*, defaults to ``"normal"``):
                ``"normal"`` for direct generation; ``"thinking"`` for
                thinking-then-generating.
            thinking_steps (`int`, *optional*, defaults to 32):
                Denoising steps per block during the thinking phase.
            thinking_gen_length (`int`, *optional*, defaults to 4096):
                Max tokens to generate during the thinking phase.
            thinking_temperature (`float`, *optional*, defaults to 0.0):
                Sampling temperature for the thinking phase.
            thinking_top_p (`float`, *optional*):
                Nucleus-sampling cutoff for the thinking phase.
            thinking_top_k (`int`, *optional*):
                Top-k filtering for the thinking phase.
        """
        image_h = image_h // 2
        image_w = image_w // 2
        tok = self._get_tokenizer(tokenizer)
        sp = self._get_special_tokens(tok, image_h // 16, image_w // 16)
        img_header = self._build_image_header(sp)
        n = (image_h // 16) * (image_w // 16)
        boi_id = sp["boi"][0] if isinstance(sp["boi"], list) else sp["boi"]

        if mode == "thinking":
            # ── Phase 1: generate thinking text ──────────────────────
            system_msg = (
                "You are a text-to-image generation assistant with a thinking process."
            )
            sys_ids, prompt_ids, asst_ids = self._build_chat(
                tok, system_msg, tok(prompt).input_ids
            )
            think_input_ids = sys_ids + prompt_ids + asst_ids

            think_out = self.generate_bd(
                data={
                    "input_ids": torch.tensor(think_input_ids)
                    .unsqueeze(0)
                    .to(self.device)
                },
                block_length=block_length,
                steps=thinking_steps,
                gen_length=thinking_gen_length,
                temperature=thinking_temperature,
                top_p=thinking_top_p,
                top_k=thinking_top_k,
                use_sprint=use_sprint,
                remasking=remasking,
                keep_ratio=keep_ratio,
                cache_warmup_steps=cache_warmup_steps,
                confidence_alpha=confidence_alpha,
                image_keep_ratio=image_keep_ratio,
                text_keep_ratio=text_keep_ratio,
            )

            # Find <boi> token to locate image start
            boi_positions = (think_out[0] == boi_id).nonzero(as_tuple=True)[0]
            if len(boi_positions) == 0:
                raise RuntimeError(
                    "Thinking phase did not produce a <boi> token. "
                    "Try increasing thinking_gen_length or adjusting parameters."
                )
            boi_pos = boi_positions[0].item()

            # Decode thinking text (between assistant tag and image header)
            thinking_text = tok.decode(
                think_out[0][len(think_input_ids) : boi_pos].tolist(),
                skip_special_tokens=True,
            )

            # ── Phase 2: generate image tokens using thinking prefix ─
            # Use everything up to and including <boi> as the prefix
            image_input_ids = think_out[:, : boi_pos + 1]

            uncond_sys, uncond_prompt, uncond_asst = self._build_chat(
                tok, system_msg, tok("<uncondition>").input_ids
            )
            unc = uncond_sys + uncond_prompt + uncond_asst + img_header

            out = self.generate_bd_image(
                data={"input_ids": image_input_ids, "uncond_ids": unc},
                block_length=block_length,
                steps=steps,
                gen_length=gen_length,
                cfg_scale=cfg_scale,
                use_sprint=use_sprint,
                remasking=remasking,
                keep_ratio=keep_ratio,
                cache_warmup_steps=cache_warmup_steps,
                confidence_alpha=confidence_alpha,
                image_keep_ratio=image_keep_ratio,
                text_keep_ratio=text_keep_ratio,
                stability_probe=stability_probe,
                stability_plot_path=stability_plot_path,
                stability_json_path=stability_json_path,
                stability_layer_index=stability_layer_index,
                stability_window=stability_window,
                stability_attn_threshold=stability_attn_threshold,
                stability_conf_threshold=stability_conf_threshold,
                stability_anchor_patience=stability_anchor_patience,
            )

            prefix_len = boi_pos + 1
            token_ids = (
                (out[0][prefix_len : prefix_len + n] - self.config.image_token_offset)
                .cpu()
                .tolist()
            )
            return {
                "token_ids": token_ids,
                "h": image_h // 16,
                "w": image_w // 16,
                "thinking": thinking_text,
            }

        else:
            # ── Normal mode (no thinking) ────────────────────────────
            sys_ids, prompt_ids, asst_ids = self._build_chat(
                tok,
                "You are a text-to-image generation assistant.",
                tok(prompt).input_ids,
            )
            ids = sys_ids + prompt_ids + asst_ids + img_header

            uncond_sys, uncond_prompt, uncond_asst = self._build_chat(
                tok,
                "You are a text-to-image generation assistant.",
                tok("<uncondition>").input_ids,
            )
            unc = uncond_sys + uncond_prompt + uncond_asst + img_header

            out = self.generate_bd_image(
                data={
                    "input_ids": torch.tensor(ids).unsqueeze(0).to(self.device),
                    "uncond_ids": unc,
                },
                block_length=block_length,
                steps=steps,
                gen_length=gen_length,
                cfg_scale=cfg_scale,
                use_sprint=use_sprint,
                remasking=remasking,
                keep_ratio=keep_ratio,
                cache_warmup_steps=cache_warmup_steps,
                confidence_alpha=confidence_alpha,
                image_keep_ratio=image_keep_ratio,
                text_keep_ratio=text_keep_ratio,
                stability_probe=stability_probe,
                stability_plot_path=stability_plot_path,
                stability_json_path=stability_json_path,
                stability_layer_index=stability_layer_index,
                stability_window=stability_window,
                stability_attn_threshold=stability_attn_threshold,
                stability_conf_threshold=stability_conf_threshold,
                stability_anchor_patience=stability_anchor_patience,
            )
            return {
                "token_ids": (
                    out[0][len(ids) : len(ids) + n] - self.config.image_token_offset
                )
                .cpu()
                .tolist(),
                "h": image_h // 16,
                "w": image_w // 16,
            }

    @torch.no_grad()
    def understand_image(
        self,
        image_tokens=None,
        image_h=None,
        image_w=None,
        question="",
        tokenizer=None,
        steps=32,
        block_length=32,
        gen_length=2048,
        use_sprint=False,
        remasking="low_confidence",
        keep_ratio=0.7,
        cache_warmup_steps=2,
        confidence_alpha=0.5,
        threshold=0.95,
        image_keep_ratio=None,
        text_keep_ratio=None,
    ):
        """Image understanding. Returns generated text.

        Args:
            image_tokens: Pre-encoded image token IDs (with image_token_offset applied).
            image_h, image_w: Semantic grid size.
            question: Text prompt for the model.
        """
        tok = self._get_tokenizer(tokenizer)
        sp = self._get_special_tokens(tok, image_h, image_w)

        img_header = self._build_image_header(sp)
        pfx = tok(question).input_ids if question else []
        ids = img_header + image_tokens + sp["eoi"] + pfx

        out = self.generate_bd(
            data={"input_ids": torch.tensor(ids).unsqueeze(0).to(self.device)},
            block_length=block_length,
            steps=steps,
            gen_length=gen_length,
            threshold=threshold,
            use_sprint=use_sprint,
            remasking=remasking,
            keep_ratio=keep_ratio,
            cache_warmup_steps=cache_warmup_steps,
            confidence_alpha=confidence_alpha,
            image_keep_ratio=image_keep_ratio,
            text_keep_ratio=text_keep_ratio,
            show_progress=False,
        )
        return tok.decode(out[0][len(ids) - len(pfx) :], skip_special_tokens=True)

    @torch.no_grad()
    def edit_image(
        self,
        image_tokens,
        image_h,
        image_w,
        instruction,
        tokenizer=None,
        steps=8,
        block_length=32,
        cfg_text_scale=4.0,
        cfg_image_scale=0.0,
        use_sprint=False,
        remasking="low_confidence",
        keep_ratio=0.7,
        cache_warmup_steps=2,
        confidence_alpha=0.5,
    ):
        """Image editing. Returns dict with token_ids, h, w."""
        tok = self._get_tokenizer(tokenizer)
        sp = self._get_special_tokens(tok, image_h, image_w)
        img_header = self._build_image_header(sp)

        sys_ids, _, asst_ids = self._build_chat(
            tok, "You are an image editing assistant.", []
        )
        instr_ids = tok(instruction).input_ids

        src_image = img_header + image_tokens + sp["eoi"]
        inp = sys_ids + src_image + instr_ids + asst_ids + img_header

        ut = (
            sys_ids + src_image + tok("<uncondition>").input_ids + asst_ids + img_header
        )
        ui = sys_ids + sp["soi"] + instr_ids + asst_ids + img_header

        out = self.generate_bd_image(
            data={
                "input_ids": torch.tensor(inp).unsqueeze(0).to(self.device),
                "uncond_text": ut,
                "uncond_img": ui,
            },
            block_length=block_length,
            steps=steps,
            gen_length=image_h * image_w,
            mode="editing",
            cfg_text_scale=cfg_text_scale,
            cfg_image_scale=cfg_image_scale,
            use_sprint=use_sprint,
            remasking=remasking,
            keep_ratio=keep_ratio,
            cache_warmup_steps=cache_warmup_steps,
            confidence_alpha=confidence_alpha,
        )
        return {
            "token_ids": (
                out[0][len(inp) : len(inp) + image_h * image_w]
                - self.config.image_token_offset
            )
            .cpu()
            .tolist(),
            "h": image_h,
            "w": image_w,
        }
