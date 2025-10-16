# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch DINOv2 model."""

import collections.abc
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, OrderedDict
from dataclasses import dataclass
import math
from collections.abc import MutableMapping

import torch
from torch import Tensor
from torch import nn

from transformers import PreTrainedModel, Dinov2Config
from transformers.utils import BackboneMixin

from mmdet.models.builder import BACKBONES
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import ATTENTION


def torch_int(x):
    """
    Casts an input to a torch int64 tensor if we are in a tracing context, otherwise to a Python int.
    """
    import torch

    return x.to(torch.int64) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)

def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index

def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).detach().clone()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.detach().clone()
        else:
            b = layer.bias[index].detach().clone()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None and causal_mask.ndim == 4:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None

class AttentionInterface(MutableMapping):
    """
    Dict-like object keeping track of allowed attention functions. You can easily add a new attention function
    with a call to `register()`. If a model needs to locally overwrite an existing attention function, say `sdpa`,
    it needs to declare a new instance of this class inside the `modeling_<model>.py`, and declare it on that instance.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "sdpa": sdpa_attention_forward,
    }

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        # First check if instance has a local override
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        # Allow local update of the default functions without impacting other instances
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        # Ensure we use all keys, with the overwritten ones on top
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value: Callable):
        cls._global_mapping.update({key: value})

    def valid_keys(self) -> List[str]:
        return list(self.keys())

ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()

@dataclass
class BackboneOutput:
    """
    Base class for outputs of backbones.

    Args:
        feature_maps (`tuple(torch.FloatTensor)` of shape `(batch_size, num_channels, height, width)`):
            Feature maps of the stages.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, num_channels, height, width)`,
            depending on the backbone.

            Hidden-states of the model at the output of each stage plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Only applicable if the backbone uses attention.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    feature_maps: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
@dataclass
class BaseModelOutput:
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    

@dataclass
class BaseModelOutputWithPooling:
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)
    
class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)

class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)


class AccurateGELUActivation(nn.Module):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1 + torch.tanh(self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3))))
    
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_accurate": AccurateGELUActivation,
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "prelu": nn.PReLU,
}
ACT2FN = ClassInstantier(ACT2CLS)


class Dinov2Embeddings(BaseModule):
    """
    Construct the CLS token, mask token, position and patch embeddings.
    """

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # if config.use_mask_token:
        #     self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.patch_embeddings = Dinov2PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        # self.use_mask_token = config.use_mask_token
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and interpolation at torch.float32 precision.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=target_dtype)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        # if bool_masked_pos is not None and self.use_mask_token:
        #     embeddings = torch.where(
        #         bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
        #     )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings


class Dinov2PatchEmbeddings(BaseModule):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


# Copied from transformers.models.vit.modeling_vit.eager_attention_forward
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
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->Dinov2
@ATTENTION.register_module()
class Dinov2SelfAttention(BaseModule):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                print(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->Dinov2
class Dinov2SelfOutput(BaseModule):
    """
    The residual connection is defined in Dinov2Layer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->Dinov2
@ATTENTION.register_module()
class Dinov2Attention(BaseModule):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.attention = Dinov2SelfAttention(config)
        self.output = Dinov2SelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class Dinov2LayerScale(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath
class Dinov2DropPath(BaseModule):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Dinov2MLP(BaseModule):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class Dinov2SwiGLUFFN(BaseModule):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        return self.weights_out(hidden)


class Dinov2Layer(BaseModule):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Dinov2Attention(config)
        self.layer_scale1 = Dinov2LayerScale(config)
        self.drop_path = Dinov2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_swiglu_ffn:
            self.mlp = Dinov2SwiGLUFFN(config)
        else:
            self.mlp = Dinov2MLP(config)
        self.layer_scale2 = Dinov2LayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in Dinov2, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->Dinov2
class Dinov2Encoder(BaseModule):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Dinov2Layer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # import ipdb; ipdb.set_trace()
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Dinov2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Dinov2Config
    base_model_prefix = "dinov2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Dinov2SwiGLUFFN"]
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Dinov2Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

            # if self.config.use_mask_token:
            #     module.mask_token.data.zero_()
        elif isinstance(module, Dinov2LayerScale):
            module.lambda1.data.fill_(self.config.layerscale_value)


DINOV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Dinov2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DINOV2_BASE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BitImageProcessor.preprocess`] for details.

        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

DINOV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BitImageProcessor.preprocess`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

class Dinov2Model(Dinov2PreTrainedModel):
    def __init__(self, config: Dinov2Config):
        super().__init__(config)
        self.config = config

        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> Dinov2PatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Dinov2Backbone(Dinov2PreTrainedModel, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)

        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> Dinov2PatchEmbeddings:
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        >>> model = AutoBackbone.from_pretrained(
        ...     "facebook/dinov2-base", out_features=["stage2", "stage5", "stage8", "stage11"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 16, 16]
        ```"""
        # import ipdb; ipdb.set_trace()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        embedding_output = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict
        )

        hidden_states = outputs.hidden_states if return_dict else outputs[1]
        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size
                    hidden_state = hidden_state.reshape(batch_size, height // patch_size, width // patch_size, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps += (hidden_state,)

        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

@BACKBONES.register_module()
class DINOv2(BaseModule):
    def __init__(self, config,
                 out_features=["stage12"],
                 ignore_mismatched_sizes=True,
                 output_hidden_states=True,
                 frozen_stages=-1,
                 mask_ratio=None
                 ):
        super(DINOv2, self).__init__()
        self.config = config
        self.out_features = out_features
        self.ignore_mismatched_sizes = ignore_mismatched_sizes
        self.output_hidden_states = output_hidden_states
        self.frozen_stages = frozen_stages
        self.mask_ratio = mask_ratio
        
        self.model = Dinov2Backbone.from_pretrained(self.config,
                                                    out_features = self.out_features,
                                                    ignore_mismatched_sizes = self.ignore_mismatched_sizes
                                                    )
        self._freeze_stages()
        
    def apply_patch_mask_to_image_batch(self, img_batch, patch_size=14):
        """
        img_batch: [B*N, C, H, W]
        mask_batch: [B*N, num_patches] — бинарная маска 0/1, где 1 — замаскировать патч
        Возвращает: img_batch с замаскированными патчами (в цвет серый 0)
        """
        Bn, C, H, W = img_batch.shape
        h = H // patch_size
        w = W // patch_size
        num_patches = h * w
        # mask_batch = torch.randint(0, 2, (Bn, num_patches))
        mask_batch = self.generate_random_mask(Bn, H, W)
        assert mask_batch.shape == (Bn, num_patches), "Размер маски не совпадает с количеством патчей"

        img_batch = img_batch.clone()

        for idx in range(Bn):
            img = img_batch[idx]
            mask = mask_batch[idx].reshape(h, w)
            for i in range(h):
                for j in range(w):
                    if mask[i, j] == 1:
                        y1 = i * patch_size
                        y2 = (i + 1) * patch_size
                        x1 = j * patch_size
                        x2 = (j + 1) * patch_size
                        img[:, y1:y2, x1:x2] = 0  # серый цвет

        return img_batch
    
    def generate_random_mask(self, batch_size, H, W, patch_size=14):
        """
        Возвращает бинарную маску [B, num_patches], где 1 — маскировать
        """
        h = H // patch_size
        w = W // patch_size
        num_patches = h * w
        num_mask = int(self.mask_ratio * num_patches)

        # Создаем маску на каждый элемент в батче
        masks = []
        for _ in range(batch_size):
            idx = torch.randperm(num_patches)[:num_mask]
            mask = torch.zeros(num_patches)
            mask[idx] = 1
            masks.append(mask)

        return torch.stack(masks)  # [B, num_patches]
        
    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        if self.mask_ratio and self.model.training:
            x = self.apply_patch_mask_to_image_batch(x)
        outputs = self.model(x, output_hidden_states=self.output_hidden_states)
        feature_maps = outputs.feature_maps
        # import ipdb; ipdb.set_trace()
        return feature_maps
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.model.embeddings.eval()
            for param in self.model.embeddings.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            for i in range(0, self.frozen_stages - 1):
                m = self.model.encoder.layer[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    
    # def __call__(self, x):
    #     outputs = self.forward(x)
    #     feature_maps = outputs.feature_maps
    #     return feature_maps[-1]
    
class _LoRA_Linear(nn.Module):
    """
    Обертка для одного nn.Linear слоя для применения LoRA.
    Добавляет выход LoRA-пути (W_b * W_a * x) к выходу оригинального слоя.
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        linear_a: nn.Linear,
        linear_b: nn.Linear,
        layer_norm: Optional[nn.Module] = None, # Опциональная нормализация перед LoRA
    ):
        super().__init__()
        self.original_linear = original_linear
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.layer_norm = layer_norm if layer_norm is not None else nn.Identity()

        # Сохраняем размерности для информации
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = linear_a.out_features

    def forward(self, x: Tensor) -> Tensor:
        original_output = self.original_linear(x)
        
        # Применяем LayerNorm перед LoRA, если он есть
        lora_input = self.layer_norm(x)
        
        # Вычисляем LoRA "дельту"
        lora_output = self.linear_b(self.linear_a(lora_input))

        # Добавляем дельту к оригинальному выходу
        return original_output + lora_output

# --- Адаптированный класс LoRA для Hugging Face DINOv2 ---
class LoRA(BaseModule):
    def __init__(
        self,
        dinov2_model: Dinov2Backbone,
        r: int=32,                 # Ранг LoRA
        lora_layer_ids: Optional[List[int]] = None, # Индексы слоев для LoRA (None = все)
        use_layer_norm: bool = False, # Использовать ли LayerNorm перед LoRA
        lora_k: bool = True,          # Применять ли LoRA к Key проекции
        lora_v: bool = True,          # Применять ли LoRA к Value проекции (рекомендуется True)
        lora_q: bool = True,          # Применять ли LoRA к Query проекции (рекомендуется True)
    ):
        super(LoRA, self).__init__()
        
        assert r > 0, "LoRA rank 'r' must be positive."
                
        self.r = r
        self.lora_layer_ids = lora_layer_ids if lora_layer_ids is not None else list(range(len(dinov2_model.encoder.layer)))
        self.use_layer_norm = use_layer_norm
        self.lora_k = lora_k
        self.lora_v = lora_v
        self.lora_q = lora_q

        # Хранилища для LoRA слоев (A и B матрицы)
        self.w_As: nn.ModuleList = nn.ModuleList()
        self.w_Bs: nn.ModuleList = nn.ModuleList()
                    

        # "Хирургия": встраиваем LoRA слои
        for layer_idx, blk in enumerate(dinov2_model.encoder.layer):
            if layer_idx not in self.lora_layer_ids:
                continue

            # Доступ к слоям внимания Q, K, V
            # Путь: Dinov2Layer -> Dinov2Attention -> Dinov2SelfAttention -> [query, key, value]
            attn_module = blk.attention.attention
            original_query = attn_module.query
            original_key = attn_module.key
            original_value = attn_module.value

            dim = original_query.in_features

            # --- Обработка Query ---
            if self.lora_q:
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                ln_q = nn.LayerNorm(dim) if use_layer_norm else None
                
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                
                attn_module.query = _LoRA_Linear(
                    original_query, w_a_linear_q, w_b_linear_q, ln_q
                )
                print(f"Applied LoRA to Query in layer {layer_idx}")

            # --- Обработка Key ---
            if self.lora_k:
                w_a_linear_k = nn.Linear(dim, r, bias=False)
                w_b_linear_k = nn.Linear(r, dim, bias=False)
                ln_k = nn.LayerNorm(dim) if use_layer_norm else None

                self.w_As.append(w_a_linear_k)
                self.w_Bs.append(w_b_linear_k)

                attn_module.key = _LoRA_Linear(
                    original_key, w_a_linear_k, w_b_linear_k, ln_k
                )
                print(f"Applied LoRA to Key in layer {layer_idx}")

            # --- Обработка Value ---
            if self.lora_v:
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                ln_v = nn.LayerNorm(dim) if use_layer_norm else None

                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)

                attn_module.value = _LoRA_Linear(
                    original_value, w_a_linear_v, w_b_linear_v, ln_v
                )
                print(f"Applied LoRA to Value in layer {layer_idx}")

        self.reset_parameters() # Инициализация LoRA весов
        self.lora_dinov2 = dinov2_model # Сохраняем модифицированную модель

        # Размораживаем только параметры LoRA и LayerNorm (если используется)
        for param in self.w_As.parameters():
            param.requires_grad = True
        for param in self.w_Bs.parameters():
            param.requires_grad = True
            
        if self.use_layer_norm:
            for blk in self.lora_dinov2.encoder.layer:
                 if hasattr(blk.attention.attention.query, 'layer_norm') and blk.attention.attention.query.layer_norm is not None:
                     for param in blk.attention.attention.query.layer_norm.parameters():
                         param.requires_grad = True
                 if hasattr(blk.attention.attention.key, 'layer_norm') and blk.attention.attention.key.layer_norm is not None:
                     for param in blk.attention.attention.key.layer_norm.parameters():
                         param.requires_grad = True
                 if hasattr(blk.attention.attention.value, 'layer_norm') and blk.attention.attention.value.layer_norm is not None:
                     for param in blk.attention.attention.value.layer_norm.parameters():
                         param.requires_grad = True


    def save_lora_parameters(self, filename: str) -> None:
        """Сохраняет параметры ТОЛЬКО LoRA слоев (A и B) в файл .pt"""
        assert filename.endswith(".pt"), "Filename must end with .pt"

        lora_state_dict = {}
        for i, w_A in enumerate(self.w_As):
            lora_state_dict[f"w_a_{i:03d}"] = w_A.state_dict()
        for i, w_B in enumerate(self.w_Bs):
             lora_state_dict[f"w_b_{i:03d}"] = w_B.state_dict()
        
        # Опционально: сохраняем параметры LayerNorm, если они использовались и обучались
        if self.use_layer_norm:
            ln_state_dict = {}
            ln_idx = 0
            for layer_idx, blk in enumerate(self.lora_dinov2.encoder.layer):
                 if layer_idx not in self.lora_layer_ids:
                      continue
                 
                 def add_ln_state(module, prefix):
                      nonlocal ln_idx
                      if hasattr(module, 'layer_norm') and isinstance(module.layer_norm, nn.LayerNorm):
                           ln_state_dict[f"{prefix}_ln_{ln_idx:03d}"] = module.layer_norm.state_dict()
                           ln_idx += 1
                           
                 if self.lora_q: add_ln_state(blk.attention.attention.query, f"query_layer_{layer_idx}")
                 if self.lora_k: add_ln_state(blk.attention.attention.key,   f"key_layer_{layer_idx}")
                 if self.lora_v: add_ln_state(blk.attention.attention.value, f"value_layer_{layer_idx}")

            if ln_state_dict:
                 lora_state_dict['lora_layer_norms'] = ln_state_dict


        torch.save(lora_state_dict, filename)
        print(f"LoRA parameters saved to {filename}")

    def load_lora_parameters(self, filename: str) -> None:
        """Загружает параметры LoRA слоев (A и B) из файла .pt"""
        assert filename.endswith(".pt"), "Filename must end with .pt"

        state_dict = torch.load(filename, map_location='cpu') # Загружаем на CPU

        # Загрузка весов A и B
        for i, w_A in enumerate(self.w_As):
            key = f"w_a_{i:03d}"
            if key in state_dict:
                w_A.load_state_dict(state_dict[key])
            else:
                print(f"Warning: LoRA weight key {key} not found in checkpoint.")
        
        for i, w_B in enumerate(self.w_Bs):
            key = f"w_b_{i:03d}"
            if key in state_dict:
                w_B.load_state_dict(state_dict[key])
            else:
                print(f"Warning: LoRA weight key {key} not found in checkpoint.")

        # Опционально: загрузка весов LayerNorm
        if self.use_layer_norm and 'lora_layer_norms' in state_dict:
             ln_state_dict = state_dict['lora_layer_norms']
             ln_idx = 0
             for layer_idx, blk in enumerate(self.lora_dinov2.encoder.layer):
                 if layer_idx not in self.lora_layer_ids:
                      continue
                 
                 def load_ln_state(module, prefix):
                     nonlocal ln_idx
                     if hasattr(module, 'layer_norm') and isinstance(module.layer_norm, nn.LayerNorm):
                         ln_key = f"{prefix}_ln_{ln_idx:03d}"
                         if ln_key in ln_state_dict:
                             module.layer_norm.load_state_dict(ln_state_dict[ln_key])
                         else:
                             print(f"Warning: LoRA LayerNorm key {ln_key} not found in checkpoint.")
                         ln_idx += 1

                 if self.lora_q: load_ln_state(blk.attention.attention.query, f"query_layer_{layer_idx}")
                 if self.lora_k: load_ln_state(blk.attention.attention.key,   f"key_layer_{layer_idx}")
                 if self.lora_v: load_ln_state(blk.attention.attention.value, f"value_layer_{layer_idx}")

        print(f"LoRA parameters loaded from {filename}")

    def reset_parameters(self) -> None:
        """Инициализация LoRA весов."""
        for w_A in self.w_As:
            # Инициализация Kaiming для матрицы A
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            # Инициализация нулями для матрицы B
            nn.init.zeros_(w_B.weight)
        
        # Опционально: инициализация LayerNorm (обычно стандартная инициализация уже хороша)
        if self.use_layer_norm:
            for blk in self.lora_dinov2.encoder.layer:
                 # Стандартная инициализация LN: weight=1, bias=0
                 if hasattr(blk.attention.attention.query, 'layer_norm') and isinstance(blk.attention.attention.query.layer_norm, nn.LayerNorm):
                      nn.init.ones_(blk.attention.attention.query.layer_norm.weight)
                      nn.init.zeros_(blk.attention.attention.query.layer_norm.bias)
                 if hasattr(blk.attention.attention.key, 'layer_norm') and isinstance(blk.attention.attention.key.layer_norm, nn.LayerNorm):
                      nn.init.ones_(blk.attention.attention.key.layer_norm.weight)
                      nn.init.zeros_(blk.attention.attention.key.layer_norm.bias)
                 if hasattr(blk.attention.attention.value, 'layer_norm') and isinstance(blk.attention.attention.value.layer_norm, nn.LayerNorm):
                      nn.init.ones_(blk.attention.attention.value.layer_norm.weight)
                      nn.init.zeros_(blk.attention.attention.value.layer_norm.bias)


    # Убираем forward метод из этого класса, так как он модифицирует
    # модель на месте. Пользователь должен использовать self.lora_dinov2
    # def forward(self, x: Tensor) -> Tensor:
    #     return self.lora_dinov2(x)
    
    
@BACKBONES.register_module()
class DINOv2_LoRA(BaseModule):
    def __init__(self, config,
                 out_features=["stage12"],
                 ignore_mismatched_sizes=True,
                 output_hidden_states=True,
                 frozen_stages=13,
                 r=32, 
                 lora_k=False,
                 lora_q=True,
                 lora_v=True,
                 use_layer_norm=False,
                 lora_layer_ids=None 
                 ):
        super(DINOv2_LoRA, self).__init__()
        self.config = config
        self.out_features = out_features
        self.ignore_mismatched_sizes = ignore_mismatched_sizes
        self.output_hidden_states = output_hidden_states
        self.frozen_stages = frozen_stages
        self.lora_rank = r
        self.lora_k = lora_k
        self.lora_q = lora_q
        self.lora_v = lora_v
        self.use_layer_norm = use_layer_norm
        self.lora_layer_ids = lora_layer_ids
        
        self.dinov2 = Dinov2Backbone.from_pretrained(self.config,
                                                    out_features = self.out_features,
                                                    ignore_mismatched_sizes = self.ignore_mismatched_sizes
                                                    )
        self._freeze_stages()
                
        lora_adapter = LoRA(self.dinov2, 
                            r=self.lora_rank, 
                            lora_k=self.lora_k, # Не применять к Key
                            lora_q=self.lora_q,  # Применять к Query
                            lora_v=self.lora_v,  # Применять к Value
                            use_layer_norm=False,
                            lora_layer_ids=lora_layer_ids 
                        )
        self.model = lora_adapter.lora_dinov2
        
        print("\nTrainable parameters:")
        total_params = 0
        trainable_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                print(f"- {name} ({param.numel()})")
                trainable_params += param.numel()
        print(f"\nTotal parameters: {total_params}")
        print(f"Trainable parameters (LoRA): {trainable_params}")
        print(f"Trainable ratio: {trainable_params / total_params * 100:.4f}%")
        
        
    
    def forward(self, x):
        outputs = self.model(x)
        feature_maps = outputs.feature_maps
        # import ipdb; ipdb.set_trace()
        return feature_maps
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.dinov2.embeddings.eval()
            for param in self.dinov2.embeddings.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            for i in range(0, self.frozen_stages - 1):
                m = self.dinov2.encoder.layer
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
            if not self.use_layer_norm:
                for param in self.dinov2.layernorm.parameters():
                    param.requires_grad = False

__all__ = ["Dinov2Model", "Dinov2PreTrainedModel", "Dinov2Backbone", 'DINOv2', 'DINOv2_LoRA']
# model = Dinov2Backbone.from_pretrained('projects/mmdet3d_plugin/models/backbones/dinov2-base', 
#                                        out_features=["stage12"],
#                                        ignore_mismatched_sizes=True
#                                        )

# from transformers import AutoImageProcessor, AutoBackbone
# model = AutoBackbone.from_pretrained("facebook/dinov2-base", out_features=["stage2", "stage5", "stage8", "stage11"])
# print weights
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# model = DINOv2_LoRA('projects/mmdet3d_plugin/models/backbones/dinov2-base')
# print(model)
# print("\nTrainable parameters:")
# total_params = 0
# trainable_params = 0
# for name, param in model.named_parameters():
#     total_params += param.numel()
#     if param.requires_grad:
#         print(f"- {name} ({param.numel()})")
#         trainable_params += param.numel()
# print(f"\nTotal parameters: {total_params}")
# print(f"Trainable parameters (LoRA): {trainable_params}")
# print(f"Trainable ratio: {trainable_params / total_params * 100:.4f}%")

# x = torch.rand(1, 3, 480, 800)
# import ipdb; ipdb.set_trace()

# from transformers import AutoImageProcessor
# processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
# from PIL import Image
# import requests

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# inputs = processor(image, return_tensors="pt")
# outputs = model(x, output_hidden_states=True)

# feature_maps = output[0]
# rest = output[1:]
# print(output.feature_maps.__len__())
# feature_maps = outputs.feature_maps
# print(feature_maps[-1].shape)
