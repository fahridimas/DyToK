# Copyright 2024 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
# ------------------------------------------------------------------------
# Modified from Transformers 4.50.0
# Copyright 2025 Yulin Li
# ------------------------------------------------------------------------


import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, eager_attention_forward


logger = logging.get_logger(__name__)


def Qwen2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Patch Qwen2Attention forward pass to assess frame importance.
    
    During the detection phase at specified layers, assess frame importance by computing
    attention weights between the last query token and visual tokens during prefilling.
    """

    # Linear projections and reshape to multi-head format
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Apply rotary position embeddings
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Update KV cache if available
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
    # ! ———— DyToK Begin ————
    # Assess frame importance during detection phase at specified layers
    if self.tracker.phase == "detect":
        if self.tracker.dytok_info['attn_layer'] is None or self.tracker.dytok_info['layer_count'] in self.tracker.dytok_info['attn_layer']:
            # Extract visual token range and metadata
            visual_start = self.tracker.dytok_info['visual_start']
            visual_length = self.tracker.dytok_info['visual_length']
            frames_num = self.tracker.dytok_info['frames_num']

            # Compute attention between last query and visual keys
            last_query_32 = query_states[:, :, -1:, :].float()
            key_states_32 = repeat_kv(key_states, self.num_key_value_groups).float()
            visual_key_32 = key_states_32[:, :, visual_start:visual_start + visual_length - 1, :]  # Discard the last newline token
            
            # Assess frame importance.
            attn_scores = torch.matmul(last_query_32, visual_key_32.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights = nn.functional.softmax(attn_scores, dim=-1)
            attn_weights = attn_weights.mean(1).squeeze(0).squeeze(0)
            frame_weights = attn_weights.view(frames_num, -1).mean(1)
            self.tracker.dytok_info['frame_weights'].append(frame_weights)

        # Increment layer counter (only process the first generated token)
        self.tracker.dytok_info['layer_count'] += 1
    # ! ———— DyToK End ————

    # Configure sliding window attention if applicable
    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    # Select attention implementation
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # Perform attention computation
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,
        **kwargs,
    )

    # Reshape and project output
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights
