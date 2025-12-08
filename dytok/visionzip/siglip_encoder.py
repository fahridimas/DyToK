#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA-NeXT (https://github.com/LLaVA-VL/LLaVA-NeXT)
# Copyright 2025 Yulin Li
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .utils import dynamic_compression, static_compression


def SigLipAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Patched SigLip attention forward pass with VisionZip metric extraction.
    
    Returns
        - attn_output: Attention output tensor.
        - attn_weights: Attention weight matrix.
        - metric: Key states metric (mean across heads) for contextual token filtering.
    """

    batch_size, q_len, _ = hidden_states.size()

    # Linear projections for Q, K, V
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape to multi-head format: (batch, num_heads, seq_len, head_dim)
    query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    # Compute attention scores
    k_v_seq_len = key_states.shape[-2]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
    if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
        raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

    # Apply attention mask if provided
    if attention_mask is not None:
        if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
            raise ValueError(f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}")
        attn_weights = attn_weights + attention_mask

    # Softmax and dropout (upcast to fp32 for numerical stability)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
        raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

    # Reshape back to (batch, seq_len, embed_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

    # Output projection
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights, key_states.mean(1)


def SigLip_EncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.FloatTensor]:
    """
    Patched SigLip encoder layer forward pass with VisionZip metric capture.
    
    Captures attention-derived metrics from the attention module and stores them
    for downstream token selection when VisionZip metadata is available.
    """

    # First residual block: Layer Normalization + Self-Attention
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    hidden_states, attn_weights, metric = self.self_attn(  # Capture metric
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
    )
    hidden_states = residual + hidden_states

    # Store metric for token selection
    self.metric = metric
    
    # Second residual block: Layer Normalization + MLP
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    # Prepare outputs
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def SigLipVisionTower_forward(self, images):
    """
    Perform forward propagation on the SigLip vision tower using VisionZip, optionally enhanced by dynamic token budget allocation (DyToK).

    Phases:
    - Detection phase: Default execution when VisionZip metadata is absent; returns raw hidden states without token compression.
    - Application phase: Activated when VisionZip metadata (`self.vision_tower._info`) is available.
        - Pooling Disabled: Performs token compression (raw VisionZip or dynamically enhanced by DyToK).
        - Pooling Enabled: Skips imediate token compression, providing raw hidden states and criteria for subsequent compression after pooling.
    """

    # Application Phase: VisionZip metadata available
    if hasattr(self.vision_tower, "_info") and self.vision_tower._info:
        # ! ———— DyToK Begin ————
        cache = getattr(self.vision_tower, "_cached_outputs", None)
        if cache is not None:
            hidden_states = cache["hidden_states"].to(images.dtype)
            attn_rec = cache["attn_rec"].to(images.dtype)
            metric = cache["metric"].to(images.dtype)

            # Clear cache after reuse to avoid stale data across requests
            self.vision_tower._cached_outputs = None
        # ! ———— DyToK End ————
        else:
            # Fallback: recompute features for raw VisionZip compatibility
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                output_attentions=True,
            )
            attn_weights = image_forward_outs.attentions[-1].to(images.dtype)
            hidden_states = image_forward_outs.hidden_states[-1].to(images.dtype)
            metric = self.vision_tower.vision_model.encoder.layers[-1].metric.to(images.dtype)
            attn_rec = attn_weights.mean(dim=1).mean(dim=1)

        assert hidden_states.shape[-2] == 729

        dominant_num = self.vision_tower._info["dominant"]
        contextual_num = self.vision_tower._info["contextual"]

        # Pooling disabled: perform immediate token compression
        if not self.vision_tower._info["pooling"]:

            # ! ———— DyToK Begin ————
            if isinstance(dominant_num, torch.Tensor):  # Dynamic token budget allocation
                hidden_states_save, topk_indices = dynamic_compression(hidden_states, attn_rec, metric, dominant_num, contextual_num)
            # ! ———— DyToK End ————
            else:  # Raw VisionZip with static token budget
                hidden_states_save, topk_indices = static_compression(hidden_states, attn_rec, metric, dominant_num, contextual_num)

            extra_info = {}

        # Pooling enabled: skip compression, provide criteria for post-pooling
        else:
            hidden_states_save = hidden_states
            topk_indices = None
            extra_info = {
                "attn_rec": attn_rec,
                "metric": metric,
                "dominant_num": dominant_num,
                "contextual_num": contextual_num
            }

    # Detection Phase: no metadata available, return raw hidden states
    else:
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=True,
        )
        hidden_states_save = image_forward_outs.hidden_states[-1].to(images.dtype)
        attn_weights = image_forward_outs.attentions[-1].to(images.dtype)
        metric = self.vision_tower.vision_model.encoder.layers[-1].metric.to(images.dtype)
        attn_rec = attn_weights.mean(dim=1).mean(dim=1)
        assert hidden_states_save.shape[-2] == 729
        topk_indices = None
        extra_info = {}

        # ! ———— DyToK Begin ————
        # Cache encoder outputs for the subsequent application phase
        self.vision_tower._cached_outputs = {
            "hidden_states": hidden_states_save,
            "attn_rec": attn_rec,
            "metric": metric,
        }
        # ! ———— DyToK End ————

    return hidden_states_save, topk_indices, extra_info
