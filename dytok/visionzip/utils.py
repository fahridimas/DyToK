import torch
import torch.nn as nn
from typing import Optional, Union, List
from transformers.utils import logging
from transformers.generation.utils import GenerateOutput
from llava.constants import IMAGE_TOKEN_INDEX


logger = logging.get_logger(__name__)


def apply_info(model, dominant_num: Union[int, torch.Tensor], contextual_num: Union[int, torch.Tensor], pooling: bool = False):
    """
    Register VisionZip metadata to the model for token compression.

    Args:
        model: Target vision encoder model
        dominant_num: Number of dominant tokens to retain.
            - int: Fixed budget across all frames (VisionZip).
            - Tensor: Per-frame budget with shape [num_frames] (DyToK).
        contextual_num: Number of contextual tokens to retain. Fixed integer for VisionZip (same for all frames).
            - int: Fixed budget across all frames (VisionZip).
            - Tensor: Per-frame budget with shape [num_frames] (DyToK).
        pooling: Whether to defer compression until after pooling
    
    The metadata includes:
        - dominant/contextual: Token budget allocation
        - pooling: Compression timing control
    """

    # Register VisionZip metadata at model level
    model._info = {
        "dominant":dominant_num,
        "contextual":contextual_num,
        "pooling":pooling,
    }

    # Propagate metadata to encoder layers for metric capture
    for module in model.modules():
        if module.__class__.__name__ in {"SigLipEncoderLayer", "Qwen2_5_VLVisionBlock"}:
            module._info = model._info


def clear_info(model):
    """
    Clear VisionZip metadata from the model.

    Args:
        model: Vision encoder model with VisionZip metadata to clear
    """

    # Clear model-level metadata
    model._info = {}

    # Propagate cleared metadata to encoder layers
    for module in model.modules():
        if module.__class__.__name__ in {"SigLipEncoderLayer", "Qwen2_5_VLVisionBlock"}:
            module._info = model._info


class DyToKTracker:
    """
    Tracks frame importance and manages dynamic token budget allocation for DyToK.
    
    Coordinates the two-phase dynamic compression process:
    - Detection phase: Collect frame importance during prefilling
    - Application phase: Allocate token budget based on frame importance
    
    Attributes:
        dytok_info (dict): Store DyToK metadata including:
            - visual_start: Starting index of visual tokens in input sequence
            - visual_length: Total number of visual tokens
            - frames_num: Number of video frames
            - attn_layer: Layer indices for importance assessment
            - frame_weights: Collected attention weights per frame
            - layer_count: Current layer counter during detection
        phase (str): Current operation phase ("detect" or "apply")
    """
    def __init__(self):
        self.dytok_info = {}
        self.phase = "detect"  # detect | apply
    
    def reset(self):
        self.dytok_info.clear()
        self.phase = "detect"


@torch.no_grad()
def generate_dynamic(
    self,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    modalities: Optional[List[str]] = ["image"],
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    """
    Generate text with DyToK-enhanced VisionZip token compression.
    
    Two-phase process:
        - Detection Phase: Assess frame importance via attention weights during prefilling.
        - Application Phase: Allocate token budget dynamically based on frame importance.
    """

    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    if images is not None:

        # ====================================================================
        # Phase 1: Detection - Assess frame importance
        # ====================================================================

        # Initialize DyToKTracker for frame importance tracking
        self.tracker = DyToKTracker()
        self.tracker.reset()
        for layer in self.model.layers:
            layer.self_attn.tracker = self.tracker
        
        # Clear VisionZip metadata to ensure correct behavior across repeated inferences
        # when the model is loaded once but processes multiple inputs sequentially (e.g., LMMs-Eval)
        clear_info(self.model.vision_tower.vision_tower)

        # Locate visual token positions in the input sequence
        visual_start_idx = torch.where(inputs[0] == IMAGE_TOKEN_INDEX)[0].item()
        visual_end_idx = -(inputs.shape[1] - visual_start_idx)

        # Prepare multi-modal inputs and calculate visual token length
        inputs_raw, position_ids_raw, attention_mask_raw = inputs, position_ids, attention_mask
        (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        visual_token_length = (inputs_embeds.shape[1] + visual_end_idx) - visual_start_idx + 1

        # Parse attention layer specification (int, float range, or None)
        if isinstance(self.attn_layer, float):  # Format: start.end (e.g., 16.23)
            s = f"{self.attn_layer:.2f}"
            start_layer, end_layer = map(int, s.split('.'))
            self.attn_layer = list(range(start_layer, end_layer + 1))
        if isinstance(self.attn_layer, int):
            self.attn_layer = [self.attn_layer]

        # Register detection metadata
        self.tracker.dytok_info.update({
            'visual_start': visual_start_idx,
            'visual_length': visual_token_length,
            'frames_num': images[0].shape[0],
            'attn_layer': self.attn_layer,
            'frame_weights': [],
            'layer_count': 0
        })

        # Run detection pass to collect frame importance weights
        self.model(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, use_cache=False)

        # ====================================================================
        # Phase 2: Application - Allocate per-frame token budgets
        # ====================================================================

        self.tracker.phase = "apply"

        # Aggregate frame weights across layers
        frame_weights = torch.stack(self.tracker.dytok_info['frame_weights'], dim=0)  # [layer_num, frames_num]
        frame_weights = frame_weights.mean(0)  # [frames_num]
        frame_weights = frame_weights / frame_weights.sum()  # Normalize

        # Allocate total token budget proportionally to frame importance
        total_tokens = (self.dominant + self.contextual) * images[0].shape[0]
        allocations = frame_weights * total_tokens

        # Distribute fractional tokens using greedy rounding
        integer_parts = torch.floor(allocations)
        remainders = allocations - integer_parts
        remaining = total_tokens - integer_parts.sum().item()
        if remaining > 0:
            _, indices = torch.topk(remainders, int(remaining))
            integer_parts[indices] += 1

        # Clamp allocations to upper limit
        upper_limit = self.upper_limit
        clamped_allocation = torch.clamp(integer_parts, max=upper_limit)

        # Redistribute excess tokens to frames below upper limit
        current_sum = clamped_allocation.sum().item()
        if current_sum < total_tokens:
            remaining_tokens = total_tokens - current_sum
            while remaining_tokens > 0:
                candidate_indices = (clamped_allocation < upper_limit).nonzero(as_tuple=False).view(-1)
                sorted_candidate_indices = candidate_indices[torch.argsort(frame_weights[candidate_indices], descending=True)]
                for idx in sorted_candidate_indices:
                    remaining_tokens -= 1
                    if remaining_tokens < 0:
                        break
                    clamped_allocation[idx] += 1             
        integer_parts = clamped_allocation
        
        # Split each frame's budget into dominant and contextual tokens (6:1 ratio)
        dominant_num = (6 * integer_parts) // 7
        contextual_num = integer_parts - dominant_num

        # Register VisionZip metadata with dynamic token budgets
        apply_info(self.model.vision_tower.vision_tower, dominant_num, contextual_num, self.pooling)

        # Re-prepare inputs with compression applied
        (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs_raw, position_ids_raw, attention_mask_raw, None, None, images, modalities, image_sizes=image_sizes)

    else:
        inputs_embeds = self.get_model().embed_tokens(inputs)
    
    # Perform final generation with compressed visual tokens
    return super(type(self), self).generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)


@torch.no_grad()
def generate_static(
    self,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    modalities: Optional[List[str]] = ["image"],
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    """
    Generate text with raw VisionZip token compression.

    Apply fixed token budgets (dominant + contextual) for all frames,
    then perform standard multi-modal generation.
    """

    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)

    # Register VisionZip metadata with static token budgets
    clear_info(self.model.vision_tower.vision_tower)
    apply_info(self.model.vision_tower.vision_tower, self.dominant, self.contextual, self.pooling)

    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    if images is not None:
        (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
    else:
        inputs_embeds = self.get_model().embed_tokens(inputs)

    return super(type(self), self).generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)


def validate_inputs(dominant, contextual, pooling, dytok, upper_limit, use_tiny, tiny_model, attn_layer):
    """
    Validate input parameters for VisionZip with DyToK.

    Args:
        dominant (int): Dominant tokens per frame.
        contextual (int): Contextual tokens per frame.
        dytok (bool): Enable dynamic token budget allocation.
        upper_limit (int): Maximum tokens per frame.
        pooling (bool): Apply pooling before VisionZip.
        use_tiny (bool): Use tiny (assisstant) model for guidance.
        tiny_model (Optional[nn.Module]): Tiny (assisstant) model for frame importance assessment.
        attn_layer (Optional[Union[int, float, str]]): Attention layer(s) specification.

    Returns:
        Optional[Union[int, float]]: Normalized attention layer(s) specification.

    Raises:
        TypeError: On invalid argument types.
    """
    int_fields = {
        "dominant": dominant,
        "contextual": contextual,
        "upper_limit": upper_limit,
    }
    for name, value in int_fields.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"Expected '{name}' to be an int, but got {type(value).__name__}")
    
    bool_fields = {
        "dytok": dytok,
        "pooling": pooling,
        "use_tiny": use_tiny,
    }
    for name, value in bool_fields.items():
        if not isinstance(value, bool):
            raise TypeError(f"Expected '{name}' to be a bool, but got {type(value).__name__}")

    # Validate tiny model requirement
    if use_tiny and tiny_model is None:
        raise ValueError("When 'use_tiny' is True, 'tiny_model' must be provided and not None.")

    # Normalize attn_layer specification
    if isinstance(attn_layer, bool):
        raise TypeError("Expected 'attn_layer' to be None, int, or float, but got bool")
    if isinstance(attn_layer, str):
        # LMMs-Eval does not support passing None directly as model_args;
        # use the lowercase string "none" to indicate aggregation over all attention layers,
        # which is then converted back to None here.
        if attn_layer.strip().lower() == "none":
            return None
        raise TypeError(f"Expected 'attn_layer' to be None, int, or float, but got str")
    if attn_layer is not None and not isinstance(attn_layer, (int, float)):
        raise TypeError(f"Expected 'attn_layer' to be None, int, or float, but got {type(attn_layer).__name__}")
    
    return attn_layer


def dynamic_compression(hidden_states, attn_rec, metric, dominant_num: torch.Tensor, contextual_num: torch.Tensor):
    """
    DyToK-enhanced VisionZip token compression with dynamic budgets per frame.
    """
    frames_num, tokens_num, _ = hidden_states.shape
    topk_indices_list, hidden_states_save_list = [], []

    for i in range(frames_num):
        d_num = int(dominant_num[i].item())
        c_num = int(contextual_num[i].item())

        # Dominant Visual Tokens
        topk_idx_i = attn_rec[i].topk(d_num, dim=0).indices
        topk_indices_list.append(topk_idx_i)
        mask_i = torch.ones(tokens_num, dtype=torch.bool, device=metric.device).scatter(0, topk_idx_i, False)  # [729]
        dominant_tokens_i = hidden_states[i][~mask_i].view(d_num, hidden_states.shape[2])  # [335, 3584]

        if c_num > 0:
            # Filter
            metric_filtered_i = metric[i][mask_i].view(tokens_num - d_num, metric.shape[2])  # [32, 729, 72] -> [729, 72] -> [394, 72]
            hidden_states_filtered_i = hidden_states[i][mask_i].view(tokens_num - d_num, hidden_states.shape[2])             
            metric_normalized_i = metric_filtered_i / metric_filtered_i.norm(dim=-1, keepdim=True)
            
            # Contextual Visual Tokens
            step = max(1, metric_normalized_i.shape[0] // c_num)
            target_indices_i = torch.arange(0, metric_normalized_i.shape[0], step, device=metric_normalized_i.device)[:c_num]
            target_tokens_i = metric_normalized_i[target_indices_i, :]  # [56, 72]

            tokens_to_merge_i = metric_normalized_i[~torch.isin(torch.arange(metric_normalized_i.shape[0], device=metric_normalized_i.device), target_indices_i), :]  # [338, 72]
            similarity_i = tokens_to_merge_i @ target_tokens_i.transpose(0, 1)  # [338, 56]
            assign_one_hot_i = torch.zeros(tokens_to_merge_i.shape[0], c_num, dtype=hidden_states_filtered_i.dtype, device=metric_normalized_i.device)
            assign_one_hot_i.scatter_(1, similarity_i.argmax(dim=1).unsqueeze(-1), 1)  # [338, 56]

            counts_i = assign_one_hot_i.sum(dim=0).clamp(min=1).unsqueeze(-1)  # min must be at least 1 to prevent division by 0
            hidden_to_merge_i = hidden_states_filtered_i[~torch.isin(torch.arange(hidden_states_filtered_i.shape[0], device=hidden_states_filtered_i.device), target_indices_i), :]  # [338, 3584]
            aggregated_hidden_i = (assign_one_hot_i.transpose(0, 1) @ hidden_to_merge_i) / counts_i  # [56, 3584]
            target_hidden_i = hidden_states_filtered_i[target_indices_i, :]  # [56, 3584]

            contextual_tokens_i = target_hidden_i + aggregated_hidden_i
        else:
            # No contextual tokens for this frame
            contextual_tokens_i = torch.empty((0, hidden_states.shape[2]), device=hidden_states.device, dtype=hidden_states.dtype)

        # Concatenate dominant and contextual tokens
        hidden_states_save_i = torch.cat([dominant_tokens_i, contextual_tokens_i], dim=0) # [335 + 56, 3584]
        hidden_states_save_list.append(hidden_states_save_i)

    return hidden_states_save_list, topk_indices_list


def static_compression(hidden_states, attn_rec, metric, dominant_num: int, contextual_num: int):
    """
    VisionZip token compression with fixed budgets for all frames.
    
    Compress visual tokens into two categories:
    1. Dominant tokens: Selected by highest attention weights
    2. Contextual tokens: Spatially sampled and aggregated via similarity-based merging
    
    Args:
        hidden_states: Visual token embeddings [batch, num_tokens, dim]
        attn_rec: Attention weights for token importance [batch, num_tokens]
        metric: Key states for contextual similarity computation [batch, num_tokens, dim]
        dominant_num: Number of dominant tokens to retain
        contextual_num: Number of contextual tokens to retain
    
    Returns:
        - hidden_states_save: Compressed tokens [batch, dominant_num + contextual_num, dim]
        - topk_indices: Indices of retained dominant tokens [batch, dominant_num]
    """

    # ========================================================================
    # Dominant Token Selection: Attention-based Top-k
    # ========================================================================

    # Extract top-k tokens with highest attention weights
    topk_indices = attn_rec.topk(dominant_num, dim=1).indices
    mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, topk_indices, False)
    dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num, hidden_states.shape[2])

    # Early return if no contextual tokens needed
    if contextual_num == 0:
        hidden_states_save = dominant_tokens
        return hidden_states_save, topk_indices
    
    # ========================================================================
    # Contextual Token Preparation: Filter and Normalize
    # ========================================================================

    # Filter out dominant tokens from metric and hidden states
    metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - dominant_num, metric.shape[2])
    hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - dominant_num, hidden_states.shape[2])

    # Normalize metric for similarity computation
    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)

    # ========================================================================
    # Contextual Token Aggregation: Spatial Sampling and Merging
    # ========================================================================

    # Spatially sample target tokens with uniform stride
    step = max(1, metric_normalized.shape[1] // contextual_num)
    target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
    target_tokens = metric_normalized[:, target_indices, :]

    # Compute similarity between non-target and target tokens
    tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
    similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))

    # Assign each token-to-merge to its most similar target token
    assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)

    # Aggregate assigned tokens by averaging
    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)  # Prevent division by zero
    hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
    aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts

    # Combine target tokens with aggregated information
    target_hidden = hidden_states_filtered[:, target_indices, :]
    contextual_tokens = target_hidden + aggregated_hidden

    # ========================================================================
    # Final Concatenation: Dominant + Contextual
    # ========================================================================

    hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1)

    return hidden_states_save, topk_indices


@torch.no_grad()
def generate_dynamic_with_tiny(
    self,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    modalities: Optional[List[str]] = ["image"],
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    """
    Generate text with DyToK-enhanced VisionZip token compression using a tiny (assisstant) model for temporal importance estimation.
    
    Two-phase process:
        - Detection Phase: Use tiny model to assess frame importance (cost-efficient)
        - Application Phase: Apply dynamic budgets to primary model
    """

    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    if images is not None:

        # ====================================================================
        # Phase 1: Detection - Assess frame importance
        # ====================================================================

        # Initialize DyToKTracker for frame importance tracking
        self.tracker = DyToKTracker()
        self.tracker.reset()
        self.tiny_model.tracker = self.tracker
        for model in [self.model, self.tiny_model.model]:
            for layer in model.layers:
                layer.self_attn.tracker = self.tracker
        
        # Clear VisionZip metadata to ensure correct behavior across repeated inferences
        # when the model is loaded once but processes multiple inputs sequentially (e.g., LMMs-Eval)
        clear_info(self.model.vision_tower.vision_tower)

        # Locate visual token positions in the input sequence
        visual_start_idx = torch.where(inputs[0] == IMAGE_TOKEN_INDEX)[0].item()
        visual_end_idx = -(inputs.shape[1] - visual_start_idx)

        # Prepare multi-modal inputs and calculate visual token length
        inputs_raw, position_ids_raw, attention_mask_raw = inputs, position_ids, attention_mask
        (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.tiny_model.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        visual_token_length = (inputs_embeds.shape[1] + visual_end_idx) - visual_start_idx + 1

        # Parse attention layer specification (int, float range, or None)
        if isinstance(self.attn_layer, float):  # Format: start.end (e.g., 16.23)
            s = f"{self.attn_layer:.2f}"
            start_layer, end_layer = map(int, s.split('.'))
            self.attn_layer = list(range(start_layer, end_layer + 1))
        if isinstance(self.attn_layer, int):
            self.attn_layer = [self.attn_layer]

        # Register detection metadata
        self.tracker.dytok_info.update({
            'visual_start': visual_start_idx,
            'visual_length': visual_token_length,
            'frames_num': images[0].shape[0],
            'attn_layer': self.attn_layer,
            'frame_weights': [],
            'layer_count': 0
        })

        # Run detection pass to collect frame importance weights
        self.tiny_model.model(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, use_cache=False)

        # ====================================================================
        # Phase 2: Application - Allocate per-frame token budgets
        # ====================================================================

        self.tracker.phase = "apply"

        # Aggregate frame weights across layers
        frame_weights = torch.stack(self.tracker.dytok_info['frame_weights'], dim=0)
        frame_weights = frame_weights.mean(0)
        frame_weights = frame_weights / frame_weights.sum()

        # Allocate total token budget proportionally to frame importance
        total_tokens = (self.dominant + self.contextual) * images[0].shape[0]
        allocations = frame_weights * total_tokens

        # Distribute fractional tokens using greedy rounding
        integer_parts = torch.floor(allocations)
        remainders = allocations - integer_parts
        remaining = total_tokens - integer_parts.sum().item()
        if remaining > 0:
            _, indices = torch.topk(remainders, int(remaining))
            integer_parts[indices] += 1

        # Clamp allocations to upper limit
        upper_limit = self.upper_limit
        clamped_allocation = torch.clamp(integer_parts, max=upper_limit)

        # Redistribute excess tokens to frames below upper limit
        current_sum = clamped_allocation.sum().item()
        if current_sum < total_tokens:
            remaining_tokens = total_tokens - current_sum
            while remaining_tokens > 0:
                candidate_indices = (clamped_allocation < upper_limit).nonzero(as_tuple=False).view(-1)
                sorted_candidate_indices = candidate_indices[torch.argsort(frame_weights[candidate_indices], descending=True)]
                for idx in sorted_candidate_indices:
                    remaining_tokens -= 1
                    if remaining_tokens < 0:
                        break
                    clamped_allocation[idx] += 1             
        integer_parts = clamped_allocation
        
        # Split each frame's budget into dominant and contextual tokens (6:1 ratio)
        dominant_num = (6 * integer_parts) // 7
        contextual_num = integer_parts - dominant_num

        # Register VisionZip metadata with dynamic token budgets
        apply_info(self.model.vision_tower.vision_tower, dominant_num, contextual_num, self.pooling)

        # Re-prepare inputs with compression applied
        (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs_raw, position_ids_raw, attention_mask_raw, None, None, images, modalities, image_sizes=image_sizes)

    else:
        inputs_embeds = self.get_model().embed_tokens(inputs)
    
    # Perform final generation with compressed visual tokens
    return super(type(self), self).generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
