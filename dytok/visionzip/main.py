from typing import Optional, Union
import torch.nn as nn
from .modeling_qwen2 import Qwen2Attention_forward
from .utils import validate_inputs, generate_static, generate_dynamic, generate_dynamic_with_tiny
from .siglip_encoder import SigLip_EncoderLayer_forward, SigLipAttention_forward, SigLipVisionTower_forward
from .llava_arch import prepare_inputs_labels_for_multimodal_visionzip, encode_images_visionzip, encode_images_visionzip_multi


def visionzip(
    model,
    dominant: int = 42,
    contextual: int = 7,
    pooling: bool = False,
    dytok: bool = False,
    upper_limit: int = 196,
    use_tiny: bool = False,
    tiny_model: Optional[nn.Module] = None,
    attn_layer: Optional[Union[int, float, str]] = None
):
    """
    Patch model with VisionZip for visual token compression.

    VisionZip reduces visual token count while preserving semantic information through:
    - Dominant tokens: Attention-weighted importance selection
    - Contextual tokens: Similarity-based spatial aggregation

    Optionally enhanced by DyToK (Dynamic Token Budget Allocation):
    - Assess frame importance during prefilling
    - Allocate variable token budgets per frame based on importance

    Args:
        model: Model to patch (Currently support LLaVA-OneVision)
        dominant (int): Dominant tokens per frame
            - Static mode: Fixed count for all frames
            - DyToK mode: Average budget; actual allocation varies by frame importance
        contextual (int): Contextual tokens per frame
            - Static mode: Fixed count for all frames
            - DyToK mode: Average budget; actual allocation varies by frame importance
        pooling (bool): Apply 2D pooling before compression
        dytok (bool): Enable dynamic per-frame token budgets
            - When True: Total budget = (dominant + contextual) x num_frames.
              Redistribute this total based on frame importance weights
        upper_limit (int): Maximum tokens per frame (DyToK only)
        use_tiny (bool): Use tiny (assisstant) model for frame importance assessment (DyToK only)
            - Require passing tiny_model to `model.generate(..., tiny_model=tiny_model)`
        tiny_model (nn.Module or None): Lightweight assisstant model for frame importance assessment (DyToK only)
        attn_layer (int, float, or None): Layer(s) for assessing frame importance (DyToK only)
            - None: Aggregate all layers
            - int: Single layer (e.g., 22)
            - float: Layer range via start.end format (e.g., 16.23 = layers 16-23)

    Raises:
        TypeError: If parameter validation fails
    
    Examples:
        >>> # Static VisionZip
        >>> visionzip(model, dominant=42, contextual=7)

        >>> # Static VisionZip with pooling
        >>> visionzip(model, dominant=42, contextual=7, pooling=True)
        
        >>> # DyToK-enhanced VisionZip
        >>> visionzip(model, dominant=42, contextual=7, pooling=True, dytok=True)
        
        >>> # DyToK-enhanced VisionZip with tiny model
        >>> visionzip(model, dominant=42, contextual=7, pooling=True, dytok=True, use_tiny=True, tiny_model=tiny_model)
    
    Notes:
    - Static VisionZip: All frames use identical token budgets (dominant + contextual)
    - DyToK-enhanced VisionZip: 
        - Total token budget = (dominant + contextual) x num_frames
        - Per-frame allocation = total_budget x frame_importance_weight
        - Dominant:contextual ratio maintained at 6:1 for each frame
        - Example: 32 frames with dominant=42, contextual=7
            - Total: (42+7) x 32 = 1568 tokens
            - Important frame (weight=0.05): 1568 x 0.05 = 78 tokens (67 dom + 11 ctx)
            - Less important frame (weight=0.02): 1568 x 0.02 = 31 tokens (27 dom + 4 ctx)
    - DyToK-enhanced VisionZip with tiny model: 
        - Reduce computational overhead by using lightweight model for frame importance assessment
        - Primary model receives per-frame budgets for token compression and performs final generation
    """

    # Validate input types
    attn_layer = validate_inputs(dominant, contextual, pooling, dytok, upper_limit, use_tiny, tiny_model, attn_layer)

    # Integrate VisionZip
    from llava.model.multimodal_encoder.siglip_encoder import SigLipEncoderLayer, SigLipAttention, SigLipVisionTower
    SigLipEncoderLayer.forward = SigLip_EncoderLayer_forward
    SigLipAttention.forward = SigLipAttention_forward
    SigLipVisionTower.forward = SigLipVisionTower_forward
    
    from llava.model.llava_arch import LlavaMetaForCausalLM
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_visionzip
    LlavaMetaForCausalLM.encode_images_visionzip_multi = encode_images_visionzip_multi
    LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip

    # Configure compression parameters
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
    LlavaQwenForCausalLM.dominant = dominant
    LlavaQwenForCausalLM.contextual = contextual
    LlavaQwenForCausalLM.pooling = pooling

    # Configure generation strategy
    if dytok:
        # ! ———— DyToK Begin ————
        # DyToK-enhanced VisionZip
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        Qwen2Attention.forward = Qwen2Attention_forward

        LlavaQwenForCausalLM.upper_limit = upper_limit
        LlavaQwenForCausalLM.attn_layer = attn_layer
        if use_tiny:
            model.tiny_model = tiny_model
            LlavaQwenForCausalLM.generate = generate_dynamic_with_tiny
        else:
            LlavaQwenForCausalLM.generate = generate_dynamic
        # ! ———— DyToK End ————
    else:
        # Static VisionZip
        LlavaQwenForCausalLM.generate = generate_static
