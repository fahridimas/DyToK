"""
LLaVA-OneVision Video Question Answering with DyToK-Enhanced VisionZip
"""

import copy
import torch
import warnings
import numpy as np
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.conversation import SeparatorStyle, conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from dytok import visionzip

warnings.filterwarnings("ignore")


def load_video(video_path: str, max_frames_num: int) -> np.ndarray:
    """
    Uniformly sample frames from the video.
    """
    vr = VideoReader(video_path if isinstance(video_path, str) else video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    frame_indices = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
    sampled_frames = vr.get_batch(frame_indices).asnumpy()
    return sampled_frames  # (frames, height, width, channels)


# Update as needed
model_path = "lmms-lab/llava-onevision-qwen2-7b-ov"
tiny_model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
video_path = "/home/lyl/checkpoints/videomme/data/Qgr4dcsY-60.mp4"
question = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\nWhat activities do students engage in within the room?\nA. Reading books.\nB. Practicing spell.\nC. Fighting with rods.\nD. Making explosion.\nThe best answer is:"

# 1) Load model
tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, "llava_qwen")
_, tiny_model, _, _ = load_pretrained_model(tiny_model_path, None, "llava_qwen")
# ! ———— DyToK Begin ————
from dytok import visionzip
visionzip(model, dominant=42, contextual=7,  # raw VisionZip settings
          pooling=True,  # enable pooling
          dytok=True, upper_limit=196, use_tiny=True, tiny_model=tiny_model, attn_layer=16.23  # DyToK settings
)
# ! ———— DyToK End ————
model.eval()

# 2) Load video frames
video_raw = load_video(video_path, 32)
video = image_processor.preprocess(video_raw, return_tensors="pt")["pixel_values"].half().to(model.device)
frames_num = video.shape[0]
video = [video]

# 3) Build prompt
conv_template = "qwen_1_5"
conv = copy.deepcopy(conv_templates[conv_template])
question = question.replace('\\n', '\n')
full_question = DEFAULT_IMAGE_TOKEN + "\n" + question
conv.append_message(conv.roles[0], full_question)
conv.append_message(conv.roles[1], None)
prompt_text = conv.get_prompt()

# 4) Tokenize
input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

# 5) Generate
pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
attention_masks = input_ids.ne(pad_token_ids).long()
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=video,
        modalities=["video"],
        attention_mask=attention_masks,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        stopping_criteria=[stopping_criteria],
        pad_token_id=pad_token_ids
    )

answer_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print("\n[LLaVA-OneVision Answer]:\n", answer_text)


