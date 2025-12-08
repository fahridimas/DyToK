#!/bin/bash
# Reproduce evaluation results for DyToK-enhanced VisionZip with lightweight assistant model
# across various retention ratios on 32-frame input LLaVA-OneVision.

# Update to your own path
export HF_HOME=/home/lyl/checkpoints
cd /home/lyl/code/DyToK

# Retention ratios: 75%, 50%, 25%, 20%, 10%
dominant_list=(126 84 42 30 12)
contextual_list=(21 14 7 5 2)
tasks=(videomme longvideobench_val_v mlvu_test)

for idx in "${!dominant_list[@]}"; do
    dominant=${dominant_list[$idx]}
    contextual=${contextual_list[$idx]}
    echo ">>> dominant=$dominant  contextual=$contextual"

    for task in "${tasks[@]}"; do
        echo ">>> task=$task dominant=${dominant_list[$idx]} contextual=${contextual_list[$idx]}"

        accelerate launch \
            -m lmms_eval \
            --model llava_onevision_dytok \
            --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,model_name=llava_qwen,conv_template=qwen_1_5,visionzip=True,dominant=${dominant_list[$idx]},contextual=${contextual_list[$idx]},pooling=True,dytok=True,upper_limit=196,use_tiny=True,tiny_pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,attn_layer=16.23,max_frames_num=32 \
            --tasks $task \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix llava_onevision_dytok \
            --output_path ./logs/dytok_visionzip_tiny_32_ov

        echo "<<< task=$task dominant=${dominant_list[$idx]} contextual=${contextual_list[$idx]} done"
    done
done
