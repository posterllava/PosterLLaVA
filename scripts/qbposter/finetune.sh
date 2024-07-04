#!/bin/bash

PROMPT_VERSION=v1
MODEL_VERSION="llava-v1.5-7b"

exp_name=llava_v1.5_7b_qbposter

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path huggingface_model/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path data/qbposter/qbposter_train_instruct.json \
    --dev_data_path data/qbposter/qbposter_val_instruct.json \
    --image_folder ./data/ \
    --vision_tower huggingface_model/clip-vit-large-patch14-336/ \
    --pretrain_mm_mlp_adapter huggingface_model/$MODEL_VERSION/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/$exp_name \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 74 \
    --eval_steps 74 \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 4 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
