PREV_STAGE_CHECKPOINT="cambrian_Llama3_2_3B"
PATH_TO_JSON="LongVU/video_prompts.json"
PATH_TO_FOLDER="vidprom/cog_videos_example"
VERSION="llama3"

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=1 \
longvu/train.py \
--output_dir "LongVU/output" \
--input_model_filename $PREV_STAGE_CHECKPOINT \
--output_model_filename "LongVU/new_checkpoints/cambrian_llama3_2_3B/" \
--data_path $PATH_TO_JSON \
--image_folder $PATH_TO_FOLDER \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir /tmp/llava/test/ \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 8 \
--save_steps 500 \
--eval_steps 500 \
--logging_steps 10 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 5e-6 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--tf32 False \
--version $VERSION \
--mm_vision_select_layer "-2" \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--dataloader_num_workers 0 \
--lazy_preprocess True \
--tune_mm_mlp_adapter False \
--freeze_mm_mlp_adapter False \
--freeze_backbone False \
--gradient_checkpointing True \
--mm_projector_type sva \
--image_token_len 144 \
--query_num_list "[144]" \
--resume True \
--lowres_token 8 \
--video_fps 0.5 \
--highres False \
--drop_threshold 0.8 \

