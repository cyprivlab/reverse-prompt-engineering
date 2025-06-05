# Reverse Prompt Engineering

This project focuses on reverse prompt engineering techniques for both image and text generation models. It implements a two-phase approach for prompt inversion:

1. Phase 1: Direct Inversion - Initial prompt reconstruction using direct inversion methods
2. Phase 2: RL Finetune - Refinement of the inverted prompts using reinforcement learning

## Project Structure

The project is organized into two main components:

### 1. Text Prompt Inversion
Located in `text_prompt_inversion/`, this component implements the two-phase approach for text prompt inversion:

- `Default_DI_for_text.ipynb`: Phase 1 implementation using direct inversion for text prompts (contains Evaluation Part)
- `Fine-tuning/`: Directory containing Phase 2 implementation with RL-based fine-tuning methods
  - For RL fine-tuning instructions, please refer to the README.md file in the Fine-tuning directory
  - The fine-tuning process uses the direct inversion (DI) model as the initial checkpoint
  - Follow the configuration files in `scripts/training/task_configs/` to customize training parameters



### 2. Image Prompt Inversion
Located in `image_prompt_inversion/`, this component implements the two-phase approach for image prompt inversion:

- `Default_DI_for_image.ipynb`: Phase 1 implementation using direct inversion for image prompts
- `Fine-tuning/`: Directory containing Phase 2 implementation with RL-based fine-tuning methods
  - For RL fine-tuning instructions, please refer to the README.md file in the Fine-tuning directory
  - The fine-tuning process uses the direct inversion (DI) model as the initial checkpoint
  - Follow the configuration files in `scripts/training/task_configs/` to customize training parameters

- `Evaluation.ipynb`: Evaluation jupyter notebook for comparing results and references

### 3. Video Prompt Inversion
Located in `video_prompt_inversion/`, this component focuses on video prompt inversion. For detailed implementation and usage instructions, please refer to the README.md file in the video_prompt_inversion directory.


## Datasets

The project uses the following datasets:

The datasets employed to evaluate attacks on text-to-text models include Alpaca-GPT4 and RetrievalQA.
   1. Alpaca-GPT4 Dataset:
      - Source: [Alpaca-GPT4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4)
      - Processed dataset: [Alpaca-GPT4](https://huggingface.co/datasets/cyprivlab/Alpaca-GPT4)

   2. RetrievalQA Dataset:
      - Source: [RetrievalQA](https://github.com/hyintell/RetrievalQA/blob/main/data/retrievalqa_gpt4.jsonl)
      - Processed dataset: [GPT4RQA](https://huggingface.co/datasets/cyprivlab/GPT4RQA)

The datasets used to evaluate attacks on text-to-image models are based on MS-COCO and Stable-Diffusion-Prompts.

   3. COCO Prompts Dataset:
      - Source: [Regen_COCO_prompts](https://huggingface.co/datasets/cyprivlab/Regen_COCO_prompts)

   4. Stable Diffusion Dataset:
      - Source: [Regen_SatableDiffusion](https://huggingface.co/datasets/cyprivlab/Regen_SatableDiffusion)

We now extend our experiments to the text-to-video modality. For this setting, we use the VidProM dataset for training.

   5. Video Dataset (CogVideo):
      - Source: [Gen_CogVideo_Video](https://huggingface.co/datasets/WenhaoWang/VidProM/resolve/main/example/cog_videos_example.tar)
   
   6. Processed Video Prompts Dataset:
      - Source: [Processed Video Prompts](https://huggingface.co/datasets/cyprivlab/processed_video_prompts)

## Result
   To view the video inversion results, please check the following folder:
   - Video Inversion Results: [Video Inversion Results](video_prompt_inversion/Examples)
   


## Usage

Each component contains Jupyter notebooks that can be run independently to perform prompt inversion experiments. The notebooks include detailed implementations and comparisons of different methods.

## Requirements

Please refer to the individual notebooks for specific requirements and dependencies.
