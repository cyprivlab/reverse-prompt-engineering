# Reverse Prompt Engineering

This project focuses on reverse prompt engineering techniques for both image and text generation models. It contains implementations and comparisons of different methods for prompt inversion.

## Project Structure

The project is organized into two main components:

### 1. Image Prompt Inversion
Located in `image_prompt_inversion/`, this component contains implementations for inverting prompts from images. It includes:

- `4methods_compare_coco.ipynb`: Comparison of 4 different methods on COCO dataset
- `4methods_compare_sd.ipynb`: Comparison of 4 different methods on Stable Diffusion
- `compare_sd.ipynb`: Comparison of different methods on Stable Diffusion
- `compare_coco.ipynb`: Comparison of different methods on COCO dataset

### 2. Text Prompt Inversion
Located in `text_prompt_inversion/`, this component focuses on text prompt inversion:

- `a2q_test.ipynb`: Main implementation for text prompt inversion
- `a2q_test_translated.ipynb`: Translated version of the text prompt inversion implementation

## Datasets

The project uses the following datasets:

1. COCO Prompts Dataset:
   - Source: [Regen_COCO_prompts](https://huggingface.co/datasets/cyprivlab/Regen_COCO_prompts)

2. Stable Diffusion Dataset:
   - Source: [Regen_SatableDiffusion](https://huggingface.co/datasets/cyprivlab/Regen_SatableDiffusion)

3. Video Dataset (CogVideo):
   - Source: [Gen_CogVideo_Video](https://huggingface.co/datasets/WenhaoWang/VidProM/resolve/main/example/cog_videos_example.tar)

## Result
   To view the video inversion results, please check the following folder:
   - Video Inversion Results: [Video Inversion Results](video_prompt_inversion/Examples)


## Usage

Each component contains Jupyter notebooks that can be run independently to perform prompt inversion experiments. The notebooks include detailed implementations and comparisons of different methods.

## Requirements

Please refer to the individual notebooks for specific requirements and dependencies.
