
## Quick Start

### Linux

```
# Create conda environment
conda env create -f longvu.yaml
conda activate longvu

# Install additional dependencies
pip install -r requirements.txt

```

Download checkpoints and put it under `./checkpoints`

| Modality | LongVU_Qwen2_7B | LongVU_Llama3_2_3B |
:--------------------------:| :--------------------------:|:--------------------------:
| Image | [Download](https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B_img) | [Download](https://huggingface.co/Vision-CAIR/LongVU_Llama3_2_3B_img) |
| Video | [Download](https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B) | [Download](https://huggingface.co/Vision-CAIR/LongVU_Llama3_2_3B) |

Run demo `python app.py` locally with minimum 40G GPU.

<details>
  <summary>Click for quick inference code</summary>
    
```python
import numpy as np
import torch
from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader

tokenizer, model, image_processor, context_len = load_pretrained_model(
    "./checkpoints/longvu_qwen", None, "cambrian_qwen",
)

model.eval()
video_path = "./examples/video1.mp4"
qs = "Describe this video in detail"

vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
fps = float(vr.get_avg_fps())
frame_indices = np.array([i for i in range(0, len(vr), round(fps),)])
video = []
for frame_index in frame_indices:
    img = vr[frame_index].asnumpy()
    video.append(img)
video = np.stack(video)
image_sizes = [video[0].shape[:2]]
video = process_images(video, image_processor, model.config)
video = [item.unsqueeze(0) for item in video]

qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
conv = conv_templates["qwen"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=video,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0.2,
        max_new_tokens=128,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
    )
pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
```
    
</details>


## Training

### Scripts

Modify the PATH_TO_JSON and PATH_TO_FOLDER arguments in the training scripts to your save folder.

```
PATH_TO_JSON=""
PATH_TO_FOLDER=""
```
Training your own model
```
# image sft
sh scripts/train_image_qwen.sh
sh scripts/train_image_llama3_2.sh
```

Modify PREV_STAGE_CHECKPOINT in the training scripts to your first stage model path

Change `image_token_len` and `query_num_list` in `config.json` to 144

```
# video sft
sh scripts/train_video_qwen.sh
sh scripts/train_video_llama3_2.sh
```

## Evaluation

See detailed evaluation code in [evaluation_forbert_xclip.ipynb](evaluation_forbert_xclip.ipynb)

