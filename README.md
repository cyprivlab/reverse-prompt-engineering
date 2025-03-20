# reverse-prompt-engineering

### Abstract
Generative models, including both text-to-text and text-to-image modalities, have underscored the significance of `prompt engineering', a technique critical for enhancing the quality of model outputs. Crafting high-quality prompts is not only time-intensive but also economically valuable, making them prime targets for manipulation. Recent research has revealed that these prompts can be stolen through a technique known as prompt inversion, which reconstructs prompts merely by analyzing the outputs of models. However, existing studies are typically confined to either text-to-text or text-to-image models and are not cross-applicable, thus limiting their real-world utility. This gap raises a crucial question: Is there a unified approach capable of addressing both model types? In this paper, we present the first comprehensive study on a unified prompt inversion approach that targets both text and image models. Our approach involves two model-agnostic phases: (1) training an inversion model to generate initial prompt approximations from model outputs, and (2) using reinforcement learning to fine-tune the inversion model for enhanced accuracy and generalizability. Experimental results highlight our approach's superior performance in comparison to existing state-of-the-art methods, which are typically optimized for a single model type. Our findings issue a stark warning about the ease with which high-quality prompts can be extracted, regardless of the generative model type employed. 


## Dataset

https://huggingface.co/datasets/cyprivlab/Regen_COCO_prompts
https://huggingface.co/datasets/cyprivlab/Regen_SatableDiffusion
