---
license: apache-2.0
language:
- en
tags:
- multimodal
- image-generation
- image-understanding
- image-editing
- diffusion
- moe
- text-to-image
library_name: transformers
pipeline_tag: any-to-any
base_model:
- inclusionAI/LLaDA2.0-mini
---

<p align="center">
 <img src="./assets/llada_logo.png" width="20%"/>
</p>
<div align="center">
 <h1> LLaDA2.0-Uni: Unifying Multimodal Understanding and Generation with Diffusion Large Language Model </h1>

  [[📑 Technical Report ](https://arxiv.org/abs/2604.20796)] &emsp; [[🌐 Github ](https://github.com/inclusionAI/LLaDA2.0-Uni)]
 
 <b>AGI Research Center, Inclusion AI </b>
</div>

## Model Capabilities

**LLaDA2.0-Uni** is a unified diffusion Large Language Model (dLLM) based on Mixture-of-Experts (MoE) that seamlessly integrates multimodal understanding and generation within a single model. It supports:

- 🖼️ **Text-to-Image Generation** — high-fidelity image synthesis with optional thinking/reasoning.
- 🔍 **Image Understanding** — visual question answering, image captioning, document understanding, etc.
- ✏️ **Image Editing** — instruction-based editing with single or multi-reference support.
- 🎨 **Interleaved Generation and Reasoning** — provide preliminary support for interleaved generation and unlock advanced interleaved reasoning.
- ⚡ **Sprint Acceleration** — KV cache reuse and adaptive unmasking for faster inference.

## Model Architecture

<img src="./assets/architecture.png" width="100%"/>

- **Unified dLLM-MoE Backbone**: Unifies multimodal understanding and generation into a simple Mask Token Prediction paradigm.
- **Discrete Semantic Tokenizer**: Utilizes SigLIP-VQ to convert visual inputs into discrete semantic tokens, significantly enhancing multimodal understanding.
- **Efficient Diffusion Decoder**: Pairs discrete tokens with a specialized diffusion decoder for high-fidelity generation, enabling rapid 8-step inference via distillation.

## Evaluation Results

<img src="./assets/performance.png" width="100%"/>

## Quick Start

> **Note:** Full installation instructions and CLI scripts are available in the [GitHub repository](https://github.com/inclusionAI/LLaDA2-Uni).

### ⚙️ Installation

#### 1. Create a conda environment

```bash
git clone https://github.com/inclusionAI/LLaDA2-Uni && cd LLaDA2-Uni
conda create -n llada2_uni python=3.10 -y
conda activate llada2_uni
```

#### 2. Install PyTorch (CUDA 12.4)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### 3. Install Flash Attention 2 (required for efficient inference)

```bash
pip install flash-attn --no-build-isolation
```

#### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 🌟 Text-to-Image Generation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from decoder import decode_vq_tokens

model_path = "inclusionAI/LLaDA2.0-Uni"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="cuda", torch_dtype="bfloat16", trust_remote_code=True
).eval()
model.tokenizer = tokenizer

# Generate image tokens
result = model.generate_image(
    "A modern Scandinavian kitchen with white cabinetry, marble countertops, and a single orchid on the island. A Nordic woman with sleek blonde ponytail, wearing an oversized sweater and dainty silver necklaces, stirs a matcha bowl with a bamboo whisk, eyes sparkling with quiet joy. Shot with 50mm, f/2.5, diffused window light, cool white balance, low saturation, clean skin retouch. Mood: serene, wholesome, hygge.",
    image_h=1024, image_w=1024,
    steps=8, cfg_scale=2.0,
)

# Decode to PIL image (default: 50-step ODE)
image = decode_vq_tokens(result["token_ids"], result["h"], result["w"], model_path, "cuda")
image.save("output.png")
```

> [!Note]
>  💡 **Faster decoding** — Use the **decoder-turbo** (distilled decoder) for **~10× faster** image decoding (8 steps instead of 50) with minimal quality loss:
> ```python
> image = decode_vq_tokens(
>     result["token_ids"], result["h"], result["w"], model_path, "cuda",
>     num_steps=8, decode_mode="decoder-turbo",
> )
> ```

### 🌟 Text-to-Image Generation with Thinking

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from decoder import decode_vq_tokens

model_path = "inclusionAI/LLaDA2.0-Uni"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="cuda", torch_dtype="bfloat16", trust_remote_code=True
).eval()
model.tokenizer = tokenizer

# Generate image tokens with thinking process
result = model.generate_image(
    "A fox with thick, dense, fluffy fur in a winter setting, possibly surrounded by snow.",
    image_h=1024, image_w=1024,
    mode="thinking",
    steps=8, cfg_scale=2.0,
    thinking_steps=32, thinking_gen_length=4096,
)

# Print thinking trace
print("Thinking:", result["thinking"])

# Decode to PIL image
image = decode_vq_tokens(result["token_ids"], result["h"], result["w"], model_path, "cuda", num_steps=8, decode_mode="decoder-turbo",)
image.save("output_thinking.png")
```

### 🌟 Image Understanding

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from encoder.image_tokenizer import ImageTokenizer
from decoder.smart_img_process import smart_resize_images

model_path = "inclusionAI/LLaDA2.0-Uni"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="cuda", torch_dtype="bfloat16", trust_remote_code=True
).eval()
model.tokenizer = tokenizer

# Encode image to discrete tokens
image_tokenizer = ImageTokenizer(model_path=model_path, device="cuda")
pil_image = smart_resize_images(["./assets/understanding_example.png"])[0]
info = image_tokenizer.encode_with_info(pil_image)
image_tokens = [x + model.config.image_token_offset for x in info["token_ids"]]
_, h, w = info["grid_thw"]

# Understand the image
response = model.understand_image(
    image_tokens, h, w,
    question="Describe this image in detail.",
    steps=32, gen_length=2048,
)
print(response)
```

### 🌟 Image Editing

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from encoder.image_tokenizer import ImageTokenizer
from decoder.utils import generate_crop_size_list, var_center_crop
from decoder import decode_vq_tokens
from PIL import Image

model_path = "inclusionAI/LLaDA2.0-Uni"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="cuda", torch_dtype="bfloat16", trust_remote_code=True
).eval()
model.tokenizer = tokenizer

# Encode source image
image_tokenizer = ImageTokenizer(model_path=model_path, device="cuda")
crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
pil_image = var_center_crop(Image.open("./assets/edit_example.png").convert("RGB"), crop_size_list=crop_size_list)
info = image_tokenizer.encode_with_info(pil_image)
image_tokens = [x + model.config.image_token_offset for x in info["token_ids"]]
_, h, w = info["grid_thw"]

# Edit the image
result = model.edit_image(
    image_tokens, h, w,
    instruction="Change the background to a beach.",
    steps=8, cfg_text_scale=4.0,
)

# Decode to PIL image
edited_image = decode_vq_tokens(result["token_ids"], result["h"], result["w"], model_path, "cuda", num_steps=8, decode_mode="decoder-turbo",)
edited_image.save("edited.png")
```

### 🌟 SPRINT Acceleration

SPRINT accelerates inference by combining **KV cache reuse**, **adaptive unmasking**, and **threshold-based batch acceptance**:

- **KV Cache Reuse & Pruning**: The prefix KV cache is computed once during warmup steps, then optionally pruned by importance scores (blending KV attention importance with token confidence). Subsequent denoising steps reuse the cached prefix, significantly reducing computation. Per-modality keep ratios (`image_keep_ratio`, `text_keep_ratio`) allow fine-grained control — e.g., retaining all image/text tokens for quality while still benefiting from cache reuse.
- **Adaptive Unmasking**: Instead of unmasking a fixed number of tokens per step, Sprint dynamically decides how many tokens to reveal based on model confidence. At each step, it computes confidence scores (via strategies like `low_confidence`, `top_k_margin`, or `neg_entropy`) and transfers the top-k most confident tokens, where k is adaptively set as `ceil(remaining_masked / steps_left)`. This allows easy positions to be resolved quickly while concentrating compute on harder tokens.
- **Batch Acceptance**: On top of adaptive scheduling, all tokens whose probability exceeds `threshold` are accepted in batch, further reducing the number of denoising iterations needed.

**Image Understanding** with Sprint:

```python
response = model.understand_image(
    image_tokens, h, w,
    question="Describe this image in detail.",
    steps=32, gen_length=4096,
    use_sprint=True,
    threshold=0.93,
    keep_ratio=0.5,
    cache_warmup_steps=1,
    image_keep_ratio=1.0,
    text_keep_ratio=1.0,
)
```

**Text-to-Image** with Sprint:

```python
result = model.generate_image(
    "A modern Scandinavian kitchen with white cabinetry, marble countertops, and a single orchid on the island. A Nordic woman with sleek blonde ponytail, wearing an oversized sweater and dainty silver necklaces, stirs a matcha bowl with a bamboo whisk, eyes sparkling with quiet joy. Shot with 50mm, f/2.5, diffused window light, cool white balance, low saturation, clean skin retouch. Mood: serene, wholesome, hygge.",
    image_h=1024, image_w=1024,
    cfg_scale=2.0,
    use_sprint=True,
    block_length=32,
    steps=8,
    keep_ratio=0.5,
    cache_warmup_steps=1,
)
```

> [!Note]
>  Sprint is supported for Simple CFG and no-CFG modes. When using Editing CFG (three-way guidance with `cfg_text_scale` / `cfg_image_scale`), Sprint automatically falls back to baseline.

## Repository Structure

```
LLaDA2-Uni/
├── config.json                          # Model configuration
├── modeling_llada2uni_moe.py            # Model implementation (trust_remote_code)
├── configuration_llada2uni_moe.py       # Config class
├── tokenizer.json                       # Tokenizer
├── model-00001-of-00013.safetensors     # MoE backbone weights (sharded, bf16)
├── ...
├── model-00013-of-00013.safetensors
├── model.safetensors.index.json
├── image_tokenizer/
│   ├── config.json
│   ├── image_tokenizer.safetensors      # SigLIP-VQ encoder
│   ├── sigvq_embedding.pt               # SigVQ embedding + projector
│   └── preprocessor_config.json
├── decoder/
│   ├── config.json
│   └── decoder_model.safetensors        # Diffusion decoder (bf16, 12GB)
├── decoder-turbo/
│   ├── config.json
│   └── decoder_model.safetensors        # Distilled few-step decoder (bf16, 12GB)
└── vae/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

## Hardware Requirements

| Component | GPU Memory |
|---|---|
| MoE Backbone (bf16, 16B total) | ~32 GB |
| Diffusion Decoder (bf16, 6.2B) | ~12 GB |
| VAE + SigVQ + Tokenizer | ~3 GB |
| **Total (generation/editing)** | **~47 GB** |
| **Total (understanding only)** | **~35 GB** |

> 💡 While only ~1B parameters are activated per token during inference, all 16B MoE parameters must be loaded into memory. The diffusion decoder is only needed for image generation/editing and is released afterwards.

## 🚀 SGLang Support (Coming Soon)

We are working on integrating [SGLang](https://github.com/sgl-project/sglang) for high-throughput serving and optimized inference. Stay tuned!

## ⚠️ License

This project is licensed under the terms of the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## 📖 BibTeX

```bibtex
@article{LLaDA2Uni,
title = {LLaDA2.0-Uni: Unifying Multimodal Understanding and Generation with Diffusion Large Language Model},
author = {Tiwei Bie and Haoxing Chen and Tieyuan Chen and Zhenglin Cheng and Long Cui and Kai Gan and Zhicheng Huang and Zhenzhong Lan and Haoquan Li and Jianguo Li and Tao Lin and Qi Qin and Hongjun Wang and Xiaomei Wang and Haoyuan Wu and Yi Xin and Junbo Zhao},
journal = {arXiv preprint arXiv:2604.20796},
year = {2026}
}
```