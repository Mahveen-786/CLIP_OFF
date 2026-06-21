# CLIPOFF 
## Benchmarking CLIP · BLIP · ALIGN · FLAVA on Vision-Language Tasks

A unified evaluation framework comparing 4 state-of-the-art Vision-Language Models on zero-shot classification and image-text retrieval.

**Research Question:** *Which VLM generalises best — and what are the real tradeoffs between speed, accuracy and task type?*

## Demo
[![CLIPOFF Demo](https://img.youtube.com/vi/ez0qJ6AEmVE/0.jpg)](https://youtube.com/watch?v=ez0qJ6AEmVE)

> Click to watch the demo

🚀 [Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/mahveen123/CLIP-OFF)

---

## Results

### Zero-Shot Classification (CIFAR-10 · 1000 images · 10 classes)
| Model | Organisation | Zero-shot Accuracy | Inference |
|---|---|---|---|
| **CLIP ViT-B/16** | OpenAI | **89.9%**  | 23.8 ms/img |
| FLAVA-full | Meta | 87.8% | 843.4 ms/img |
| BLIP-base | Salesforce | 80.5% | 48.2 ms/img |
| ALIGN-base | Google | 77.4% | 47.6 ms/img |

### Image-Text Retrieval (Flickr30k · 1000 images)
| Model | R@1 | R@5 | R@10 | Inference |
|---|---|---|---|---|
| **BLIP-base** | **86.8%**  | **97.8%** | **98.5%** | 48.2 ms/img |
| ALIGN-base | 81.6% | 96.6% | 97.9% | 47.6 ms/img |
| FLAVA-full | 68.8% | 92.2% | 96.3% | 843.4 ms/img |
| CLIP ViT-B/16 | 70.4% | 91.5% | 95.7% | 23.8 ms/img |

---

## Key Findings

**1. CLIP leads zero-shot classification (89.9%)** — OpenAI's 400M-pair pretraining gives it the strongest zero-shot visual representations across diverse categories.

**2. BLIP leads image-text retrieval (R@1 86.8%)** — Salesforce's bootstrapped captioning approach produces tighter image-text alignment, making it the best retrieval model.

**3. FLAVA is competitive but 18× slower** — 87.8% zero-shot accuracy but 843ms per image vs CLIP's 24ms. Accurate but not production-ready for latency-sensitive applications.

**4. No single model wins all tasks** — model selection should depend on use case: CLIP for speed + classification, BLIP for retrieval, FLAVA when accuracy matters more than speed.

---

## Models

| Model | Organisation | Pretraining Data | Architecture |
|---|---|---|---|
| CLIP ViT-B/16 | OpenAI | 400M image-text pairs | Dual encoder |
| BLIP-base | Salesforce | 129M + bootstrapped | Encoder-decoder |
| ALIGN-base | Google | 1.8B noisy pairs | Dual encoder |
| FLAVA-full | Meta | Multiple sources | Unified encoder |

---

## Evaluation Setup

**Task 1 — Zero-shot Image Classification**
- Dataset: CIFAR-10 (1000 images, 100 per class)
- Method: Cosine similarity between image embeddings and class text embeddings
- Prompt ensembling across 4 templates for all models

**Task 2 — Image-Text Retrieval**
- Dataset: Flickr30k subset (1000 images, 1 caption per image)
- Metrics: Recall@1, Recall@5, Recall@10
- Image-to-text retrieval evaluated for all models

---

## Setup

```bash
git clone https://github.com/Mahveen-786/vlm-arena
cd vlm-arena
pip install torch torchvision transformers gradio huggingface_hub Pillow numpy timm open_clip_torch scikit-learn
```

Open `VLM_Arena.ipynb` in Google Colab (T4 GPU recommended). Runtime: ~45-60 minutes.

---

## Tech Stack
PyTorch · HuggingFace Transformers · open_clip · CIFAR-10 · Flickr30k · Gradio

---

*Built as part of research preparation in vision-language models and multimodal learning.*
