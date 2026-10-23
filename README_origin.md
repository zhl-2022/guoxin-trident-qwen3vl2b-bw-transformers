---
license: mit
language:
  - en
pipeline_tag: feature-extraction
base_model: Qwen/Qwen3-VL-2B-Instruct
base_model_relation: finetune
datasets:
  - OpenBMB/Trident-Train
tags:
  - multimodal-embedding
  - mixed-modality-retrieval
  - multimodal-retrieval
  - visual-document-retrieval
  - qwen3-vl
---

<h1 align="center">Trident-Qwen3VL-2B</h1>

<p align="center">
  Official model for <a href="https://arxiv.org/abs/2610.11816"><i>Chaos in the Text: Revealing the Modality Preference in Mixed-Modality Retrievers</i></a>
</p>
<p align="center">
  <a href="https://chaos-in-texts.github.io/"><img src="https://img.shields.io/badge/%F0%9F%94%B1%20Project%20Page-chaos--in--texts-5b3fd1.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2610.11816"><img src="https://img.shields.io/badge/arXiv-2610.11816-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/OpenBMB/Trident"><img src="https://img.shields.io/badge/GitHub-OpenBMB%2FTrident-blue.svg" alt="Code"></a>
  <a href="https://huggingface.co/OpenBMB/Trident-Qwen3VL-2B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Trident--Qwen3VL--2B-yellow.svg" alt="Model"></a>
  <a href="https://huggingface.co/datasets/OpenBMB/Trident-Train"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Data-Trident--Train-orange.svg" alt="Train data"></a>
  <a href="https://huggingface.co/datasets/OpenBMB/Trident-Eval"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Data-Trident--Eval-orange.svg" alt="Eval data"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

**Trident-Qwen3VL-2B** is a multimodal embedding model for **mixed-modality retrieval**. It maps text, images, and fused text-image inputs into one shared space, so a text query can rank text-only, image-only, and fused documents in a single list. It is fine-tuned from [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) on [Trident-Train](https://huggingface.co/datasets/OpenBMB/Trident-Train), as described in the paper [Chaos in the Text: Revealing the Modality Preference in Mixed-Modality Retrievers](https://arxiv.org/abs/2610.11816). The LoRA adapter is already merged, so no base model or [PEFT](https://github.com/huggingface/peft) is needed.

## Quickstart

**1. Set up the code and download the weights**

```bash
git clone https://github.com/OpenBMB/Trident.git
cd Trident
conda create -n trident python=3.12 -y && conda activate trident
pip install -r requirements.txt     # tested with transformers 4.57.1 (Qwen3-VL needs >= 4.57)

hf download OpenBMB/Trident-Qwen3VL-2B --local-dir checkpoints/Trident-Qwen3VL-2B
```

**2. Encode queries and documents** (run from the root of the Trident repository)

```python
import sys
sys.path.insert(0, "src")

from embedding_backends.common import (
    ensure_tokenizers_thread_safety,
    ensure_transformers_special_tokens_compat,
)
from embedding_backends.trident_qwen3vl import TridentQwen3VLBackend

ensure_transformers_special_tokens_compat()
ensure_tokenizers_thread_safety()

CKPT = "checkpoints/Trident-Qwen3VL-2B"

backend = TridentQwen3VLBackend(
    {"model_type": "trident_qwen3vl", "checkpoint": CKPT, "dtype": "bfloat16"},
    device="cuda:0",
)
backend.load()


def encode(items, role):
    """items: list of {"text": str | None, "images": [paths]}; role: "query" or "doc"."""
    backend.args_dict["role"] = role
    instruction = backend.default_instruction(role)
    inputs = [{**item, "instruction": instruction} for item in items]
    embeddings, _ = backend.process(inputs, role=role)
    return embeddings  # [N, dim] numpy array, L2-normalised


queries = [
    {"text": "How many stores did Saint Laurent operate in Western Europe in 2020?", "images": []},
]

documents = [
    {"text": "A bar chart showing the number of Saint Laurent stores by region and year.", "images": []},  # text-only
    {"text": None, "images": ["figs/example_chart.png"]},                                                # image-only
    {"text": "Number of stores by region.", "images": ["figs/example_chart.png"]},                       # text + image
]

q = encode(queries, role="query")      # [1, dim]
d = encode(documents, role="doc")      # [3, dim]

scores = q @ d.T                       # cosine similarity, comparable across all document types
print(scores)
print(scores[0].argsort()[::-1])       # document ranking for the query
```

Each input is a dict with an optional `text` string and an `images` list of local image paths. Use `text` only, `images` only, or both for fused inputs. Queries and documents use different default instructions, which are part of the training format.

## License

The Trident-specific contributions, including the fine-tuned weights, are released under the [MIT License](https://github.com/OpenBMB/Trident/blob/main/LICENSE). The model is fine-tuned from [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), which is released under Apache-2.0, and the base model's license terms continue to apply to the weights derived from it. The training images come from third-party datasets (see [Trident-Train](https://huggingface.co/datasets/OpenBMB/Trident-Train)) and remain subject to their original licenses.

## Citation

If you find this work useful, please cite:

```bibtex
@article{sun2026chaos,
  title={Chaos in the Text: Revealing the Modality Preference in Mixed-Modality Retrievers},
  author={Sun, Yubo and Peng, Chunyi and Yan, Yukun and Liu, Zhenghao and Xu, Zhipeng and Mei, Sen and Xin, Linlin and Zeng, Zheni and Sun, Maosong},
  journal={arXiv preprint arXiv:2610.11816},
  year={2026}
}
```
