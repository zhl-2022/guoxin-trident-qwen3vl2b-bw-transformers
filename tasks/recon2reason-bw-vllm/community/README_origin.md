---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3-VL-4B-Instruct
base_model_relation: finetune
tags:
  - qwen3-vl
  - vision-language
  - spatial-reasoning
  - spatial-understanding
  - embodied-ai
  - indoor-scenes
---

# Recon2Reason Reasoning 4B

Recon2Reason Reasoning 4B is a 4B-scale vision-language model specialized for spatial reasoning in indoor and embodied scenes. It is fine-tuned from Qwen3-VL-4B and improves reasoning about metric distance, relative position, object configuration, and spatial relations from visual inputs.

The checkpoint retains the standard `Qwen3VLForConditionalGeneration` architecture. No custom model code or `trust_remote_code=True` is required.

## Highlights

- Standard Qwen3-VL Transformers interface
- 4,437,815,808 parameters
- BF16 weights in sharded Safetensors format
- Single-image and multi-image visual inputs
- Strong results on metric and qualitative spatial reasoning benchmarks

This repository contains the reasoning model only. The retrieval-augmented scene-reconstruction extension is released separately.

## Model details

| Property | Value |
|---|---|
| Model name | Recon2Reason Reasoning 4B |
| Model type | Vision-language conditional generation model |
| Architecture | `Qwen3VLForConditionalGeneration` |
| Parameters | 4,437,815,808 |
| Weight dtype | BF16 |
| Weight format | Safetensors, 2 shards |
| Primary domain | Indoor spatial reasoning |
| Base model | Qwen3-VL-4B-Instruct |
| Tested Transformers version | 4.57.1 |
| License | Apache-2.0 |

## Quick start

```bash
pip install "transformers==4.57.1" "torch>=2.6" accelerate safetensors pillow
```

```python
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model_id = "BAAI/Recon2Reason-Reasoning-4B"

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
).eval()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/scene.jpg"},
            {
                "type": "text",
                "text": "Which object is closest to the chair? ",
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )

generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
answer = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]
print(answer)
```

The model uses the standard Qwen3-VL chat format. For benchmark reproduction, use greedy decoding (`do_sample=False`) unless a benchmark specifies otherwise.

## Evaluation

The checkpoint was evaluated in BF16 with PyTorch 2.8.0, Transformers 4.57.1, SDPA attention, and greedy decoding on NVIDIA RTX PRO 6000 Blackwell GPUs.

## License

The model is released under the Apache License 2.0, subject to final confirmation that all training data and upstream artifacts permit this distribution. See `LICENSE`.

## Acknowledgements

This work builds on [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct). We thank the creators of the evaluation datasets and the open-source Transformers ecosystem.
