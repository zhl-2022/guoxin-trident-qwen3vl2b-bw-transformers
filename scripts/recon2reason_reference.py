#!/usr/bin/env python3
"""Bounded local Transformers checks; not a server or an accuracy evaluation.

No dependencies are installed and no model files are downloaded. CPU preflight
imports the native stack and tokenizes images without loading model weights.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings


MAX_SOURCE_PIXELS = 16_777_216
MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_IMAGES = 8
MAX_INPUT_TENSOR_BYTES = 128 * 1024 * 1024
MAX_PROMPT_TOKENS = 8192


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="existing local model directory")
    parser.add_argument("--image", type=Path, action="append", default=[],
                        help="repeat for at least two different local images")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; never overwritten")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--sdpa-backend", choices=("auto", "math"), default="auto",
                        help="auto dispatch (default), or explicitly enable only the SDPA math backend")
    parser.add_argument("--max-image-pixels", type=int, default=262144,
                        help="processed pixels per image, 1024..1048576 (default: 262144)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="CPU imports and single/two-image tokenization; no weights or CUDA checks")
    parser.add_argument("--image-only-processor", action="store_true",
                        help="explicit 4.57.6 image-only processor; omit unused torchvision video component")
    parser.add_argument("--create-fixtures", type=Path, metavar="NEW_DIRECTORY",
                        help="create two functional Pillow fixtures in a new independent directory")
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    if not 1 <= args.max_new_tokens <= 1024:
        parser.error("max-new-tokens must be between 1 and 1024")
    if not 1024 <= args.max_image_pixels <= 1048576:
        parser.error("max-image-pixels must be between 1024 and 1048576")
    if args.output.exists():
        parser.error("output already exists; choose a new JSON filename")
    if not args.output.parent.is_dir():
        parser.error("output parent directory must already exist")
    if not args.create_fixtures and not 2 <= len(args.image) <= MAX_IMAGES:
        parser.error(f"provide 2..{MAX_IMAGES} images, or use --create-fixtures")
    if args.create_fixtures and args.image:
        parser.error("use either --image or --create-fixtures for this run")
    if args.create_fixtures:
        args.create_fixtures = args.create_fixtures.resolve()
        if args.create_fixtures.exists() or not args.create_fixtures.parent.is_dir():
            parser.error("fixture directory must be new and its parent must exist")
        if args.create_fixtures == args.model or args.model in args.create_fixtures.parents:
            parser.error("fixture directory must be outside the model directory")
    return args


def create_fixtures(directory: Path, image_module, draw_module) -> list[Path]:
    directory.mkdir(exist_ok=False)
    paths = []
    for index, color in enumerate(("red", "blue"), start=1):
        path = directory / f"functional_fixture_{index}.png"
        image = image_module.new("RGB", (384, 256), "white")
        draw = draw_module.Draw(image)
        if index == 1:
            draw.rectangle((35, 70, 155, 190), fill=color)
        else:
            draw.ellipse((225, 70, 345, 190), fill=color)
        draw.text((16, 16), f"FUNCTIONAL FIXTURE {index} - {color.upper()}", fill="black")
        image.save(path, format="PNG")
        image.close()
        paths.append(path)
    return paths


def read_image(path: Path, pixel_limit: int, image_module, image_ops) -> tuple:
    path = path.resolve(strict=True)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_SOURCE_BYTES:
        raise ValueError(f"Image must be a nonempty local file no larger than 20 MiB: {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("error", image_module.DecompressionBombWarning)
        with image_module.open(path) as source:
            if source.width * source.height > MAX_SOURCE_PIXELS:
                raise ValueError(f"Source image exceeds {MAX_SOURCE_PIXELS} pixels: {path}")
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError(f"Only single-frame images are accepted: {path}")
            original_size = list(source.size)
            # Bound decompression first; shrink before RGB conversion/processor copies.
            factor = min(1.0, math.sqrt(pixel_limit / (source.width * source.height)))
            target = (max(1, int(source.width * factor)), max(1, int(source.height * factor)))
            source.thumbnail(target, resample=image_module.Resampling.LANCZOS)
            oriented = image_ops.exif_transpose(source)
            image = oriented.convert("RGB")
            if oriented is not source:
                oriented.close()
    return image, {
        "path": str(path), "sha256": sha256(path), "file_bytes": path.stat().st_size,
        "original_size_wh": original_size, "bounded_size_wh": list(image.size),
        "bounded_pixels": image.width * image.height,
    }


def prepare(processor, images: list, prompt: str, pixel_limit: int, torch_module) -> tuple:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        images_kwargs={"min_pixels": 1024, "max_pixels": pixel_limit},
    )
    token_length = int(inputs["input_ids"].shape[-1])
    tensor_bytes = sum(value.numel() * value.element_size() for value in inputs.values()
                       if torch_module.is_tensor(value))
    grid = inputs.get("image_grid_thw")
    if grid is None or len(grid) != len(images):
        raise ValueError("Processor did not produce one image_grid_thw row per input image")
    if token_length > MAX_PROMPT_TOKENS or tensor_bytes > MAX_INPUT_TENSOR_BYTES:
        raise ValueError("Processor output exceeds bounded prompt/tensor limits")
    # Qwen3-VL grid dimensions count patches before spatial merging.
    patch_size = getattr(processor.image_processor, "patch_size", 16)
    if any(int(row[1]) * int(row[2]) * patch_size**2 > pixel_limit for row in grid):
        raise ValueError("Processor exceeded max-image-pixels; refusing an unbounded forward pass")
    return inputs, {
        "prompt": prompt, "rendered_chat_template": rendered,
        "prompt_tokens": token_length, "image_grid_thw": grid.tolist(),
        "input_tensor_bytes": tensor_bytes,
        "tensor_shapes": {name: list(value.shape) for name, value in inputs.items()
                          if torch_module.is_tensor(value)},
    }


def load_processor(model_path: Path, transformers_module, image_only: bool) -> tuple:
    """Select an explicit path; never retry a failed AutoProcessor silently.

    The image-only extension is pinned to the inspected upstream 4.57.6 API:
    ProcessorMixin builds components from cls.attributes and loads the original
    chat template through get_processor_dict. Qwen3VL's image computation is
    inherited unchanged. No global class, file, package, or registry is patched.
    """
    if not image_only:
        processor = transformers_module.AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
        )
        return processor, {"mode": "AutoProcessor", "explicit_image_only": False}
    if transformers_module.__version__ != "4.57.6":
        raise RuntimeError("The explicit image-only extension was audited only for transformers 4.57.6")

    class ImageOnlyQwen3VLProcessor(transformers_module.Qwen3VLProcessor):
        # Native Qwen3VL.__init__ delegates to ProcessorMixin, which validates
        # these two components. Declaring None as a third component would fail
        # its BaseVideoProcessor type check when torchvision is absent.
        attributes = ["image_processor", "tokenizer"]
        video_processor = None

        @classmethod
        def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            image_processor = transformers_module.AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path, use_fast=False, **kwargs,
            )
            # Match the native Qwen3VL default fast tokenizer. Passing
            # use_fast=False to the whole processor would also change this.
            tokenizer = transformers_module.Qwen2TokenizerFast.from_pretrained(
                pretrained_model_name_or_path, **kwargs,
            )
            return [image_processor, tokenizer]

        def __call__(self, *args, **kwargs):
            if kwargs.get("videos") is not None:
                raise ValueError("This explicit image-only processor does not accept videos")
            return super().__call__(*args, **kwargs)

    template_path = model_path / "chat_template.jinja"
    if not template_path.is_file() or template_path.stat().st_size > 1024 * 1024:
        raise ValueError("Image-only mode requires the original local chat_template.jinja, at most 1 MiB")
    expected_template = template_path.read_text(encoding="utf-8")
    processor = ImageOnlyQwen3VLProcessor.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
    )
    if processor.chat_template != expected_template:
        raise RuntimeError("Loaded processor chat template differs from the saved chat_template.jinja")
    return processor, {
        "mode": "explicit_image_only_Qwen3VLProcessor_extension",
        "explicit_image_only": True, "audited_transformers_version": "4.57.6",
        "video_component_loaded": False, "video_inputs_supported": False,
        "image_processor_class": type(processor.image_processor).__name__,
        "image_processor_use_fast": False,
        "tokenizer_class": type(processor.tokenizer).__name__,
        "chat_template_file": str(template_path), "chat_template_file_sha256": sha256(template_path),
        "loaded_chat_template_sha256": hashlib.sha256(processor.chat_template.encode("utf-8")).hexdigest(),
        "chat_template_matches_saved_file": True,
        "tokenizer_json_sha256": sha256(model_path / "tokenizer.json"),
        "preprocessor_config_sha256": sha256(model_path / "preprocessor_config.json"),
        "accuracy_equivalence_verified": False,
    }


def execute(args: argparse.Namespace, report: dict, persist) -> None:
    report["stage"] = "imports"
    persist()
    import torch
    import transformers
    from PIL import Image, ImageDraw, ImageOps
    from transformers import Qwen3VLForConditionalGeneration

    report["runtime"] = {
        "torch_version": torch.__version__, "torch_file": torch.__file__,
        "torch_hip": torch.version.hip, "torch_cuda_build": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "transformers_file": transformers.__file__,
        "cuda_checked": False,
    }
    if not args.model.is_dir():
        raise ValueError(f"Model must be an existing local directory: {args.model}")
    config_path = args.model / "config.json"
    if config_path.stat().st_size > 1024 * 1024:
        raise ValueError("Model config.json exceeds 1 MiB")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("model_type") != "qwen3_vl"
            or "Qwen3VLForConditionalGeneration" not in config.get("architectures", [])):
        raise ValueError("Expected the native qwen3_vl / Qwen3VLForConditionalGeneration architecture")
    report["model_config"] = {key: config.get(key) for key in
                              ("model_type", "architectures", "transformers_version")}
    report["config_sha256"] = sha256(config_path)
    report["stage"] = "images"
    persist()
    image_paths = args.image
    if args.create_fixtures:
        image_paths = create_fixtures(args.create_fixtures, Image, ImageDraw)
        report["fixtures"] = {"directory": str(args.create_fixtures), "accuracy_evidence": False}
    paths = [path.resolve(strict=True) for path in image_paths]
    if not 2 <= len(set(paths)) <= MAX_IMAGES:
        raise ValueError(f"Provide 2..{MAX_IMAGES} distinct image paths")
    images = []
    try:
        report["images"] = []
        for path in paths:
            image, source = read_image(path, args.max_image_pixels, Image, ImageOps)
            images.append(image)
            report["images"].append(source)
        if len({source["sha256"] for source in report["images"]}) < 2:
            raise ValueError("At least two images must have different contents")
        report["stage"] = "processor_load"
        persist()
        processor, processor_details = load_processor(args.model, transformers, args.image_only_processor)
        report["processor_class"] = type(processor).__name__
        report["processor_path"] = processor_details
        cases = [(f"single_image_{index + 1}", [index],
                  "Describe the visible objects, colors and relative positions in this image.")
                 for index in range(len(images))]
        cases.append(("two_images", [0, 1],
                      "Compare the visible objects and their relative positions in the two images."))
        report["cases"] = []
        report["stage"] = "cpu_tokenization"
        for name, indices, prompt in cases:
            inputs, details = prepare(processor, [images[i] for i in indices], prompt,
                                      args.max_image_pixels, torch)
            report["cases"].append({"name": name, "image_indices": indices,
                                    "preflight": details, "generation": None})
            del inputs
            persist()
        if args.preflight_only:
            report["status"] = "cpu_preflight_passed"
            report["weights_loaded"] = False
            return

        report["stage"] = "device_check"
        report["runtime"]["cuda_checked"] = True
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA/HIP device unavailable; weights will not be loaded")
        torch.cuda.set_device(0)
        report["device"] = {
            "name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count(),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
        if not report["device"]["bf16_supported"]:
            raise RuntimeError("Device does not report BF16 support; no precision fallback is allowed")
        probe = torch.ones((16, 16), dtype=torch.bfloat16, device="cuda:0")
        product = probe @ probe
        torch.cuda.synchronize()
        if product.dtype != torch.bfloat16 or not bool(torch.isfinite(product).all().item()):
            raise RuntimeError("BF16 device matmul probe failed")
        del product, probe
        report["stage"] = "weight_load"
        persist()
        started = time.perf_counter()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.model), dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True, trust_remote_code=False,
        ).to("cuda:0").eval()
        torch.cuda.synchronize()
        report["model_load_seconds"] = time.perf_counter() - started
        report["weights_loaded"] = True
        if any(parameter.is_floating_point() and parameter.dtype != torch.bfloat16
               for parameter in model.parameters()):
            raise RuntimeError("Loaded model has floating parameters outside BF16")
        report["attention_implementation"] = {
            name: getattr(getattr(model.config, name, model.config), "_attn_implementation", None)
            for name in ("text_config", "vision_config")
        }
        if any(value != "sdpa" for value in report["attention_implementation"].values()):
            raise RuntimeError("Model did not retain SDPA for both text and vision attention")
        report["stage"] = "generation"
        persist()
        for result, (_, indices, prompt) in zip(report["cases"], cases):
            report["active_case"] = result["name"]
            persist()
            inputs, _ = prepare(processor, [images[i] for i in indices], prompt,
                                args.max_image_pixels, torch)
            device_inputs = {
                name: value.to(device="cuda:0", dtype=torch.bfloat16 if value.is_floating_point()
                               else value.dtype) if torch.is_tensor(value) else value
                for name, value in inputs.items()
            }
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
            baseline_memory = torch.cuda.memory_allocated(0)
            attention_context = nullcontext()
            if args.sdpa_backend == "math":
                from torch.nn.attention import SDPBackend, sdpa_kernel
                attention_context = sdpa_kernel([SDPBackend.MATH])
            started = time.perf_counter()
            with torch.inference_mode():
                with attention_context:
                    output = model.generate(**device_inputs, do_sample=False,
                                            max_new_tokens=args.max_new_tokens)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak_allocated = torch.cuda.max_memory_allocated(0)
            peak_reserved = torch.cuda.max_memory_reserved(0)
            sequence_ids = output[0].detach().cpu().tolist()
            generated_ids = sequence_ids[inputs["input_ids"].shape[-1]:]
            if not generated_ids:
                raise RuntimeError("Generation returned no tokens")
            result["generation"] = {
                "do_sample": False, "max_new_tokens": args.max_new_tokens, "dtype": "bfloat16",
                "sdpa_backend_requested": args.sdpa_backend,
                "synchronized_generate_seconds": elapsed,
                "timing_scope": "generate only; excludes loading, preprocessing and input transfer; no warmup",
                "baseline_allocated_bytes": baseline_memory,
                "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                "generated_tokens": len(generated_ids), "generated_token_ids": generated_ids,
                "full_sequence_token_ids": sequence_ids,
                "raw_generated_text": processor.decode(generated_ids, skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False),
                "answer": processor.decode(generated_ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False),
            }
            del output, device_inputs, inputs
            persist()
            if not result["generation"]["answer"].strip():
                raise RuntimeError("Decoded answer is empty; raw tokens are retained in the report")
        report.pop("active_case", None)
        report["status"] = "functional_generation_completed"
    finally:
        for image in images:
            image.close()


def main() -> int:
    args = arguments()
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "Pillow", "accelerate", "safetensors"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "schema_version": 1, "created_at_utc": stamp(), "status": "running", "stage": "start",
        "scope": "native Transformers functional reference; not an official framework service or accuracy pass",
        "official_accuracy": False, "official_framework_service": False,
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": packages, "model_directory": str(args.model),
        "script_sha256": sha256(Path(__file__).resolve()),
        "sdpa_backend_requested": args.sdpa_backend,
        "preflight_only": args.preflight_only, "weights_loaded": False,
        "image_only_processor_requested": args.image_only_processor,
        "limits": {"max_image_pixels": args.max_image_pixels, "max_source_pixels": MAX_SOURCE_PIXELS,
                   "max_source_bytes": MAX_SOURCE_BYTES, "max_images": MAX_IMAGES,
                   "max_prompt_tokens": MAX_PROMPT_TOKENS,
                   "max_input_tensor_bytes": MAX_INPUT_TENSOR_BYTES},
    }
    # Exclusive creation reserves this path. Subsequent writes update only this run's new file.
    with args.output.open("x", encoding="utf-8") as handle:
        def persist() -> None:
            handle.seek(0)
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.truncate()
            handle.flush()

        code = 0
        try:
            execute(args, report, persist)
            report["stage"] = "complete"
        except (Exception, KeyboardInterrupt) as exc:
            report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            report["error"] = {"type": type(exc).__name__, "message": str(exc),
                               "traceback": traceback.format_exc()}
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        report["finished_at_utc"] = stamp()
        persist()
    print(json.dumps({"status": report["status"], "stage": report["stage"],
                      "output": str(args.output)}, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
