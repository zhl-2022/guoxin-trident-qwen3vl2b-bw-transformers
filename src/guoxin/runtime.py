"""Bounded environment probes and explicit-file evidence manifests."""

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import display_path, file_hash, project_path

PACKAGES = (
    "torch", "torchvision", "transformers", "accelerate", "safetensors", "numpy",
    "vllm", "modelscope", "paddlepaddle", "onnxruntime",
)
OUTPUT_LIMIT = 16 * 1024


def probe(argv: list[str], timeout: float = 10) -> dict:
    if shutil.which(argv[0]) is None:
        return {"status": "unavailable", "command": argv[0]}
    try:
        completed = subprocess.run(
            argv, capture_output=True, timeout=timeout, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "command": argv[0], "timeout_seconds": timeout}
    except OSError as exc:
        return {"status": "error", "command": argv[0], "error_type": type(exc).__name__}
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "command": argv[0],
        "exit_code": completed.returncode,
        "stdout": completed.stdout[:OUTPUT_LIMIT],
        "stderr": completed.stderr[:OUTPUT_LIMIT],
        "output_truncated": len(completed.stdout) > OUTPUT_LIMIT or len(completed.stderr) > OUTPUT_LIMIT,
    }


def collect_environment() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    # Importing accelerator libraries runs in a child with a fixed timeout.
    torch_probe = "\n".join([
        "import json, torch",
        "out = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'hip_runtime': getattr(torch.version, 'hip', None), 'cuda_available': torch.cuda.is_available()}",
        "out['device_count'] = torch.cuda.device_count()",
        "out['devices'] = [torch.cuda.get_device_name(i) for i in range(out['device_count'])]",
        "print(json.dumps(out, ensure_ascii=False))",
    ])
    probes = {
        "nvidia": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
        "rocm": ["rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "vram"],
        "hygon": ["hy-smi", "--showproductname", "--showdriverversion"],
        "ascend": ["npu-smi", "info"],
    }
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "harness_version": __version__,
        "python": platform.python_version(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "packages": versions,
        "devices": {key: probe(argv) for key, argv in probes.items()},
        "torch_runtime": probe([sys.executable, "-c", torch_probe], timeout=20),
        "environment_variables_collected": False,
    }


def export_manifest(project: Path, config: dict, task_path: Path) -> dict:
    explicit = [(display_path(project, task_path), "task_config")]
    explicit.extend((item["path"], "artifact") for item in config["artifacts"])
    explicit.extend((item, "evidence") for item in config.get("evidence_files", []))
    model_config = config["model"].get("config_path")
    if model_config is not None:
        explicit.append((model_config, "model_config"))
    rows = {}
    missing = []
    for value, role in explicit:
        path = project_path(project, value)
        relative = display_path(project, path)
        if not path.is_file():
            missing.append({"path": relative, "reason": "missing_regular_file"})
            continue
        if relative in rows:
            if role not in rows[relative]["roles"]:
                rows[relative]["roles"].append(role)
            continue
        rows[relative] = {
            "path": relative, "roles": [role],
            "bytes": path.stat().st_size, "sha256": file_hash(path),
        }
    return {
        "schema_version": 1,
        "task": config["slug"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": not missing,
        "files": sorted(rows.values(), key=lambda row: row["path"]),
        "missing": missing,
        "scope": "only paths explicitly listed by the task configuration and task config itself",
        "official_evaluation": "not_run",
    }
