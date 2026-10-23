"""Build a bounded project bundle; never include credentials, weights, or arbitrary home files."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
allowed = [root / name for name in ("README.md", "pyproject.toml", ".gitignore")]
allowed.extend(root / name for name in (
    "evidence/model-config.json", "evidence/scnet-preflight.json",
    "evidence/reference-cpu-001.json", "evidence/reference-cpu-002.json",
    "evidence/reference-gpu-001.json", "evidence/reference-gpu-002.json",
    "evidence/scnet-cpu-tests.log", "evidence/rules/runtime-local-validation.json",
    "evidence/rules/validate_runtime_local.py", "evidence/rules/smoke-protocol-review.json",
    "evidence/vllm-probe-001.json", "evidence/vllm-state-001.json",
    "evidence/vllm-stopped-state-001.json", "evidence/vllm-smoke-001.jsonl",
    "evidence/vllm-server-full-001.log", "evidence/robobrain-model-config.json",
    "evidence/robobrain-preflight-001.json", "evidence/recon-preflight-hashed-001.json",
    "evidence/robobrain-preflight-hashed-001.json", "evidence/robobrain-vllm-state-001.json",
    "evidence/robobrain-vllm-stopped-state-001.json", "evidence/robobrain-vllm-smoke-001.jsonl",
    "evidence/robobrain-protocol-smoke-001.jsonl", "evidence/robobrain-vllm-server-full-001.log",
    "evidence/robobrain-reference-gpu-001.json", "evidence/robobrain-runtime-source-001.json",
    "evidence/robobrain-runtime-source-002.json", "evidence/robobrain-vllm-state-math-002.json",
    "evidence/robobrain-vllm-stopped-state-math-002.json", "evidence/robobrain-vllm-smoke-math-002.jsonl",
    "evidence/robobrain-protocol-smoke-math-002.jsonl", "evidence/robobrain-vllm-server-full-math-002.log",
    "evidence/robobrain-math-compat-numerical-001.json", "evidence/robobrain-vllm-profile-state-003.json",
    "evidence/robobrain-vllm-profile-stopped-003.json", "evidence/robobrain-vllm-profile-server-003.log",
    "evidence/robobrain-vllm-state-compat-004.json", "evidence/robobrain-vllm-stopped-state-compat-004.json",
    "evidence/robobrain-vllm-server-full-compat-004.log", "evidence/robobrain-heldout-004.jsonl",
    "evidence/robobrain-heldout-manifest-004.json", "evidence/asr-model-config.json",
    "evidence/asr-preflight-001.json", "evidence/asr-preflight-002.json", "evidence/asr-probe-001.json",
    "evidence/asr-config-source-001.json", "evidence/asr-audio-reviewed-plan-002.json",
    "evidence/asr-audio-validation-002.json", "evidence/asr-vllm-state-001.json",
    "evidence/asr-vllm-stopped-state-001.json", "evidence/asr-vllm-server-full-001.log",
    "evidence/asr-vllm-state-002.json", "evidence/asr-vllm-stopped-state-002.json",
    "evidence/asr-vllm-server-full-002.log", "evidence/asr-vllm-smoke-zh-002.jsonl",
    "evidence/asr-vllm-smoke-en-002.jsonl", "evidence/local-tests-002.log", "evidence/compat-validation-run-001.log",
    "evidence/platform-checkpoint-20261019-002.json", "evidence/session-validation-002.json",
    "evidence/prepare_audio_env-run003.sh", "evidence/remote-idle-002.json",
    "evidence/scnet-final-tests-002.log", "evidence/release-sync-002.json",
    "evidence/development-manifest-recon-vllm-002.json",
    "evidence/development-manifest-robobrain-002.json", "evidence/development-manifest-asr-002.json",
    "evidence/final-delivery-validation-002.json",
    "evidence/rules/robobrain-vision-math-compat-cpu-002.json",
    "evidence/parallel-preparation-20261020.json", "evidence/parallel-cpu-tests-20261020.log",
    "evidence/parallel-entry-checks-20261020.json",
    "evidence/parallel-cpu-tests-20261020-attempt001.log",
    "evidence/rules/validate_parallel_entries_20261020.py",
    "evidence/rules/robobrain-vision-math-compat-cpu.json",
    "evidence/rules/robobrain-vision-math-diagnostic-cpu.json",
    "evidence/submission-readiness-audit-20261020.json",
    "evidence/official-materials/20261019/report-template.md",
    "fixtures/asr/manifest.json", "fixtures/asr/zh_system_tts.wav", "fixtures/asr/en_system_tts.wav",
    "tasks/qwen3-asr-bw-vllm/compat/qwen_asr_compat-0.0.1.dist-info/METADATA",
))
for folder in ("src", "scripts", "tasks", "docs", "tests"):
    allowed.extend(path for path in (root / folder).rglob("*") if path.is_file() and path.suffix in (".py", ".sh", ".ps1", ".json", ".md", ".txt", ".toml") and "__pycache__" not in path.parts)
for path in (root / "fixtures/asr").rglob("*"):
    if path.is_file() and path.suffix in (".json", ".wav"):
        allowed.append(path)
output = root / "dist" / "guoxin-development.zip"
output.parent.mkdir(exist_ok=True)
manifest = []
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(allowed)):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or resolved != path:
            raise ValueError(f"Non-local or symlinked bundle source: {path.name}")
        name = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        manifest.append({"path": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
        archive.writestr(name, payload)
    archive.writestr("bundle-manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"path": str(output), "files": len(manifest), "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
