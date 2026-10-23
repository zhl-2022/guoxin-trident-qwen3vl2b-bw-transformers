#!/usr/bin/env bash
# Add CPU WAV dependencies in a new venv; never change the native vendor stack.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != "--apply" ) ]]; then
  echo "Usage: bash prepare_audio_env.sh NEW_ABSOLUTE_VENV_DIR [--apply]" >&2
  exit 2
fi
exec "${SCNET_BASE_PYTHON:-/usr/bin/python3}" -B - "$1" "${2:-}" <<'PY'
import hashlib
import importlib.metadata as md
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.parse
import venv
import wave

sys.dont_write_bytecode = True
target = Path(sys.argv[1])
apply = sys.argv[2] == "--apply"
if not target.is_absolute() or target.exists():
    raise SystemExit("Require a new absolute venv path; existing paths are refused")
if sys.version_info[:2] != (3, 10) or platform.system() != "Linux":
    raise SystemExit("This recipe targets the observed Linux Python 3.10 native image")

def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()

def installed():
    result = {}
    for d in md.distributions():
        if d.metadata.get("Name"):
            result.setdefault(canonical(d.metadata["Name"]), {
                "version": d.version, "root": str(Path(d.locate_file("")).resolve())
            })
    return result

# Pins are only constraints for packages absent from the base environment.
# Present packages are preserved; any resolver attempt to shadow one is refused.
pins = {
    "librosa": "0.11.0", "soundfile": "0.13.1", "audioread": "3.0.1",
    "numba": "0.61.2", "llvmlite": "0.44.0", "scikit-learn": "1.6.1",
    "joblib": "1.4.2", "decorator": "5.2.1", "pooch": "1.8.2",
    "soxr": "0.5.0.post1", "typing-extensions": "4.13.2",
    "lazy-loader": "0.4", "msgpack": "1.1.0", "cffi": "1.17.1",
    "pycparser": "2.22", "threadpoolctl": "3.6.0", "packaging": "24.2",
    "platformdirs": "4.3.7",
}
protected = (
    "torch", "torchvision", "transformers", "vllm", "vllm-hcu", "numpy",
    "scipy", "triton", "flash-attn", "lightop", "aiter", "python-multipart",
)
before = installed()
for name in protected:
    if name not in before:
        raise SystemExit("Native required distribution missing: " + name)
if before["numpy"]["version"] != "1.25.0" or before["scipy"]["version"] != "1.15.3":
    raise SystemExit("Recipe requires the observed NumPy 1.25.0 / SciPy 1.15.3; re-review this image")
if "+das" not in before["torch"]["version"]:
    raise SystemExit("Expected the native DAS Torch distribution")

# Reuse the base image's pip through system-site-packages. Ubuntu may omit
# ensurepip; running that pip via the venv interpreter still targets sys.prefix.
venv.EnvBuilder(system_site_packages=True, with_pip=False).create(target)
python = str(target / "bin" / "python")
record = target / "audio-env-evidence"
record.mkdir()

def save(name, obj):
    with (record / name).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")

save("native-before.json", {
    "python": sys.version, "executable": sys.executable,
    "protected": {name: before[name] for name in protected},
    "installed_versions": {name: item["version"] for name, item in before.items()},
    "script_note": "CPU audio dependencies only; no model load or GPU allocation",
})
constraints = record / "constraints.txt"
constraints.write_text("".join(
    f"{name}=={version}\n" for name, version in pins.items() if name not in before
) + "".join(f"{name}=={before[name]['version']}\n" for name in protected), encoding="utf-8")
env = dict(os.environ)
for key in list(env):
    if key.startswith("PIP_"):
        del env[key]
env["PIP_CONFIG_FILE"] = os.devnull
env["PYTHONNOUSERSITE"] = "1"
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["NUMBA_CACHE_DIR"] = str(target / "numba-cache")

def run(command, timeout=180, echo=True):
    started = time.monotonic()
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    if echo or result.returncode:
        print(result.stdout[-16000:], end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); venv retained at {target}")
    return result, time.monotonic() - started

pip = [python, "-B", "-m", "pip", "--disable-pip-version-check", "--no-input"]
help_result, _ = run(pip + ["install", "--help"], timeout=30, echo=False)
if "--dry-run" not in help_result.stdout or "--report" not in help_result.stdout:
    raise SystemExit("Base pip needs --dry-run and --report support; no pip upgrade attempted")
plan_path = record / "pip-plan.json"
run(pip + ["install", "--dry-run", "--report", str(plan_path),
           "--index-url", "https://pypi.org/simple", "--only-binary=:all:",
           "--no-cache-dir", "-c", str(constraints),
           "librosa==0.11.0", "soundfile==0.13.1"])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
locked = []
selected = []
for item in plan.get("install", []):
    meta = item["metadata"]
    name, version = canonical(meta["name"]), meta["version"]
    if name in before:
        raise SystemExit(f"Refusing to shadow existing package {name}=={before[name]['version']}")
    if name not in pins or version != pins[name]:
        raise SystemExit(f"Unreviewed dependency/version in plan: {name}=={version}")
    download = item["download_info"]
    url = download["url"]
    parsed = urllib.parse.urlparse(url)
    digest = download.get("archive_info", {}).get("hashes", {}).get("sha256")
    if (parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org"
            or not parsed.path.endswith(".whl") or not digest
            or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise SystemExit("Require a PyPI-hosted wheel and its SHA256")
    locked.append(f"{name} @ {url} --hash=sha256:{digest}\n")
    selected.append({"name": name, "version": version, "url": url, "sha256": digest})
lock = record / "audio-wheels.lock.txt"
lock.write_text("".join(locked), encoding="utf-8")
save("reviewed-plan.json", {"apply_requested": apply, "selected": selected,
                           "protected": {name: before[name] for name in protected}})
print("Reviewed missing CPU packages: " + json.dumps(selected, ensure_ascii=False), flush=True)
if not apply:
    print(f"Plan only; nothing installed. Review {record}; use another new directory with --apply.")
    raise SystemExit(0)
if locked:
    run(pip + ["install", "--no-deps", "--require-hashes", "--only-binary=:all:",
               "--no-cache-dir", "-r", str(lock)])

check = r'''
import importlib.metadata as md, io, json, sys, wave
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import numba
import soxr
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
before = json.loads(Path(sys.argv[1]).read_text())
protected = before["protected"]
for name, expected in protected.items():
    d = md.distribution(name)
    actual = {"version": d.version, "root": str(Path(d.locate_file("")).resolve())}
    if actual != expected:
        raise RuntimeError("Protected native distribution changed: " + name)
# Validate the installed audio dependency closure rather than unrelated base-image conflicts.
seen = set()
def check_deps(name):
    name = canonicalize_name(name)
    if name in seen: return
    seen.add(name)
    d = md.distribution(name)
    for raw in d.requires or ():
        r = Requirement(raw)
        if r.marker and not r.marker.evaluate({"extra": ""}): continue
        child = md.distribution(r.name)
        if r.specifier and not r.specifier.contains(child.version, prereleases=True):
            raise RuntimeError(f"Unsatisfied audio requirement {raw}: {child.version}")
        check_deps(r.name)
check_deps("librosa")
check_deps("soundfile")
buf = io.BytesIO()
with wave.open(buf, "wb") as wav:
    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
    wav.writeframes(b"\0\0" * 1600)
waveform, sr = librosa.load(io.BytesIO(buf.getvalue()), sr=16000)
assert sr == 16000 and waveform.shape == (1600,) and waveform.dtype == np.float32
resampled = librosa.resample(waveform, orig_sr=16000, target_sr=8000)
assert resampled.shape == (800,) and bool(np.isfinite(resampled).all())
print(json.dumps({"status": "passed", "python": sys.executable,
    "protected_native_unchanged": True, "audio_closure_packages": sorted(seen),
    "librosa": librosa.__version__, "soundfile": sf.__version__,
    "numba": numba.__version__, "soxr": soxr.__version__,
    "wav": {"sample_rate": sr, "samples": len(waveform), "dtype": str(waveform.dtype)},
    "resample_samples": len(resampled), "gpu_allocated": False,
    "model_loaded": False, "official_accuracy_evidence": False}, indent=2))
'''
result, elapsed = run([python, "-B", "-c", check, str(record / "native-before.json")], timeout=180)
save("audio-validation.json", {"elapsed_seconds": elapsed, "result": json.loads(result.stdout)})
print(f"Audio venv ready: {python}; evidence: {record}")
print("Launch vLLM with this absolute Python path; the shell's native Python remains unchanged.")
PY
