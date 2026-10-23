#!/usr/bin/env python3
"""Capture PyPI metadata for approved CPU wheels; does not download wheels."""
from __future__ import annotations
import concurrent.futures
import json
from pathlib import Path
import urllib.request

from packaging.tags import compatible_tags, cpython_tags, parse_tag

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "transformers": ["4.57.6", "4.57.1"], "huggingface-hub": ["0.36.0"], "tokenizers": ["0.22.1"],
    "accelerate": ["1.12.0"], "numpy": ["1.26.4"], "nagisa": ["0.2.11"], "DyNET38": ["2.2"],
    "soynlp": ["0.0.493"], "six": ["1.17.0"], "Cython": ["3.0.12"], "librosa": ["0.11.0"],
    "soundfile": ["0.13.1"], "scipy": ["1.15.3"], "scikit-learn": ["1.6.1"], "psutil": ["6.1.1"],
    "qwen-vl-utils": ["0.0.14"], "av": ["14.2.0"], "filelock": ["3.18.0"], "fsspec": ["2025.3.2"],
    "packaging": ["25.0"], "PyYAML": ["6.0.2"], "regex": ["2024.11.6"], "requests": ["2.32.3"],
    "safetensors": ["0.5.3"], "tqdm": ["4.67.1"], "typing-extensions": ["4.13.2"], "hf-xet": ["1.1.10"],
    "Pillow": ["11.2.1"], "joblib": ["1.4.2"], "threadpoolctl": ["3.6.0"], "audioread": ["3.0.1"],
    "decorator": ["5.2.1"], "numba": ["0.61.2"], "llvmlite": ["0.44.0"], "pooch": ["1.8.2"],
    "soxr": ["0.5.0.post1"], "lazy-loader": ["0.4"], "msgpack": ["1.1.0"], "platformdirs": ["4.3.7"],
    "cffi": ["1.17.1"], "pycparser": ["2.22"], "certifi": ["2025.1.31"], "idna": ["3.10"],
    "charset-normalizer": ["3.4.1"], "urllib3": ["2.3.0"]}


def capture(pair):
    name, version = pair
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    data = json.load(urllib.request.urlopen(url, timeout=25))
    platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)] + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    tags = list(cpython_tags((3, 10), ["cp310"], platforms)) + list(compatible_tags((3, 10), "cp310", platforms))
    rank = {tag: index for index, tag in enumerate(tags)}
    candidates = []
    for item in data["urls"]:
        if not item["filename"].endswith(".whl") or item.get("yanked"):
            continue
        wheel_tags = parse_tag("-".join(item["filename"][:-4].split("-")[-3:]))
        matches = [rank[tag] for tag in wheel_tags if tag in rank]
        if matches:
            candidates.append((min(matches), item))
    if not candidates:
        raise ValueError(f"No approved Linux x86_64 cp310 wheel for {name}=={version}")
    item = min(candidates, key=lambda x: x[0])[1]
    requirements = [x for x in data["info"].get("requires_dist", []) or [] if "extra ==" not in x and "extra==" not in x]
    return {"name": data["info"]["name"], "version": version, "filename": item["filename"],
            "url": item["url"], "sha256": item["digests"]["sha256"], "size": item["size"],
            "requires_dist": requirements, "source_metadata_url": url}


def main():
    jobs = [(name, version) for name, versions in PINS.items() for version in versions]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        pool = list(executor.map(capture, jobs))
    output = {"schema_version": 1, "target": "Linux x86_64 / CPython3.10 / glibc>=2.28",
              "wheel_payload_bytes_downloaded": 0, "native_gpu_libraries_in_pool": False, "wheels": pool}
    (ROOT / "scripts/candidate_cpu_wheels_20261023.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Captured {len(pool)} CPU wheel metadata records; no wheel payloads downloaded")


if __name__ == "__main__":
    main()
