#!/usr/bin/env python3
"""Fetch small, public provenance files only; never fetch checkpoint weights."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import urllib.request
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen3-forcedaligner-bw-transformers": ("Qwen/Qwen3-ForcedAligner-0.6B", "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"),
    "mimo-v2.6-distill-qwen9b-bw-vllm": ("XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", "2367e865d009c13ac81713a2878291d33ab28177"),
    "trident-qwen3vl2b-bw-transformers": ("openbmb/Trident-Qwen3VL-2B", "a5aedf8733f2107ab75b42214c78496d85759077"),
}
REPOS = {
    "qwen3-forcedaligner-bw-transformers": ("QwenLM/Qwen3-ASR", "7c6daf77a2421100f5fb066495372c00129d39ff", ("qwen_asr/", "examples/example_qwen3_forced_aligner.py", "LICENSE", "pyproject.toml", "setup.py", "README.md")),
    "trident-qwen3vl2b-bw-transformers": ("OpenBMB/Trident", "7fbf4b2033e1b3a43a99185fc125d765a549c52d", ("src/", "scripts/", "configs/", "data/download_data.sh", "LICENSE", "README.md", "requirements.txt")),
}


def fetch(url: str, limit: int = 20_000_000) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "guoxin-candidate-provenance/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                data = response.read(limit + 1)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)
    if len(data) > limit:
        raise ValueError("Provenance response exceeded bounded small-file limit")
    return data


def save_json(path: Path, value: object) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) == value:
            return
        raise ValueError("Existing provenance differs; refusing overwrite: " + str(path))
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    all_sources = []
    for slug, (model, revision) in MODELS.items():
        target = ROOT / "tasks" / slug
        target.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true"
        info = json.loads(fetch(url))
        if info["sha"] != revision:
            raise ValueError("Upstream model revision changed")
        selected = []
        for item in info["siblings"]:
            name = item["rfilename"]
            if "/" in name or name in {"README.md", ".gitattributes", "configuration.json"}:
                continue
            if not (name.endswith((".json", ".jinja", ".txt", ".safetensors"))):
                continue
            row = {"path": name, "size": item["size"]}
            if "lfs" in item:
                row["sha256"] = item["lfs"]["sha256"]
            else:
                row["git_blob_sha1"] = item["blobId"]
            selected.append(row)
        lock = {"schema_version": 1, "model_id": model, "revision": revision,
                "source_url": url, "files": selected,
                "weight_bytes": sum(x["size"] for x in selected if x["path"].endswith(".safetensors")),
                "downloaded_weight_bytes": 0}
        save_json(target / "model-lock.json", lock)
        all_sources.append({"task": slug, "model_api": url, "model_revision": revision,
                            "license": info.get("cardData", {}).get("license"), "weight_bytes": lock["weight_bytes"]})
        # Capture publisher documentation without tokenizer/weights downloads.
        readme = fetch(f"https://huggingface.co/{model}/resolve/{revision}/README.md", 200_000)
        (target / "README_origin.md").write_bytes(readme)
        if slug not in REPOS:
            continue
        repo, commit, includes = REPOS[slug]
        archive_url = f"https://codeload.github.com/{repo}/zip/{commit}"
        archive = fetch(archive_url)
        entries = []
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(archive)) as source, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as dest:
            for item in source.infolist():
                if item.is_dir():
                    continue
                parts = item.filename.split("/", 1)
                if len(parts) != 2:
                    continue
                relative = parts[1]
                if ".." in Path(relative).parts or relative.startswith("/"):
                    raise ValueError("Unsafe upstream archive path")
                if not any(relative == prefix or (prefix.endswith("/") and relative.startswith(prefix)) for prefix in includes):
                    continue
                payload = source.read(item)
                entries.append({"path": relative, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
                info_zip = zipfile.ZipInfo(relative, (2026, 10, 23, 0, 0, 0))
                info_zip.compress_type = zipfile.ZIP_DEFLATED
                dest.writestr(info_zip, payload)
        selected_archive = output.getvalue()
        (target / "upstream-source.zip").write_bytes(selected_archive)
        save_json(target / "upstream-lock.json", {"schema_version": 1, "repository": repo,
            "revision": commit, "source_url": archive_url,
            "original_archive_sha256": hashlib.sha256(archive).hexdigest(),
            "archive_file": "upstream-source.zip", "archive_sha256": hashlib.sha256(selected_archive).hexdigest(),
            "selection": list(includes), "files": sorted(entries, key=lambda x: x["path"]),
            "modified_upstream_files": False})
        all_sources[-1].update({"source_repo": f"https://github.com/{repo}/tree/{commit}", "source_revision": commit,
                                "selected_source_bytes": len(selected_archive), "selected_source_files": len(entries)})
    save_json(ROOT / "evidence/task-expansion-sources-20261023.json", {
        "schema_version": 1, "checked_date_client": "2026-10-23", "sources": all_sources,
        "formal_evaluation_materials_complete": False, "gpu_verified": False,
        "sources_are_publisher_materials_not_contest_acceptance": True,
        "total_weight_bytes": sum(x["weight_bytes"] for x in all_sources),
        "weights_downloaded": False})
    print(json.dumps(all_sources, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
