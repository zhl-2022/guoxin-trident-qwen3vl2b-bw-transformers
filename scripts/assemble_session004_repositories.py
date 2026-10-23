#!/usr/bin/env python3
"""Plan/build six new local source repositories. Never git, ZIP, upload or run GPU.

The report's formal claims remain pending. A build only proves an audited file
snapshot; it does not prove contest acceptance or checkpoint execution.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from urllib.parse import quote
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HELPER_SPEC = importlib.util.spec_from_file_location(
    "session004_markdown_helpers", Path(__file__).with_name("prepare_task_repositories.py"))
helpers = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(helpers)


@dataclass(frozen=True)
class RepoSpec:
    slug: str
    model: str
    track: str
    service: bool

    @property
    def report(self):
        return f"docs/report-004-{self.slug}.md"


REGISTRY = (
    RepoSpec("recon2reason-bw-vllm", "BAAI/Recon2Reason-Reasoning-4B", "T02", True),
    RepoSpec("robobrain-bw-vllm", "BAAI/RoboBrain2.5-8B-NV", "T02", True),
    RepoSpec("qwen3-asr-bw-vllm", "Qwen/Qwen3-ASR-1.7B", "T04", True),
    RepoSpec("mimo-v2.6-distill-qwen9b-bw-vllm", "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", "T01", True),
    RepoSpec("trident-qwen3vl2b-bw-transformers", "openbmb/Trident-Qwen3VL-2B", "T02", True),
    RepoSpec("qwen3-forcedaligner-bw-transformers", "Qwen/Qwen3-ForcedAligner-0.6B", "T04", False),
)
SPECS = {item.slug: item for item in REGISTRY}
TEMPLATE = "国芯国模黑客松-技术报告模板.md"
TEMPLATE_SHA256 = "b151143f6fa38c87e301cfc9335a94d17fcd49c7113a53fe05f89c04b45f349f"
LOCKED_ZIPS = {
    "tasks/trident-qwen3vl2b-bw-transformers/upstream-source.zip":
        ("1cd6a1695c4bfc1c92616738b9785b868b94a0f82cb74d34c07984d91207289a", "7fbf4b2033e1b3a43a99185fc125d765a549c52d"),
    "tasks/qwen3-forcedaligner-bw-transformers/upstream-source.zip":
        ("0c6c04e035c7f4d60c2cfed518569c709bb786281971a34627aefebec9980fa0", "7c6daf77a2421100f5fb066495372c00129d39ff"),
}
LOCKED_WHEELS = {
    "tasks/qwen3-forcedaligner-bw-transformers/validation-deps/packaging-25.0-py3-none-any.whl":
        "29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484",
}
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_REPO_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_EXPANDED_BYTES = 10 * 1024 * 1024
EXCLUDED_DIRS = helpers.EXCLUDED_DIRS | {"audio-venv", "venv-cpu", "cpu-wheels", "logs", ".cache", "cache"}
EXCLUDED_TESTS = {"test_task_repositories.py", "test_update_task_repositories.py", "test_submission_entries.py"}
TEXT_SUFFIXES = helpers.TEXT_SUFFIXES | {".lock", ".stdout", ".stderr"}
ASSET_SUFFIXES = TEXT_SUFFIXES | {".png", ".jpg", ".jpeg", ".svg", ".wav", ".pdf", ".gif", ".webp"}
TEXT_NAMES = helpers.TEXT_NAMES | {"COPYRIGHT-SCOPE.txt", "entry_points.txt"}
SEED_EVIDENCE = (
    "evidence/vllm-state-001.json", "evidence/asr-audio-reviewed-plan-002.json",
    "evidence/robobrain-runtime-source-002.json",
    "evidence/session004/sglang-backend-004g-20261023/upstream-bench_serving.py",
    "evidence/session004/sglang-backend-004g-20261023/upstream-image.py",
)
MANIFEST = "repository-manifest004.json"
STANDARD_SHELLS = ("env.sh", "run.sh", "stop.sh", "env_check.sh", "smoke.sh", "verify.sh", "bench.sh")
TRIDENT = "trident-qwen3vl2b-bw-transformers"
ASR_PREPARE_FILES = {"validation.stdout", "validation.stderr", "ready.json", "pip-install.log",
                     "native-before.json", "missing-wheels.lock.txt"}


def archived_asr_prepare(name, *, directory=False):
    """Only the exporter's selected ASR preparation metadata, never a venv."""
    match = re.fullmatch(
        r"evidence/session004/[^/]+/project/tasks/qwen3-asr-bw-vllm/runtime/audio-venv"
        r"(?:/asr-prepare-evidence(?:/([^/]+))?)?", name)
    return bool(match and (directory and match.group(1) is None
                          or not directory and match.group(1) in ASR_PREPARE_FILES))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def reject_links(path):
    """lstat every existing ancestor; never follow a reparse point silently."""
    path = Path(path).absolute()
    for part in reversed((path, *path.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/reparse path refused: " + str(part))


def safe_path(root, name):
    name = helpers.relative_name(name)
    candidate = root.joinpath(*PurePosixPath(name).parts)
    reject_links(candidate)
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("Source escaped workspace: " + name)
    return candidate


def forbidden(name, archived=False, source_archive=False):
    parts = PurePosixPath(name).parts
    leaf = parts[-1].lower()
    # Archived runtime evidence is allowed only through explicit evidence inputs;
    # live task/runtime and credential/weight names are always rejected.
    excluded = EXCLUDED_DIRS - ({"runtime", "logs"} if archived and parts[0] == "evidence" else set())
    if archived and archived_asr_prepare(name):
        excluded = excluded - {"audio-venv"}
    if source_archive:
        excluded = excluded - {"models"}  # Locked upstream Python package src/mmemb/models.
    return (any(p.lower() in excluded for p in parts)
            or leaf == ".env" or leaf.startswith(".env.")
            or leaf in helpers.SECRET_NAMES or PurePosixPath(name).suffix.lower() in helpers.SECRET_SUFFIXES
            or leaf in {"cookies.json", "cookie.txt", "auth.json", "authstate.json"})


def scan(name, raw, archived=False, source_archive=False):
    helpers.relative_name(name)
    if forbidden(name, archived, source_archive):
        raise ValueError("Runtime/credential/weight source refused: " + name)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("Individual source exceeds50MiB: " + name)
    suffix = PurePosixPath(name).suffix.lower()
    allowed_suffixes = ASSET_SUFFIXES | ({".dict"} if source_archive else set())
    if suffix not in allowed_suffixes and PurePosixPath(name).name not in TEXT_NAMES:
        if name not in LOCKED_ZIPS and name not in LOCKED_WHEELS:
            raise ValueError("Unapproved source type: " + name)
    if suffix in TEXT_SUFFIXES or (source_archive and suffix == ".dict") or PurePosixPath(name).name in TEXT_NAMES:
        captured_log = suffix in {".log", ".stdout", ".stderr"}
        if captured_log:
            # Raw terminal captures may use the host locale or UTF-16. Keep
            # their bytes, but inspect a decoded representation for secrets.
            encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
            inspection = raw.decode(encoding, errors="replace").encode("utf-8")
        else:
            raw.decode("utf-8", errors="strict")
            inspection = raw
        if (not captured_log and b"\0" in raw) or re.search(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----", inspection) or helpers.CREDENTIAL_LITERAL.search(inspection):
            raise ValueError("Possible credential/private key or NUL: " + name)
    if name == TEMPLATE and digest(raw) != TEMPLATE_SHA256:
        raise ValueError("Fixed Chinese template SHA differs")
    if name in LOCKED_WHEELS and digest(raw) != LOCKED_WHEELS[name]:
        raise ValueError("Fixed validation wheel SHA differs")


def validate_upstream_archive(name, raw, lock_raw):
    expected_hash, revision = LOCKED_ZIPS[name]
    lock = json.loads(lock_raw)
    if digest(raw) != expected_hash or lock.get("archive_sha256") != expected_hash or lock.get("revision") != revision:
        raise ValueError("Fixed upstream ZIP SHA/revision differs: " + name)
    rows = lock.get("files", [])
    expected = {helpers.relative_name(row["path"]): row for row in rows}
    if not rows or len(rows) != len(expected):
        raise ValueError("Missing/duplicate upstream lock member")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("Upstream ZIP members differ from fixed lock")
        if sum(item.file_size for item in infos) > MAX_ARCHIVE_EXPANDED_BYTES:
            raise ValueError("Upstream ZIP expanded size exceeds10MiB")
        for item in infos:
            member = helpers.relative_name(item.filename)
            mode = item.external_attr >> 16
            if item.is_dir() or stat.S_ISLNK(mode) or forbidden(member, source_archive=True) or item.flag_bits & 1:
                raise ValueError("Unsafe upstream ZIP member: " + member)
            row = expected[member]
            if item.file_size != row["size"]:
                raise ValueError("Upstream member size differs: " + member)
            content = archive.read(item)
            # The member's own type/credential policy is checked as source too.
            scan(member, content, source_archive=True)
            if digest(content) != row["sha256"]:
                raise ValueError("Upstream member SHA differs: " + member)
        if archive.testzip() is not None:
            raise ValueError("Upstream ZIP CRC failure")


class Collector:
    def __init__(self, root):
        reject_links(root)
        self.root = Path(root).resolve()
        self.payloads = {}
        self.archived_names = set()
        self.skipped = []

    def add(self, name, archived=False):
        name = helpers.relative_name(name)
        if name in self.payloads:
            return
        path = safe_path(self.root, name)
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Required source is not a regular file: " + name)
        if before.st_size > MAX_FILE_BYTES:
            raise ValueError("Individual source exceeds50MiB: " + name)
        with path.open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or len(raw) != after.st_size:
            raise ValueError("Source changed during read: " + name)
        scan(name, raw, archived)
        self.payloads[name] = raw
        if archived:
            self.archived_names.add(name)
        if sum(map(len, self.payloads.values())) > MAX_REPO_BYTES:
            raise ValueError("Collected source exceeds100MiB")

    def tree(self, folder, archived=False, excluded_tests=False):
        folder = helpers.relative_name(folder)
        base = safe_path(self.root, folder)
        if not base.is_dir():
            raise ValueError("Required directory missing: " + folder)

        def raise_scan_error(error):
            raise error

        for directory, dirs, files in os.walk(base, followlinks=False, onerror=raise_scan_error):
            for leaf in list(dirs):
                candidate = Path(directory) / leaf
                reject_links(candidate)
                allowed_asr_metadata = archived and archived_asr_prepare(
                    candidate.relative_to(self.root).as_posix(), directory=True)
                if (leaf.lower() in EXCLUDED_DIRS - ({"runtime", "logs"} if archived else set())
                        and not allowed_asr_metadata) or leaf.endswith(".egg-info"):
                    self.skipped.append(candidate.relative_to(self.root).as_posix() + "/")
                    dirs.remove(leaf)
            for leaf in sorted(files):
                if excluded_tests and leaf in EXCLUDED_TESTS:
                    self.skipped.append("tests/" + leaf)
                    continue
                self.add((Path(directory) / leaf).relative_to(self.root).as_posix(), archived)

    def report_dependencies(self, seeds):
        """Follow prose references from reports/docs, not foreign upstream READMEs."""
        pending = list(seeds)
        checked = set()
        while pending:
            name = pending.pop()
            if name in checked:
                continue
            checked.add(name)
            self.add(name)
            for target in helpers.markdown_targets(self.payloads[name].decode("utf-8")):
                local = helpers.local_target(self.root, name, target)
                if local is None:
                    continue
                dependency, _ = local
                if dependency not in self.payloads:
                    top = PurePosixPath(dependency).parts[0]
                    if top not in {"docs", "evidence", "tasks", "fixtures", "scripts"} and dependency != TEMPLATE:
                        raise ValueError("Report references unapproved root file: " + dependency)
                    self.add(dependency)
                if dependency.startswith("docs/") and dependency.endswith(".md"):
                    pending.append(dependency)


def rebase_report(root, origin, raw):
    def replace_target(target):
        result = helpers.local_target(root, origin, target)
        return target if result is None else quote(result[0], safe="/-._~") + result[1]

    def change(line):
        def inline(match):
            target = match.group(1) or match.group(2)
            old = "<" + target + ">" if match.group(1) is not None else target
            new = replace_target(target)
            return match.group(0).replace(old, "<" + new + ">" if match.group(1) is not None else new, 1)
        line = helpers.LINK.sub(inline, line)
        reference = helpers.REFERENCE_LINK.match(line)
        if reference:
            target = reference.group(2) or reference.group(3)
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            changed = replace_target(target)
            line = reference.group(1) + ("<" + changed + ">" if reference.group(2) is not None else changed) + reference.group(4) + ending
        return line
    return helpers.outside_fences(raw.decode("utf-8"), change).encode("utf-8")


def report_check(raw):
    headings = []
    def collect(line):
        if re.match(r"^##\s+", line):
            match = re.match(r"^## ([1-8])\.\s", line)
            headings.append(int(match.group(1)) if match else None)
        return line
    helpers.outside_fences(raw.decode("utf-8"), collect)
    if headings != list(range(1, 9)):
        raise ValueError("Report must have exactly eight ordered numbered sections")


def wrapper(task, entry):
    if entry == "env.sh":
        return ("#!/usr/bin/env bash\n"
                '_TASK_REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
                f'source "$_TASK_REPOSITORY_ROOT/tasks/{task}/env.sh"\n'
                "unset _TASK_REPOSITORY_ROOT\n").encode()
    return ("#!/usr/bin/env bash\nset -euo pipefail\n"
            'REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
            f'exec bash "$REPOSITORY_ROOT/tasks/{task}/{entry}" "$@"\n').encode()


def python_wrapper(task, entry):
    if entry == "verify.sh":
        invocation = "os.execvp('bash', ['bash', str(entry), *sys.argv[1:]])"
    else:
        invocation = "os.execv(sys.executable, [sys.executable, str(entry), *sys.argv[1:]])"
    return ("#!/usr/bin/env python3\n"
            '"""Transparent canonical task entry; does not synthesize evaluation results."""\n'
            "import os\nfrom pathlib import Path\nimport sys\n\n"
            f"entry = Path(__file__).resolve().parent / 'tasks/{task}/{entry}'\n"
            "try:\n    " + invocation + "\nexcept OSError as exc:\n"
            "    print(f'Cannot execute canonical entry: {exc}', file=sys.stderr)\n    raise SystemExit(2)\n").encode()


def common_sources(root, extra_evidence=()):
    sources = Collector(root)
    for name in ("pyproject.toml", ".gitignore", TEMPLATE):
        sources.add(name)
    for folder in ("src", "scripts", "fixtures"):
        sources.tree(folder)
    sources.tree("tests", excluded_tests=True)
    for item in REGISTRY:
        sources.tree("tasks/" + item.slug)
        config = json.loads(sources.payloads[f"tasks/{item.slug}/task.json"])
        model = config.get("model_id") or config.get("model", {}).get("id")
        if model != item.model or config.get("track") != item.track:
            raise ValueError("Task identity/track differs from explicit registry: " + item.slug)
        for name in config.get("evidence_files", []):
            name = helpers.relative_name(name)
            if PurePosixPath(name).parts[0] not in {"evidence", "docs", "tasks", "fixtures"}:
                raise ValueError("Unapproved task evidence path: " + name)
            sources.add(name)
    for name in SEED_EVIDENCE:
        sources.add(name)
    sources.tree("evidence/trident-vllm-probe-20261023")
    for name in extra_evidence:
        name = helpers.relative_name(name)
        if not name.startswith("evidence/session004/"):
            raise ValueError("Explicit new run evidence must be under evidence/session004/")
        path = safe_path(sources.root, name)
        if path.is_dir():
            sources.tree(name, archived=True)
        else:
            sources.add(name, archived=True)
    sources.report_dependencies(item.report for item in REGISTRY)
    for name in LOCKED_ZIPS:
        validate_upstream_archive(name, sources.payloads[name], sources.payloads[str(PurePosixPath(name).with_name("upstream-lock.json"))])
    return sources


def framework_scope(spec, files):
    """Describe entry code separately from report-owned runtime evidence."""
    result = {"framework": "vllm" if spec.service else "transformers_engineering_reference",
              "standard_submission_ready": False}
    if spec.slug == TRIDENT:
        profile_name = f"tasks/{TRIDENT}/service-profile.json"
        profile = json.loads(files[profile_name])
        config = json.loads(files[f"tasks/{TRIDENT}/task.json"])
        observation = config.get("session004", {})
        if (profile.get("framework") != "vllm"
                or profile.get("offline_GPU_pilot_verified") is not True
                or observation.get("framework") != "native_vllm_pooling_conversion"
                or observation.get("development_contract_passed") is not True
                or observation.get("HTTP_service_GPU_verified") is not False
                or profile.get("HTTP_service_GPU_verified") is not False
                or profile.get("official_embedding_evaluator_bridge_implemented") is not False):
            raise ValueError("Trident runtime scope changed; review offline/HTTP/official boundaries explicitly")
        result.update({
            "framework": "vllm",
            "framework_execution_mode_reported_by_task": "native_vllm_pooling_conversion",
            "framework_evidence_scope": "Archived offline LLM.embed GPU pilot; new HTTP entry code has no GPU replay",
            "offline_pooling_GPU_verified_reported_by_task": True,
            "HTTP_service_GPU_verified": False,
            "official_embedding_evaluator_bridge_implemented": False,
            "runtime_evidence_source": {
                "task": f"tasks/{TRIDENT}/task.json",
                "task_sha256": digest(files[f"tasks/{TRIDENT}/task.json"]),
                "service_profile": profile_name, "service_profile_sha256": digest(files[profile_name]),
                "assembler_independently_verified_GPU_execution": False,
            },
        })
    return result


def task_payloads(sources, spec):
    files = dict(sources.payloads)
    raw_report = files[spec.report]
    report_check(raw_report)
    files["README.md"] = rebase_report(sources.root, spec.report, raw_report)
    files[".gitattributes"] = b"# Preserve evidence/source bytes, including LF shell scripts.\n* -text\n"
    config_name = f"tasks/{spec.slug}/task.json"
    config = json.loads(files[config_name])
    if spec.slug in helpers.TASKS:
        files.update(helpers.community_payloads(sources, spec.slug, config))
    else:
        for leaf in ("COPYRIGHT-SCOPE.txt", "README_origin.md"):
            files[leaf] = files[f"tasks/{spec.slug}/{leaf}"]
    if spec.service:
        for entry in STANDARD_SHELLS:
            if f"tasks/{spec.slug}/{entry}" not in files:
                raise ValueError("Canonical standard entry missing: " + spec.slug + "/" + entry)
        # Optional task-specific canonical scripts, including ASR prepare/protocol.
        for name in sorted(files):
            path = PurePosixPath(name)
            if len(path.parts) == 3 and path.parts[:2] == ("tasks", spec.slug) and path.suffix == ".sh":
                files[path.name] = wrapper(spec.slug, path.name)
        files["verify.py"] = python_wrapper(spec.slug, "verify.sh")
    else:
        if f"tasks/{spec.slug}/entry.py" not in files:
            raise ValueError("Canonical engineering entry missing")
        files["entry.py"] = python_wrapper(spec.slug, "entry.py")
    status = {
        "schema_version": 1, "session": "session004", "primary_task": spec.slug,
        "model": spec.model, "track": spec.track, "card": "BW", "precision": "bf16",
        "canonical_task": config_name, "canonical_task_sha256": digest(files[config_name]),
        "report_source": spec.report, "report_source_sha256": digest(raw_report),
        "root_report_sha256": digest(files["README.md"]),
        **framework_scope(spec, files),
        "standard_service_entry_complete": spec.service,
        "declared_framework_is_contest_listed": spec.service,
        "framework_eligibility_confirmed": False,
        "candidate_gpu_execution_claimed_by_assembler": False,
        "formal_ready": False, "official_accuracy_verified": False,
        "official_performance_verified": False, "official_accepted": False,
        "submitted": False, "publication_performed": False, "repository_url": None,
        "local_snapshot_only": True, "local_filesystem_privacy_is_remote_acl": False,
        "shared_task_sources_are_other_submission_targets": False,
        "copyright_policy": "Retain user copyright scope and upstream licenses; no repository-wide license invented",
        "incomplete_load_disclosure_is_automatic_rejection": False,
        "archived_runtime_evidence": sorted(sources.archived_names),
        "notes": "Assembler validates bytes/interfaces only. Actual run records and report conclusions require separate review. No models/GPU/git/ZIP/remote operations.",
    }
    files["ROOT_TASK.json"] = encoded(status)
    for target in helpers.markdown_targets(files["README.md"].decode("utf-8")):
        resolved = helpers.local_target(sources.root, "README.md", target)
        if resolved is not None and resolved[0] not in files:
            raise ValueError("Root report unresolved packaged link: " + resolved[0])
    validate_payloads(spec.slug, files, sources.archived_names)
    return files


def validate_payloads(task, files, archived_names=()):
    if task not in SPECS or MANIFEST in files:
        raise ValueError("Unknown task or reserved manifest supplied")
    if sum(map(len, files.values())) > MAX_REPO_BYTES:
        raise ValueError("Repository exceeds100MiB")
    for name, raw in files.items():
        if not isinstance(raw, bytes):
            raise TypeError("Payload must be bytes: " + name)
        scan(name, raw, name in archived_names)
    for name in LOCKED_ZIPS:
        validate_upstream_archive(name, files[name], files[str(PurePosixPath(name).with_name("upstream-lock.json"))])
    spec = SPECS[task]
    config = json.loads(files[f"tasks/{task}/task.json"])
    if (config.get("model_id") or config.get("model", {}).get("id")) != spec.model or config.get("track") != spec.track:
        raise ValueError("Build task identity differs from registry")
    report_check(files["README.md"])
    status = json.loads(files["ROOT_TASK.json"])
    if (status.get("primary_task") != task or status.get("model") != spec.model
            or status.get("standard_service_entry_complete") is not spec.service
            or any(status.get(key) is not False for key in ("formal_ready", "official_accuracy_verified", "official_performance_verified", "submitted", "publication_performed"))):
        raise ValueError("Build status identity/scope differs")
    if any(status.get(key) != value for key, value in framework_scope(spec, files).items()):
        raise ValueError("Build framework/runtime/official scope differs")
    if spec.service:
        for entry in STANDARD_SHELLS:
            if files.get(entry) != wrapper(task, entry):
                raise ValueError("Root standard wrapper differs: " + entry)
        if files.get("verify.py") != python_wrapper(task, "verify.sh"):
            raise ValueError("Root verifier must transparently forward argv")
    elif files.get("entry.py") != python_wrapper(task, "entry.py") or any(entry in files for entry in STANDARD_SHELLS + ("verify.py",)):
        raise ValueError("Engineering candidate must not invent standard service entries")


def output_check(root, output):
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("Output must be an absolute new directory")
    reject_links(output)
    resolved = output.resolve()
    if resolved == root or root.is_relative_to(resolved):
        raise ValueError("Output cannot contain/replace source workspace")
    if any(part.casefold() == "repositories-stage003" for part in resolved.parts):
        raise ValueError("Historical stage003 may not be used")
    if resolved.is_relative_to(root) and resolved.relative_to(root).parts[0] in {"src", "scripts", "tests", "tasks", "docs", "evidence", "fixtures"}:
        raise ValueError("Output may not replace or create canonical source/evidence paths")
    if output.exists():
        raise ValueError("Output directory already exists; choose a fresh version")
    return output


def plan(root, output, tasks=tuple(SPECS), extra_evidence=()):
    root = Path(root)
    reject_links(root)
    root = root.resolve()
    output_check(root, output)
    tasks = tuple(tasks)
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in SPECS for task in tasks):
        raise ValueError("Select distinct registered task slugs")
    sources = common_sources(root, extra_evidence)
    prepared = {task: task_payloads(sources, SPECS[task]) for task in tasks}
    return sources, prepared


def build(root, output, prepared, archived_names=()):
    """Validate complete plan before creating an exclusive fresh directory tree."""
    root = Path(root).resolve()
    output = output_check(root, output)
    if not prepared:
        raise ValueError("Empty build refused")
    for task, files in prepared.items():
        validate_payloads(task, files, archived_names)
    # No mutation occurs until all six byte/name/identity policies pass.
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for task, files in prepared.items():
        destination = output / task
        destination.mkdir()
        rows = []
        for name, raw in sorted(files.items()):
            path = destination.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            reject_links(path)
            with path.open("xb") as stream:
                stream.write(raw)
            if name.endswith(".sh"):
                path.chmod(path.stat().st_mode | stat.S_IXUSR)
            if path.read_bytes() != raw:
                raise ValueError("Destination readback differs: " + str(path))
            rows.append({"path": name, "bytes": len(raw), "sha256": digest(raw)})
        manifest = encoded({"schema_version": 1, "session": "session004", "primary_task": task,
                            "files": rows, "payload_bytes": sum(row["bytes"] for row in rows),
                            "manifest_self_hash_excluded": True, "source_bytes_normalized": False,
                            "formal_ready": False, "submitted": False, "publication_performed": False})
        with (destination / MANIFEST).open("xb") as stream:
            stream.write(manifest)
        if (destination / MANIFEST).read_bytes() != manifest:
            raise ValueError("Manifest readback differs: " + str(destination))
        results.append({"task": task, "directory": str(destination), "payload_files": len(rows),
                        "manifest_sha256": digest(manifest), "all_payload_readback_verified": True})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/repositories-stage004")
    parser.add_argument("--task", action="append", choices=tuple(SPECS))
    parser.add_argument("--extra-evidence", action="append", default=[], help="Explicit archived originals under evidence/session004/")
    parser.add_argument("--build", action="store_true", help="Create new local source trees; no ZIP/git/remote action")
    args = parser.parse_args(argv)
    sources, prepared = plan(args.root, args.output, args.task or tuple(SPECS), args.extra_evidence)
    result = {"session": "session004", "mode": "build_local_source_only" if args.build else "dry_run_no_writes",
              "output": str(args.output), "source_files": len(sources.payloads),
              "source_bytes": sum(map(len, sources.payloads.values())),
              "excluded_test_modules": sorted(EXCLUDED_TESTS), "skipped_source_paths": sorted(sources.skipped),
              "archived_runtime_evidence_explicit": sorted(sources.archived_names),
              "tasks": [{"task": task, "files": len(files), "bytes": sum(map(len, files.values())),
                         "standard_service_entry_complete": SPECS[task].service,
                         "formal_ready": False, "submitted": False} for task, files in prepared.items()],
              "source_files_modified": False, "git_or_remote_or_gpu_used": False}
    if args.build:
        result["built"] = build(sources.root, args.output, prepared, sources.archived_names)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
