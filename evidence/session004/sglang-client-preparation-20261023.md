# Session004 SGLang client preparation

This is an independent SGLang 0.5.12 performance-client wrapper. It does not replace the native vLLM service, modify upstream benchmark source, certify official accuracy, or change archived session003 records.

Use helper package **004b** or later. Its driver preserves the absolute venv interpreter path without resolving its symlink to the base Python executable. The original 004 package remains a historical CPU snapshot and must not be used for deployment.

## Prepare the client

The bootstrap targets the recorded Linux x86_64, Python 3.10, glibc >=2.34 image. It creates a **new** venv with system site packages and installs only the pinned official 7.8 MB SGLang wheel with `--ignore-installed --no-deps --require-hashes`. It does not resolve Torch/CUDA dependencies. Existing or partial venvs are refused and preserved.

```bash
python3 -B scripts/prepare_sglang_client.py \
  --venv /root/private_data/sglang-client-004

python3 -B scripts/prepare_sglang_client.py \
  --venv /root/private_data/sglang-client-004 \
  --apply
```

The actual `--help` import is a compatibility probe, not a guaranteed result. Its original log and the before/after identity of twelve protected native distributions are saved under the new venv's `client-bootstrap-evidence/`. A failed import does not justify replacing native Torch, Transformers, NumPy, or accelerator packages.

If a missing CPU package is identified, review its exact official PyPI wheel and provide a JSON array using `--extra-cpu-lock`. Every entry must have `name`, `version`, `url`, and `sha256`; protected packages and packages outside the CPU allowlist are rejected. Use a new venv path after a failed partial attempt. Do not run an automatic full dependency resolver.

## Run the upstream client

For the upstream 0.5.12 image dataset, the processor loads `served_model_name`, rather than the separate tokenizer path. Therefore configure the new server's served name as its **same local model path**, and pass that path to the client. The wrapper refuses an unresolved short alias to avoid an unintended model download. HF network downloads and client accelerator visibility are disabled for workload subprocesses.

```bash
python3 -B scripts/t02_sglang_bench.py \
  --client-python /root/private_data/sglang-client-004/bin/python \
  --model-path "$MODEL_PATH" \
  --served-model-name "$MODEL_PATH" \
  --runtime-dir "$TASK_DIR/runtime" \
  --expected-run-id "$RUN_ID" \
  --output-dir evidence/session004/recon-sglang-001 \
  --timeout-seconds 1800 \
  --total-timeout-seconds 6400
```

The above only prints a plan. Add `--run` after checking the owned server identity and the platform's remaining budget. `--load 4096:1` selects a smaller independent workload; `--round 2` explicitly selects only that round, without relabeling another run as warmup. Use a different output directory for every execution. The supervisor and platform must be stopped separately; this client never owns or stops a server.

## Workload and evidence boundaries

- Actual upstream module: `sglang.bench_serving`; version fixed to 0.5.12. Backend `vllm-chat`, image dataset, one random JPEG of 1024×1024, fixed length ratio 1, seed 1, temperature 0, ignore EOS, streaming usage requested, detailed original results saved.
- Six loads × two rounds plan 608 measured requests. Upstream default warmup is explicitly one additional request per command, hence 12 extra warmups for a full matrix. `--warmup-requests 0` is an explicit variant and is recorded.
- Text targets 4096/16384 exclude separately accounted image/chat tokens and possible detokenization differences. Image policy still requires the task-specific materials.
- Upstream `output_lens` may use stream usage or fall back to the requested output length. Matching 1024 is preserved as a **client report**, not independent proof of exact generated token count. `exact_output_tokens_independently_verified` stays false.
- Native JSONL and logs remain unchanged under `native/` and per-row logs. Wrapper `round1.jsonl`/`round2.jsonl` contain execution/provenance records; they are not substituted for native original files. `summary.md` displays observed client metrics.
- The archive includes wrapper source, actual installed upstream client source/SHA, client/native package metadata, bound server state/config/logs before and after, commands, deadlines, failures, and a SHA/size manifest. Exceeding a total deadline leaves later rows unattempted and all existing raw files intact.
- `complete_client_reports` means the selected upstream client reports passed structural checks. It does not mean official accuracy, official acceptance, or source equivalence to earlier vLLM random-mm development results.

Primary sources: [pinned client](https://github.com/sgl-project/sglang/blob/v0.5.12/python/sglang/bench_serving.py), [pinned image dataset](https://github.com/sgl-project/sglang/blob/v0.5.12/python/sglang/benchmark/datasets/image.py), [official PyPI version metadata](https://pypi.org/pypi/sglang/0.5.12/json).
