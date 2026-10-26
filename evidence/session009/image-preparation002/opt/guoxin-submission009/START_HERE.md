# BW publisher-baseline evaluation image

ASR and Trident checkpoints, Python environments, complete selected datasets,
evaluation tools and immutable results are on the system disk. Other eight models
retain development-only sources under `/opt/guoxin-reviewed008`.

## ASR

Start: `bash /opt/guoxin-asr-tools009-final/run_asr.sh`

Evaluate into a NEW output directory:

`/opt/guoxin-asr009/venv/bin/python /opt/guoxin-asr-tools009-final/evaluate_asr.py --data /opt/guoxin-asr009/data --output /root/asr-review-new`

Benchmark:

`/opt/guoxin-asr009/venv/bin/python /opt/guoxin-asr-tools009-final/benchmark_asr.py --data /opt/guoxin-asr009/data --output /root/asr-bench-new`

Stop with the exact run ID returned by the launcher:

`/opt/guoxin-asr009/venv/bin/python /opt/guoxin-asr009/project/scripts/recon2reason_runtime.py stop --task-dir /opt/guoxin-asr009/project/tasks/qwen3-asr-bw-vllm --run-id RUN_ID`

## Trident

Start the original native wrapper with relocated environment/model variables:

`TRIDENT_MODEL_PATH=/opt/guoxin-trident009/model TRIDENT_PYTHON_BIN=/opt/guoxin-trident009/venv/bin/python TRIDENT_MAX_RUNTIME_SECONDS=5400 bash /opt/guoxin-trident009/project/tasks/trident-qwen3vl2b-bw-transformers/run.sh`

Evaluate all ChartQA Origin requests and two performance rounds:

`/opt/guoxin-trident009/venv/bin/python /opt/guoxin-trident-tools009-run002/evaluate_trident.py --data /opt/guoxin-trident009/data --project /opt/guoxin-trident009/project --output /root/trident-review-new`

Stop with the returned run ID using `trident_embedding_runtime_004.py stop` and
the same task directory. Run the two models serially on one card.

## Evidence and limitations

Results: `/opt/guoxin-submission009/results/`.
Dataset/record identities: `IMAGE-CONTENTS.json`.
The first Trident run is retained with its one filename-loading failure.
ASR scoring uses the disclosed Whisper English normalizer; the Qwen publisher
did not publish its exact normalization implementation. The image and measured
results are not official contest acceptance. No final form was submitted.
