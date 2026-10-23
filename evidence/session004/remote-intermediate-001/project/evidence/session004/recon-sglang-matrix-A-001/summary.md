# SGLang client performance record

Official acceptance: false. upstream output_lens may use stream usage or requested-length fallback; exact 1024-token output is not independently certified
Configured server framework: vllm. Upstream client sender alias: sglang-oai-chat; OpenAI /v1/chat/completions. SGLang0.5.12 image dataset permits sglang-oai-chat, not vllm-chat; both aliases map to the same upstream OpenAI chat sender. The sender alias does not identify the server framework or create a SGLang-serving contest piece.

| Round | Text target | Concurrency | Completed | Output token/s | Mean TTFT ms | Client wall s | Status |
|---|---|---|---|---|---|---|---|
| 1 | 4096 | 8 |  |  |  | 156.231193 | invalid_native_result |
| 2 | 4096 | 8 |  |  |  | 160.955256 | invalid_native_result |
| 1 | 4096 | 64 |  |  |  | 619.059157 | invalid_native_result |
| 2 | 4096 | 64 |  |  |  | 608.173795 | invalid_native_result |
| 1 | 16384 | 8 |  |  |  | 268.074816 | invalid_native_result |
| 2 | 16384 | 8 |  |  |  | 264.964626 | invalid_native_result |
| 1 | 16384 | 64 |  |  |  |  | running |
| 2 | 16384 | 64 |  |  |  |  | planned |
