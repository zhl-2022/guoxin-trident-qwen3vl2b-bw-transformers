# Archived SGLang native result reanalysis

Native output_lens can come from stream usage or requested-length fallback; raw SSE usage, terminal framing, and exact 1024 generated tokens are not independently certified by this JSONL. Upstream combines reasoning_content/content.

Only request_rate Infinity is normalized. Original producer statuses and every native file SHA remain in analysis.json.

| Load | Producer status | Exit/process | Reanalysis | Completed | Output token/s | Wall seconds |
|---|---|---|---|---:|---:|---:|
| round1_4096_1024_c8 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 16 | 147.26358778469879 | 156.231193 |
| round2_4096_1024_c8 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 16 | 148.31421966721786 | 160.955256 |
| round1_4096_1024_c64 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 128 | 269.37477986308517 | 619.059157 |
| round2_4096_1024_c64 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 128 | 269.34815636083914 | 608.173795 |
| round1_16384_1024_c8 | invalid_native_result | 0/exited | complete_native_client_report_reanalyzed | 16 | 77.61654714889822 | 268.074816 |
| round2_16384_1024_c8 | invalid_native_result | 0/exited | not_eligible | 16 | 78.05437192898316 | 264.964626 |
| round1_16384_1024_c64 | invalid_native_result | 0/exited | not_eligible | 128 | 91.46728496375748 | 1622.878336 |
| round2_16384_1024_c64 | invalid_native_result | 0/exited | not_eligible | 128 | 91.40938298233905 | 1600.62053 |

Eligible native client reports: 5/8. Official acceptance: false. Exact output token proof: false.
