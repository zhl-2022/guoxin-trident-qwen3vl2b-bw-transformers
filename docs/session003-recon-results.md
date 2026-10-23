# Session003：Recon2Reason 最终开发结果审计

更新日期：2026-10-20（北京时间）。本报告依据最终下载ZIP的原始字节、native逐请求结果、服务状态及平台checkpoint003；本地审计没有调用GPU。当前结论为 **11/12格开发覆盖、480个可核完整请求、491520个输出token**，不是一份完整矩阵，正式验收与赛事提交均未完成。

## 1. 结论与范围

- 新七项图文功能检查通过，生成均以stop结束、未截断；独立的maxseq4/c1及maxseq16/c8校准继续保留。
- 原矩阵完成10/12格、352个完整请求、360448个输出token；16384/c64第一轮超时，第二轮未尝试。
- 独立补跑仅第一轮128请求完整、131072输出token；第二轮1307.366197秒超时，没有native JSON。两份run均为`incomplete_or_failed`。
- 480条已核请求均有一条output_lens=1024、一份非空generated_text和空error；总输出与逐项和、请求数×1024一致。最后128个计划请求的完成数与输出总量为**未知**，不能记为0，也不从报告删去。
- 最终服务state为stopped，进程启动token均不再存活、端口关闭；平台另有实际关机与保存环境证据。

逐格/逐请求独立审计见 [final audit](../evidence/recon-matrix-final-audit-003.json)，精简结果及来源见 [final summary](../evidence/recon-final-results-summary-003.json)。480是按执行格统计的完整请求数，不代表480个独立数据样本；固定seed使不同轮次和负载可能复用输入。

## 2. 原矩阵与独立补跑

[原run](../evidence/session003/recon-matrix/run.json) 从01:41:41运行至02:48:22，单格上限1800秒、总预算4000秒。前10格完整；`round1_16384_1024_c64`在745.996889秒时因剩余总预算到期而timeout，`round2_16384_1024_c64`为`not_attempted_total_timeout`。没有为这两格生成native结果。

[独立retry](../evidence/session003/recon-matrix-retry/run.json) 使用新服务，从03:01:33运行至03:46:31，单格上限1800秒、动态总预算2697秒。第一轮client wall1388.661320秒，第二轮1307.366197秒超时。原run保持原样，独立成功单元只用于标记开发覆盖；没有将两份run捏造成同次完整测试。

| 文本目标输入 | 并发 | 轮次 | 结果来源 | 完整请求/计划 | 输出token和 | 原生duration s | 输出token/s | Client wall s | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| 4096 | 1 | 1 | [original](../evidence/session003/recon-matrix/native/round1_4096_1024_c1.json) | 8/8 | 8192 | 377.155 | 21.721 | 409.042 | 完整性verified |
| 4096 | 1 | 2 | [original](../evidence/session003/recon-matrix/native/round2_4096_1024_c1.json) | 8/8 | 8192 | 375.722 | 21.803 | 408.089 | 完整性verified |
| 4096 | 8 | 1 | [original](../evidence/session003/recon-matrix/native/round1_4096_1024_c8.json) | 16/16 | 16384 | 95.446 | 171.657 | 128.997 | 完整性verified |
| 4096 | 8 | 2 | [original](../evidence/session003/recon-matrix/native/round2_4096_1024_c8.json) | 16/16 | 16384 | 95.332 | 171.862 | 129.275 | 完整性verified |
| 4096 | 64 | 1 | [original](../evidence/session003/recon-matrix/native/round1_4096_1024_c64.json) | 128/128 | 131072 | 450.967 | 290.646 | 489.986 | 完整性verified |
| 4096 | 64 | 2 | [original](../evidence/session003/recon-matrix/native/round2_4096_1024_c64.json) | 128/128 | 131072 | 452.448 | 289.695 | 490.823 | 完整性verified |
| 16384 | 1 | 1 | [original](../evidence/session003/recon-matrix/native/round1_16384_1024_c1.json) | 8/8 | 8192 | 428.704 | 19.109 | 462.214 | 完整性verified |
| 16384 | 1 | 2 | [original](../evidence/session003/recon-matrix/native/round2_16384_1024_c1.json) | 8/8 | 8192 | 377.263 | 21.714 | 411.054 | 完整性verified |
| 16384 | 8 | 1 | [original](../evidence/session003/recon-matrix/native/round1_16384_1024_c8.json) | 16/16 | 16384 | 150.656 | 108.751 | 184.974 | 完整性verified |
| 16384 | 8 | 2 | [original](../evidence/session003/recon-matrix/native/round2_16384_1024_c8.json) | 16/16 | 16384 | 97.036 | 168.845 | 132.324 | 完整性verified |
| 16384 | 64 | 1 | [retry](../evidence/session003/recon-matrix-retry/native/round1_16384_1024_c64.json) | 128/128 | 131072 | 1335.164 | 98.169 | 1388.661 | 完整性verified |
| 16384 | 64 | 2 | retry；原run未尝试 | 未知/128 | 未知 | — | — | 1307.366 | timeout，无native JSON |

以上11份native JSON的completed均等于该格请求数、failed=0；另外的timeout/未尝试仍明确保留。已发生但未形成完整native JSON的请求，不属于已核480请求。完整计划为608请求、622592输出token，当前缺`round2_16384_1024_c64`一格128请求的完整结果。审计工具因此返回exit1，`matrix_complete_verified=false`、`combined_development_coverage_complete=false`。

## 3. 服务、缓存与源码身份

| 服务 | 原始证据 | 实际状态及profile |
|---|---|---|
| maxseq4，run `142884db17fa4053a243bf2c5822ed27` | [state](../evidence/recon-calibration-server-maxseq4-state-003.json)、[log](../evidence/recon-calibration-server-maxseq4-server-003.log) | 01:13:50 ready、01:33:07 stopped；44.95GiB、327296 KV tokens；实际服务上限3600秒 |
| 原maxseq16，run `d70f7e5f590745e28488c55820169551` | [config](../evidence/session003/recon-original-runtime/config-d70f7e5f590745e28488c55820169551.json)、[state](../evidence/session003/recon-original-runtime/state-after-stop.json)、[log](../evidence/session003/recon-original-runtime/server.log) | 01:35:35 ready、02:56:41 stopped；44.93GiB、327168 KV tokens；实际服务上限5100秒 |
| 补跑maxseq16，run `bb6a20bae32640c0b924511637d6a222` | [config](../evidence/session003/recon-final-runtime/config-bb6a20bae32640c0b924511637d6a222.json)、[state](../evidence/session003/recon-final-runtime/state.json)、[log](../evidence/session003/recon-final-runtime/server.log) | 03:01:32 ready、03:46:43 stopped；44.93GiB、327168 KV tokens；动态服务上限2883秒 |

两份maxseq16原始日志现在都直接证明44.93GiB／327168，而非早期现场转述。原runtime归档日志中含maxseq4及maxseq16两个run marker；新服务启动前旧日志在精确旧ID停止、token/端口检查后rename保存，新日志从新run开始。

原生栈为vLLM `0.18.1+das.fa71803.dtk2604`、Torch `2.10.0+das.opt1.dtk2604.20260325.g6b060a`、Python3.10.12、BW/BF16，见 [env](../evidence/recon-functional-env-003.log)。成功路径为TP1、eager、maxlen32768、显存比例0.85、服务image2/video0；默认env仍是maxseq4，开发矩阵使用独立maxseq16覆盖。日志显示chunked prefill的`max_num_batched_tokens=2048`与`enable_prefix_caching=True`；2048是实际引擎配置观测，不是声称extra argv显式传入。

所有开发客户端使用seed=1、每请求一张1024×1024随机图、文本目标4096/16384、输出1024、temperature0、ignore-eos。持续服务内跨轮次和并发复用prefix cache，第二轮并非独立冷启动；retry又重启了服务，因此不把跨run汇总当作同一缓存状态下的公平性能基线。4096/16384仅描述生成文本，视觉/chat token与重编码差异尚未认证。正式模板的SGLang客户端与本次native vLLM random-mm负载未建立同口径等价关系，不能把此表写成正式成绩。

| 实际执行源码 | SHA256 |
|---|---|
| 原runtime | `04b0c91fd0596360f142021800f1487af5a83264cb6de5971b954b45dc80b422` |
| 独立补跑runtime | `66e9a97c63469698bfc09fbc1158a28fbd7a48f52877abb21156785a33ab1221` |
| 两份run的driver | `bc09711bb58fb5a7ae4676a840956a4da1caf49f6e6034b8015c5dd9a5bae14e` |
| retry helper | `56c6d4d693d9115794479a2e546c08cc90615ae814b1460508d7c11ada6afb4d` |

[源码更新记录](../evidence/session003/source-update-003-004.json)与[控制事件](../evidence/session003/recon-retry-control/control-events.json)保留执行身份。实际retry run ID以control/state原文`bb6a...`为准，导出使用显式`--expected-run-id`绑定该原文。runtime66已实际完成新的GPU服务launch/stop；其它后续CPU工具不得继承这项GPU执行结论。

## 4. 功能与早期校准

[新七项smoke](../evidence/recon-functional-smoke-003.jsonl)在01:17:16完成：health、models、单图、多图、多轮两次及图像SSE，均通过；非流式与流式均stop、未截断。[红方块](../evidence/recon-functional-images-heldout_red_square-003.png)和[蓝圆形](../evidence/recon-functional-images-heldout_blue_circle-003.png)是本地生成的功能图片，不是官方空间推理评测数据。

| 独立校准 | 请求/输出 | 原生duration s | Client wall s | 输出token/s | 完整性 |
|---|---|---|---|---|---|
| maxseq4 / c1 / round1 | 8×1024 | 380.702118 | 414.878568 | 21.518136 | verified |
| maxseq16 / c8 / round1 | 16×1024 | 107.451735 | 141.348396 | 152.477761 | verified |

两份校准都为`full_matrix=false`，空round2文件原样保留。初始26份扁平副本、来源SHA和当时未取得profile的审计状态保存在 [先期审计](../evidence/recon-completed-audit-003.json)，不追改该历史快照。本报告通过最终原始日志补足profile。两校准并发与服务maxseq同时变化，不将吞吐之比声称为单一代码改动的优化倍数。

## 5. ZIP、原始文件与证据闭合

最终ZIP为4287828字节，SHA256 `e120e32c0d00c67545c7a50c7a572ba3d8a828152cabbad180821ac7e5114bef`，含67个payload及1个manifest。已验证V4身份、文件全集、全部大小/SHA/CRC及安全路径，再提取到全新目录；68份文件合并canonical时均为新文件，覆盖差异文件数为0。见 [ZIP核验](../evidence/recon-final-zip-verification-003.json)、[合并清单](../evidence/recon-final-merge-003.json)和 [原始导出manifest](../evidence/session003/recon-final-export.json)。68份新增raw路径及审计/平台/测试记录均已登记到task的`evidence_files`，不靠摘要替代原件。

默认审计只读；本次将报告写入新文件。严格审计命令使用最终下载的driver源码作为身份对照，不导入Torch/vLLM、不安装依赖、不调用env_check。原run、retry和最终状态均保持下载时字节，失败和未尝试单元没有被删除。任务证据、Markdown依赖与所有68份raw的只读闭合检查见 [检查记录](../evidence/recon-final-local-closure-003.json)。

## 6. 清理、平台终态及本地验证

[最终清理快照](../evidence/session003/recon-final-runtime/cleanup-check.json)于03:47:37确认新run的supervisor/server启动token均不再存活、port30000关闭，`owned_service_cleanup_verified=true`。[helper清理](../evidence/session003/recon-retry-control/cleanup-check.json)与原服务清理也分别保留。服务清理和平台关机是两个证据层级。

[checkpoint003](../evidence/platform-checkpoint-20261020-003.json)实际确认三个Notebook均关机，本轮运行的A/B均选择保存环境。A于03:48:43左右手动确认、03:49观察停止，早于03:55自动兜底。可见余额493.5055555555卡时、快照差2.2605555555卡时可能包含延迟结算，不声称等于墙钟卡时或最终账单。checkpoint002继续仅为历史记录。

最新 [Windows完整回归](../evidence/local-validation-session003-final-003.json)为175项：169通过、6个明确Linux/WSL skip；46个canonical Python源在测试前后SHA一致，见 [日志](../evidence/local-tests-session003-final-003.log)。新增局部审计Windows8/8、WSL8/8单独记录，其中回归覆盖“整个retry不完整但第一格完整→11/12，complete仍false”。本次文档/JSON/原件闭合检查属于后续本地检查，不冒充重新执行完整测试或GPU。

## 7. 剩余工作

1. `round2_16384_1024_c64`仍无完整native结果；本轮遵守既定时限，不延长或自动补跑。
2. 正式固定数据、GPU baseline/参考、原版判分器与selfcheck、适用指标/解码/图像负载仍缺；功能和开发吞吐不能替代正式验收。
3. 可获取的完整私有镜像引用、干净镜像一键复跑及代码仓库最新结果同步/下载核验仍需独立完成。
4. 赛事提交按用户后续通知执行，当前`official_accuracy=false`、`official_performance=false`、`formal_ready=false`、`submitted=false`。
