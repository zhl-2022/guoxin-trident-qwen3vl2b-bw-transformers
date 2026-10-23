# Session003 三题开发交付报告

更新日期：2026-10-20（北京时间）。**本轮开发证据已整理，正式验收仍受材料缺口阻挡；赛事尚未提交。** 三题均取得实际BW GPU结果。Recon原执行与独立补测合计覆盖11/12格、480条完整请求、491520个输出token；16K/c64第二轮超时，完整矩阵没有通过。所有平台实例已关机，本轮A/B均保留环境，不再继续占卡。

本报告冻结本轮开发、审计和停机事实。对应结构化状态见 [session003-delivery-status.json](../evidence/session003-delivery-status.json)。三题已严格核验的post-metadata仓库快照单独列明；报告冻结后的最终增量、GitHub与交付包核验，由工作区另存的closure记录承接，不提前写为成功，也不反向装入被验证manifest形成自引用。

## 1. 交付结论

| 任务 | 已完成的实际开发结果 | 明确限制 | 详细材料 |
|---|---|---|---|
| Recon2Reason，T02 | 七项功能检查；两个开发校准点；原矩阵10格加独立补测1格，合计480条完整请求 | 16K/c64第二轮超时，11/12格开发覆盖；正式精度/性能未验收 | [任务README](../tasks/recon2reason-bw-vllm/README.md)、[结果审计](session003-recon-results.md) |
| RoboBrain2.5，T02 | 24项BW数值对照，含六项广播预算检查；七次新尺寸图片请求完整结束 | 合成图定位不是正式accuracy，尚缺正式图集、空间指标、图像政策及性能 | [任务README](../tasks/robobrain-bw-vllm/README.md)、[结果审计](session003-robobrain-results.md) |
| Qwen3-ASR，T04 | 新隔离venv实际准备；中/英及39秒派生WAV转写；完整WAV上传后SSE输出完成 | 无已核人工reference，未测正式WER/CER、自然连续长录音或实时音频输入 | [任务README](../tasks/qwen3-asr-bw-vllm/README.md)、[结果审计](session003-asr-results.md) |
| 共同工程交付 | 社区根文件及来源齐备；最新175项Windows回归：169通过、6项明确平台跳过；三题post-metadata快照strict核验通过 | 最终收尾增量、GitHub及bundle核验另存closure，不由旧快照覆盖 | [社区材料](../evidence/community-metadata-003.json)、[最新回归](../evidence/local-validation-session003-final-003.json) |

当前保持 `formal_ready=false`、`official_accuracy=false`、`official_performance=false`、`submitted=false`。功能、数值、开发压测与正式验收分别记录。

## 2. Recon：11/12格已核开发覆盖，末格超时保留

### 2.1 原执行与独立补测

[最终离线审计](../evidence/recon-matrix-final-audit-003.json)核对了原矩阵、独立重试、驱动/原生源码身份、每格原始JSON、日志SHA和轮次汇总。原矩阵为文本目标输入4096/16384 × 并发1/8/64 × 两轮，共12格、计划608请求。

| 执行与范围 | 实际结果 | 完整性处理 |
|---|---|---|
| 原矩阵前10格 | 352条完整请求、360448输出token | 每条正文非空、errors空、输出恰为1024，原文/哈希/汇总一致 |
| 原16384/c64第一轮 | timeout，客户端wall745.996889秒 | 原始超时保留，没有可核完整native结果 |
| 原16384/c64第二轮 | not_attempted_total_timeout | 明确未执行，不计作完成 |
| 独立16384/c64第一轮 | 128条完整请求、131072输出token；客户端wall1388.66132秒，原生输出吞吐98.16924655749037 token/s | 逐请求完整性及原文/哈希通过 |
| 独立16384/c64第二轮 | timeout，客户端wall1307.366197秒，无native JSON | 该格未通过，不能用第一轮替代计分 |
| 组合开发覆盖 | **11/12格，480条完整请求、491520输出token** | `matrix_complete_verified=false`、`combined_development_coverage_complete=false` |

480是已完整核验的请求数，不代表全部尝试数；超时格没有完整native计数，不能猜其中完成或失败的请求数。独立重试没有覆盖原矩阵的失败终态，也没有被合并成一次12格完整通过。

实际独立run ID为 **`bb6a20bae32640c0b924511637d6a222`**。原始state/control记录中的ID没有改变；此前摘要的首四位误传已纠正，以原始证据为准。服务03:01:32 ready，第一格03:01:34开始，第二格03:24:43开始，服务03:46:43 stopped。最终导出记录同run的supervisor/server身份均不存活、端口30000关闭、`owned_service_cleanup_verified=true`。

[最终ZIP核验](../evidence/recon-final-zip-verification-003.json)确认归档4,287,828字节、67个payload加manifest共68条目，SHA256为 `e120e32c0d00c67545c7a50c7a572ba3d8a828152cabbad180821ac7e5114bef`；CRC、manifest和逐文件哈希通过。[合入记录](../evidence/recon-final-merge-003.json)保留68份文件的来源，未覆盖不同的既有原件。矩阵是否完整以离线审计为准，不以ZIP完整性代替压测完整性。 [本地闭合检查](../evidence/recon-final-local-closure-003.json)另存原始文件、链接和收集范围的检查时点。

### 2.2 功能、校准与测量范围

[七项功能原文](../evidence/recon-functional-smoke-003.jsonl)覆盖健康、模型身份、单图、多图、多轮和图像输出流，生成均stop结束、未截断。另有[已下载校准审计](../evidence/recon-completed-audit-003.json)：

| 独立开发校准 | 完成/失败 | 每条实际输出 | 输出吞吐 |
|---|---|---:|---:|
| maxseq4 / c1 / 4096文本目标输入，第一轮 | 8/0 | 1024 token | 21.518136 token/s |
| maxseq16 / c8 / 4096文本目标输入，第一轮 | 16/0 | 1024 token | 152.477761 token/s |

这些校准请求不与最终矩阵的480条重复累加。4096/16384是文本目标输入，图像/chat额外token与重编码差异未认证；固定seed和持久服务允许跨轮/负载的prefix-cache复用，未验证独立冷启动或禁缓存基线。native vLLM `random-mm` 与正式指定客户端/图像口径是否等价仍待确认，吞吐不作为正式成绩或单一改动的优化倍数。

最终原始服务日志直接记录maxseq16配置的 **44.93GiB、327168 KV tokens**，补齐此前只有现场观察的证据。该日志与状态已纳入最终审计，不用旧maxseq4的44.95GiB/327296数据替代。

## 3. RoboBrain与ASR实际结果

### 3.1 RoboBrain

兼容源码SHA256为 `083dc53f3e4f8171b774469bffa724c2269a9d289769f56a9d7a5624cd81659f`。[24项BW数值报告](../evidence/robobrain-numerical-003.json)全部通过，最大绝对误差0.0009765625；六项广播检查逐块满足score预算。相同集合两次执行均通过，仍计24个不同case。

[七次图片请求](../evidence/robobrain-heldout-003.jsonl)均非空、stop结束、无截断。640×448红图和512×512蓝图的合成框IoU为0.8839659402511187/0.9024416623854203，两次点选均在框内；实际vision长度1120/1024，81次eligible Math调用完成。run ID `c43dadd550fb44dabd84f52df49711d5`的ready/stopped和清理结果保留。

范围为张量数值及合成图功能：没有正式空间accuracy、轨迹/深度真值或服务性能。默认1024×1024初始化profiling已实测，不保证任意大图。原件、历史失败及平面副本双SHA见[源码和证据审计](../evidence/robobrain-source-audit-003.json)。

### 3.2 Qwen3-ASR

新任务venv实际安装七个锁定CPU wheel，12个受保护原生包版本/目录未变，解码与重采样通过，见[prepare结果](../evidence/asr-clean-prepare-003.json)。

| 实际输入 | 时长 | 单次HTTP往返 | 非空返回字符 |
|---|---:|---:|---:|
| 中文WAV | 4.2039375秒 | 4.889441秒 | 14 |
| 英文WAV | 15.05125秒 | 1.807355秒 | 188 |
| 中英人工拼接WAV | 39.260375秒 | 4.204148秒 | 393 |

[样本来源与转换](../evidence/asr-fixture-manifest-003.json)、[中文](../evidence/asr-vllm-smoke-zh-003.jsonl)、[英文](../evidence/asr-vllm-smoke-en-003.jsonl)、[39秒输入](../evidence/asr-vllm-smoke-long-003.jsonl)保留原文及SHA。39秒输入为中英片段重复加静音，不是自然连续长录音。没有独立人工reference，未计算WER/CER；普通转写无finish reason，日志未截短不代表模型完整性已证明。

[实际中文SSE](../evidence/asr-vllm-sse-zh-003.jsonl)包含12个事件、stop、DONE和65+11=76 token usage，保留 `language Chinese<asr_text>` 前缀及非空正文。它是完整WAV上传后的输出流，不是实时音频输入。run ID `6b00889feafc4ea1af2133b5c4a3317a`已停止，警告及清理证据见[ASR审计](../evidence/asr-copy-provenance-003.json)。单次HTTP用时不构成正式RTF或吞吐成绩。

## 4. 工程材料、许可与版本验证

### 4.1 社区根文件

三题[社区工程材料](../evidence/community-metadata-003.json)已补齐，组装器提升以下五文件，同时保留根README为该题报告：

| 根文件 | 内容与边界 |
|---|---|
| model.properties | 真实Project ID为Recon15915、Robo15916、ASR15917；官方分类多模态/语音识别，实际框架pytorch/vllm，设备仅记实测BW |
| icon.png | 固定官方社区来源，原样410×410 PNG，来源及SHA已保存 |
| README_origin.md | 固定模型revision的发布方原文，原始字节保留 |
| LICENSE | **用户选择暂保留版权范围说明，未选择全仓许可证**；上游代码和模型权重许可分开，`license_ready=false` |
| requirements.txt | 原生环境声明，不安装/替换Torch或vLLM；ASR额外依赖通过隔离venv准备 |

没有猜BW型号后缀、完整镜像引用或未知许可。用户的版权范围选择已落实，不作为仍等待答复的阻碍。

### 4.2 最新CPU回归与实际执行版本

[final-003完整回归](../evidence/local-validation-session003-final-003.json)为 **175项：169通过、6项明确Linux/WSL相关跳过，exit0**，unittest用时56.183秒；[原始日志](../evidence/local-tests-session003-final-003.log)保留。46份canonical Python源在本轮测试前后SHA一致。此前[final-002](../evidence/local-validation-session003-final-002.json)记录的46份编译与CLI检查作为历史证据引用，**本轮未重复编译/CLI**；定向Windows8项和WSL8项审计回归不与175项累加。

版本边界分别记录：

- 共享runtime `66e9a97c...` 已在Recon独立补测中**实际完成GPU服务launch及同run stop/清理**，最终执行源码快照、state和token/端口检查已归档；不再只标CPU验证。
- `session003_robo_asr_checks.py` 的新控制编排补丁 `20f358fa...` 与SSE客户端补丁 `3ae83568...` 仅有CPU/loopback回归，没有重新执行Robo/ASR GPU流程；见[控制补丁记录](../evidence/rules/session003-run-identity-cpu-004.json)和[SSE补丁记录](../evidence/rules/asr-sse-integrity-cpu-004.json)。
- Robo/ASR原003执行快照继续绑定当时的编排器 `3fdd8ee4...`、runtime `04b0c91f...` 与SSE客户端 `20df684b...`；新版本不继承旧版本的GPU通过状态，历史失败和警告不改写。

## 5. 已核仓库快照与独立closure

三题Sourcefind和GitHub仓库均为private，是开发交付与备份，**不是赛事提交**。

| 任务 | Sourcefind私有仓库 | GitHub私有备份 |
|---|---|---|
| Recon | [guoxin-recon2reason-bw-vllm](https://developer.sourcefind.cn/codes/ghu_30ebea88e/guoxin-recon2reason-bw-vllm) | [zhl-2022/guoxin-recon2reason-bw-vllm](https://github.com/zhl-2022/guoxin-recon2reason-bw-vllm) |
| RoboBrain | [guoxin-robobrain-bw-vllm](https://developer.sourcefind.cn/codes/ghu_30ebea88e/guoxin-robobrain-bw-vllm) | [zhl-2022/guoxin-robobrain-bw-vllm](https://github.com/zhl-2022/guoxin-robobrain-bw-vllm) |
| ASR | [guoxin-qwen3-asr-bw-vllm](https://developer.sourcefind.cn/codes/ghu_30ebea88e/guoxin-qwen3-asr-bw-vllm) | [zhl-2022/guoxin-qwen3-asr-bw-vllm](https://github.com/zhl-2022/guoxin-qwen3-asr-bw-vllm) |

已严格下载核验的post-metadata冻结快照为：

| 快照 | main commit | strict ZIP结果 |
|---|---|---|
| Recon | `af6eccdcf268c78fc871076c869ce0e21018cabd` | [315/315逐字节一致](../evidence/sourcefind-recon-post-metadata-003-verification.json) |
| RoboBrain | `40884bbd`（前缀） | [318/318逐字节一致](../evidence/sourcefind-robo-post-metadata-003-verification.json) |
| ASR | `b4913d3e`（前缀） | [317/317逐字节一致](../evidence/sourcefind-asr-post-metadata-003-verification.json) |

三份快照均无missing/extra/mismatch。Robo此前旧task JSON的唯一差异已经修复；旧核验记录仍保留。原始003证据提交及旧快照没有被新结果覆盖。

这些通过结论只绑定各自commit/archive。最终Recon原始结果、停机证据及本报告会形成后续共同增量；**最终远端增量、GitHub commit和bundle核验尚不能由上述快照证明**。它们由工作区独立closure记录保存，报告冻结后不再为了把核验记录装回自身而循环修改manifest。该closure不作为本报告已完成的声明。

## 6. 实际停机、余额与镜像身份

### 6.1 所有实例已停止

[平台最终checkpoint](../evidence/platform-checkpoint-20261020-003.json)与[已关机截图](../evidence/screenshots/notebooks-stopped-final-003.png)确认：

| 平台 | 实例ID | 实际终态 |
|---|---|---|
| A：Recon | 2112186448839770114 | 03:48:43左右选择保存环境ON并确认手动关机；03:49观察已关机 |
| B：Robo→ASR | 2112237945233408001 | 02:03保存环境并关机，最终页面仍为已关机 |
| reference | 2097876209970565121 | 已关机，本轮没有启动 |

Recon服务已在03:46:43先行stopped，原始导出核对supervisor/server token及端口均已释放。A在03:55后端自动关机前手动提前停止，[自动关机确认](../evidence/screenshots/recon-autooff-0355-confirmed-003.png)作为兜底配置保留，**没有将03:55配置冒充实际关机时间**。A本轮最多3卡时授权保持，checkpoint记录的保守wall卡时上界2.82不作为账单换算。

[关机后余额截图](../evidence/screenshots/balance-after-stop-003.png)显示493.5055555555卡时，本轮前为495.766111111，快照差2.2605555555。该读数可能延迟结算，不能等同最终账单或wall卡时；没有充值或购买。

### 6.2 镜像可定位信息与剩余缺口

SCNet实际私有镜像ID为2071293281204580354，版本2082674417745739777；名称 `vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`，tag为 `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`。镜像表单没有给出完整registry引用，51GB导出已取消，没有因此修改镜像。

公开ModelZoo的基础镜像 `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423` 不能冒充实际Jupyter镜像的完整引用。Torch以实测2.10为准，不用UI的2.9替代。实际可供评审获取的完整registry/repo:tag仍缺。

## 7. 正式验收缺口与下一次执行边界

| 缺口 | 影响 |
|---|---|
| 三题原版evaluator/selfcheck及输入输出schema | smoke和自建开发审计不能替代赛事判分 |
| 固定评测子集与独立参考 | 合成图及发布方音频示例不能替代官方数据 |
| 各模型官方GPU baseline | 未能判断是否达到正式精度门槛 |
| 指标、解析、解码与正式负载 | Recon/机器人空间口径、图像要求，ASR WER/CER规范化及性能口径仍待确认 |
| Recon16K/c64第二轮 | 独立补测仍超时，12格完整开发性能覆盖没有通过 |
| 最终可获取的完整镜像地址/tag、干净镜像复跑 | 当前镜像ID与tag不足以形成正式从零复现闭环 |

查询依据见[公开材料复核](official-material-search-round2-20261020.md)和[云文档复核](cloud-materials-and-repository-access-20261020.md)。分题表图片内verify.py文字未提供可下载URL；不外推为所有渠道均未发布。独立认领回执和可选digest不额外作为开发前置条件。

本轮GPU工作已停止，原始成功、超时和未执行记录全部保留。接下来只做最终共同增量、仓库/交付包核验并另存closure；不因末格超时擅自续开卡。正式材料到位后，再按用户授权安排精度、完整性能及镜像从零复跑。**最终赛事提交继续等待用户明确通知，本轮没有执行提交。**
