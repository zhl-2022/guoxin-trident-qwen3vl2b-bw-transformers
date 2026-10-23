# Recon2Reason-Reasoning-4B on BW + vLLM 0.18.1（session004 最终实测记录）

## 1. 简介

本文记录 `BAAI/Recon2Reason-Reasoning-4B` 的 BW 单卡、BF16、vLLM 适配。本轮已完成真实服务功能检查，并核验 UTC `2026-10-23T12:14:56.329311` 的最终导出：八格客户端全部正常退出、八份 native JSONL 均已保存；三格合计十二条空正文，因此严格整格检查为五格合格，c1 的四格本轮未跑。服务已停止并独立核验进程/端口清理。`formal_ready=false`、`submitted=false`；这是最终实测原件记录，不是完整 12 格或官方验收通过声明。最后验收表由用户手动提交。

最终 ZIP 为 1,164,740 字节，SHA256 `baf2f84d3d717d3256a9d63ece4f274912ac6285a625ff59eb9ec9dd6a472671`，与远端打印值一致；255 个 ZIP entry 对应 254 份 payload 加 manifest。全部路径、CRC、大小、SHA 和解包回读通过，91 份脚本导出前后 SHA 一致。见 [最终独立完整性检查](../evidence/session004/remote-final-001.local-verification.json) 和 [最终独立审计](../evidence/session004/remote-final-001-audit.json)。exporter 保留实时、非原子的采集语义；服务终态由另外的清理原件核验。

UTC `11:30:29.978614` 的中间 ZIP（SHA256 `8b4481d5fcc8d1080cd5ffa8cf14509ab9577d8ea4b30089eb5b1f049ce702d5`）及其 [检查](../evidence/session004/remote-intermediate-001.local-verification.json)、[阶段审计](../evidence/session004/remote-intermediate-001-audit.json) 原样保留。前六份 native 文件在最终导出中逐字节未变，空输出记录没有因最终整理被删除。

权重：[发布方模型](https://huggingface.co/BAAI/Recon2Reason-Reasoning-4B)。既有私有仓库：[光合](https://developer.sourcefind.cn/codes/ghu_30ebea88e/guoxin-recon2reason-bw-vllm)、[GitHub 备份](https://github.com/zhl-2022/guoxin-recon2reason-bw-vllm)。这些地址已保存 session003，本轮 stage004 和远端分支按独立构建、推送及回读回执核对。实际最终镜像 `registry/repo:tag` 与评审获取方式待填写，展示名不能替代完整引用。

## 2. 验证环境

<!-- session004-platform-closure -->

本轮已独立核对[SCNet 平台关机与余额记录](../evidence/platform-checkpoint-20261023-004.json)：A/B 两个本轮实例及旧参考实例均为“已关机”；A 手动关机并选择保存环境，B 按预定时限自动关机。可见余额为 **488.0238888888 卡时**，与开机前可见余额差 4.6686111111 卡时；余额差可能包含延迟结算，不当作最终账单。资源为华中一区 A 区 113 组、BW64GB×1、CPU15核、内存59GB。账户镜像 ID `2071293281204580354` / version `2082674417745739777`，Notebook 可选择标签 `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`；完整实际 registry/repo:tag 和评审获取方式尚缺，digest 仅为可复现补充项。平台配置不替代任务原始环境/驱动实测。

本轮代码仓库目标：[zhl-2022/guoxin-recon2reason-bw-vllm 的 codex/session004 分支](https://github.com/zhl-2022/guoxin-recon2reason-bw-vllm/tree/codex/session004)。仓库保持私有；实际推送 commit 和逐文件回读以独立交付回执为准，评审权限仍须落实。光合既有三仓库的 session003 与本轮 GitHub 分支分别记录，不假称光合已同步新版。

历史镜像展示名为 `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`。session003 的 [env.log](../evidence/recon-functional-env-003.log) 另保留历史环境；本轮客户端实测版本直接取自 [client-identity.log](../evidence/session004/remote-final-001/project/evidence/session004/recon-sglang-matrix-A-001/client-identity.log)：Python 3.10.12、Torch `2.10.0+das.opt1.dtk2604.20260325.g6b060a`、vLLM `0.18.1+das.fa71803.dtk2604`、Transformers 5.5.0、Triton `3.6.0+gitc73250c4.staging`，原生包位于 `/usr/local/lib/python3.10/dist-packages`。客户端 SGLang 为隔离环境中的 `0.5.12`，服务框架仍为 vLLM。

| 本轮字段 | 当前报告内容 |
|---|---|
| 实例 ID、CPU/内存/存储、实际设备 | A `2112186448839770114`；平台 BW64GB×1/15核/59GB；存储配额与独立设备型号清单未新增认证 |
| 镜像 ID/version、完整引用与访问方式 | 待实际最终镜像记录 |
| 解释器与原生包 | 服务 `/usr/bin/python`；客户端 `/root/private_data/sglang-client-004c/bin/python`；版本见本轮 identity 原件 |
| 模型路径 | `/root/private_data/SothisAI/model/Aihub/Recon2Reason-Reasoning-4B/main/Recon2Reason-Reasoning-4B`；本报告未新增全量 checkpoint SHA 认证 |
| 实际精度、TP、上下文、端口 | BF16、TP=1、32768、`127.0.0.1:30000`；矩阵服务 max-num-seqs=16 |
| DTK/HIP 与设备完整清单 | 当前版本标签含 DTK26.04；独立 env/设备清单及最终镜像引用仍待补齐，不从标签推导完整驱动事实 |
| 服务映射及截图 | 服务仅 127.0.0.1:30000，无自定义外网服务映射；关机截图在本轮平台记录中 |

## 3. 适配优化方案

原成功路径保留权重、结构、tokenizer 与采样语义，使用原生 vLLM，无新增模型 attention patch。框架原日志中的 GELU tanh→exact 行为保留披露，是否满足正式精度须靠固定集验证。

本轮增加透明根 `verify.py`、默认功能 fixture、真实官方 selfcheck-only 执行桥，以及与 vLLM 服务分离的 SGLang 压测客户端。见 [入口检查](../evidence/session004-interface-validation.json)、[官方桥 CPU 检查](../evidence/session004/official-bridge-validation-20261023/validation.json)。Ubuntu 缺少 ensurepip 的修复使用新 venv + 系统可见 pip，仍以 venv 解释器安装，见 [004c 检查](../evidence/session004/sglang-bootstrap-validation-004c-20261023/validation.json)。依赖准备的 CPU 检查不当作 GPU 或正式评测成绩。

实际执行使用 [004g driver](../scripts/t02_sglang_bench.py)，SHA256 `ff8eef852e19f336e99884cc031353627ca304c6a7534c22375116758a850000`。原版 SGLang `0.5.12` 的 image 数据集允许 `sglang-oai-chat`，其 sender 为通用 `/v1/chat/completions`；该 alias 不把 vLLM 服务变成 SGLang 服务。DAS 客户端 import 需要读取设备标签，本次明确允许 device discovery，并原样保留调用 shell 的可见设备变量，没有扩展设备范围；原始策略位于运行 JSON 的 `client_identity.client_device_discovery`。

exporter 对 Python 源码只保存导出前后 SHA，不保存源码 payload。重新分析使用已单独核对 SHA 的 [export-manifest.json](../evidence/session004/remote-final-001/export-manifest.json)，绑定原始输出目录、run/native 文件大小与 SHA、004g producer 和五份原版客户端模块的来源路径/SHA。记录明确标为 `export_manifest_before_after_script_hash_only`，没有补造远端源码文件。原版 `bench_serving.py` SHA 为 `129e40b9e361be192306a9df56df41507f0397d5768696db3a9b994421daa259`。

## 4. 服务启动脚本

canonical 入口为 [env.sh](../tasks/recon2reason-bw-vllm/env.sh)、[run.sh](../tasks/recon2reason-bw-vllm/run.sh)、[stop.sh](../tasks/recon2reason-bw-vllm/stop.sh)。默认 max-num-seqs=4；本轮矩阵覆盖为 16。实际 [config 原件](../evidence/session004/remote-final-001/project/evidence/session004/recon-sglang-matrix-A-001/runtime/before/config-c2277b5d03b54683a5c7ba0193e5e55b.json) 记录 BF16、TP=1、上下文 32768、显存比例 0.85、eager、max-num-seqs=16、image=2/video=0，服务 run ID 为 `c2277b5d03b54683a5c7ba0193e5e55b`。创建 UTC `10:41:53.518926`，ready UTC `10:43:31.275280`；完整 argv、6200 秒监督时限和客户端 6000 秒总时限均按原件披露。

独立仓库根入口由 [session004 装配器](session004-assembler-usage.md) 透明生成，已在系统临时目录验证参数转发和缺官方材料时返回 2。最终文件身份以 stage004 的 repository-manifest004.json 与独立构建回执为准；本工程可使用 `tasks/recon2reason-bw-vllm/` 对应 canonical 入口，下面是装配后仓库根目录的命令：

```bash
bash run.sh --dry-run
# 以下仅在选定最终镜像、模型和本轮运行限额后执行。
bash run.sh
bash env_check.sh
bash smoke.sh
python verify.py --selfcheck
source env.sh && python verify.py
bash bench.sh
bash stop.sh
```

正式 verify 和正式 bench 需要真实官方材料/适用契约接入。缺材料时返回 2；未知评分 schema 的完整执行记录仍为 `executed_uninterpreted`。本轮矩阵实际由 [归档 controller](../evidence/session004/remote-final-001/project/evidence/session004/executed-recon-matrix-driver.sh.txt) 调用原版 SGLang 客户端，不能把 root `bench.sh` 的正式结果预先填为通过。

矩阵 producer 从 UTC `10:43:32.061379` 运行到 `12:11:57.025297`，实际 wall=5304.963429 秒。服务 state 在 UTC `12:12:05.973275` 记录 `stopped`、`reason=stop requested`、elapsed=5412.454 秒。UTC `12:14:55.823368` 的 [独立 cleanup 原件](../evidence/session004/remote-final-001/project/evidence/session004/recon-final-cleanup-001.json) 再核对相同 run ID，supervisor/server 的进程 token 都不存在，30000 端口关闭，`cleanup_verified=true`。这些证明服务清理；SCNet Notebook 关机与账单终态由平台另行确认，不从进程停止推导。

## 5. Smoke 测试

无参 smoke 默认使用 [固定 fixture manifest](../fixtures/smoke-session004/manifest.json) 中两张不同内容的合成图，可用两个 `--image` 完全替换。检查健康、模型名、单图、多图、多轮、流式，保存输入 hash 和原始回答；非空回答不能当空间推理正确率。

本轮 [smoke.log 原件](../evidence/session004/remote-final-001/project/tasks/recon2reason-bw-vllm/smoke.log) 记录七项检查全部完成：health、models、single_image、multiple_images、multiturn_first、multiturn_followup、streaming_image。功能检查 ID 为 `2a5aa189359a4aa6a422f89362489147`，UTC `09:54:33.280337` 写出汇总；与稍后矩阵服务 run ID 分开保存。非流式四个回答均非空、`finish_reason=stop`、无 length 截断；流式有 DONE、stop、非空文本和空错误列表。独立回读结果见阶段审计的 `A.Recon_functional`，最终原件 SHA 与中间版一致。这些检查证明功能/API 交换，不提供空间推理正确率。session003 的 [旧七项功能原件](../evidence/recon-functional-smoke-003.jsonl) 保持历史身份，没有替代本轮结果。

## 6. 精度测评

正式结果：**未填；分题材料尚未在本报告闭合**。需要本模型固定数据/reference、原版 evaluator/selfcheck、GPU baseline、图像预处理、最大输出、思考/解码和答案/坐标解析契约。不能把发布方完整性检查、合成图回答、OCRBench/MMMU 名称或通用阈值擅自组合为本题验收。

`python verify.py --selfcheck` 只执行真实脚本自检；缺脚本返回 2，不生成假结果。正式报告需要归档原版文件 SHA、自检输出、逐样本预测、`verify_result.json` 和评分解释。精确待填字段见 [report-fields](stage004-report-fields.json)。

## 7. 性能参考

历史 [最终审计](../evidence/recon-matrix-final-audit-003.json) 核验 native vLLM random-mm 开发覆盖 **11/12 格、480 个完整请求、491520 个输出 token**。最后 `round2_16384_1024_c64` 超时且无 native JSON。输入目标仅是随机文本，1024×1024 一图额外 token/chat 包装未认证为模板总长度；持续 prefix cache 与独立重启条件也分别保存。这些结果不改名为 SGLang client 数据。

本轮实际 [run.json](../evidence/session004/remote-final-001/project/evidence/session004/recon-sglang-matrix-A-001/run.json) 选择 4096/16384 文本目标、1024 输出、并发 8/64、两轮，共八格、576 个测量请求和八次 warmup；八个客户端进程全部 `exited/0` 且已回收，各有一份 native 原件。**本轮没有执行 c1 两档的两轮；默认完整 12 格/608 请求是另一份计划，不能写为本轮已跑。** 图像为一张 1024×1024 随机 JPEG，seed=1；请求体显式 temperature=0、ignore_eos=true、stream usage=true。第二轮是报告取数轮，但先后顺序不等同于独立重启/冷缓存；完整输入 token 含图像/chat 包装，4096/16384 是生成文本目标，不是已认证的模板总 token 长度。

原版客户端合法支持 CLI `--request-rate inf`，但默认 `json.dumps` 把它输出为非标准 JSON 数值 `request_rate: Infinity`。原 driver 的严格 JSON parser 因此把已正常退出的格标为 `invalid_native_result`；Infinity 本身并不是标准 JSON 数字。独立 [analyzer](../scripts/analyze_sglang_results_004.py) 只把顶层该正 Infinity 常量记录为字符串 `"inf"`，保留原始 SHA 和 producer 状态；其他 NaN/Inf、溢出、重复 key、超时或 mock 结果仍不通过。没有改写 driver、run.json 或 native 原件。真实重分析见 [analysis.json](../evidence/session004/remote-final-001-analysis/analysis.json) 和 [summary.md](../evidence/session004/remote-final-001-analysis/summary.md)。

UTC `12:14:56.329311` 最终导出的八格结果如下，轮次链接指向对应原始 native JSONL。token/s、TTFT、TPOT 均为客户端原报值；不合格格的数值保留披露，不能计为完整输出通过。wall 包括客户端启动与 warmup 等开销，不能与 native 测量 duration 混用。

| 轮次 | 文本目标 | 并发 | 原报完成 | native duration(s) | 输出token/s | mean TTFT(ms) | mean TPOT(ms) | 客户端wall(s) | 严格回读 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [1](../evidence/session004/remote-final-001/project/evidence/session004/recon-sglang-matrix-A-001/native/round1_4096_1024_c8.jsonl) | 4096 | 8 | 16 | 111.2563 | 147.2636 | 2685.2537 | 51.5739 | 156.2312 | 完成 |
| [2](../evidence/session004/remote-final-001/project/evidence/session004/recon-sglang-matrix-A-001/native/round2_4096_1024_c8.jsonl) | 4096 | 8 | 16 | 110.4682 | 148.3142 | 1963.9933 | 51.9127 | 160.9553 | 完成 |
| [1](../evidence/session004/remote-final-001/project/evidence/session004/recon-sglang-matrix-A-001/native/round1_4096_1024_c64.jsonl) | 4096 | 64 | 128 | 486.5786 | 269.3748 | 139825.6770 | 57.6298 | 619.0592 | 完成 |
| [2](../evidence/session004/remote-final-001/project/evidence/session004/recon-sglang-matrix-A-001/native/round2_4096_1024_c64.jsonl) | 4096 | 64 | 128 | 486.6267 | 269.3482 | 139972.6450 | 57.5879 | 608.1738 | 完成 |
| [1](../evidence/session004/remote-final-001/project/evidence/session004/recon-sglang-matrix-A-001/native/round1_16384_1024_c8.jsonl) | 16384 | 8 | 16 | 211.0890 | 77.6165 | 19519.5332 | 83.3836 | 268.0748 | 完成 |
| [2](../evidence/session004/remote-final-001/project/evidence/session004/recon-sglang-matrix-A-001/native/round2_16384_1024_c8.jsonl) | 16384 | 8 | 16 | 209.9050 | 78.0544 | 18418.5317 | 83.8609 | 264.9646 | 不合格：一个空输出 |
| [1](../evidence/session004/remote-final-001/project/evidence/session004/recon-sglang-matrix-A-001/native/round1_16384_1024_c64.jsonl) | 16384 | 64 | 128 | 1432.9932 | 91.4673 | 413590.4622 | 182.0693 | 1622.8783 | 不合格：七个空输出 |
| [2](../evidence/session004/remote-final-001/project/evidence/session004/recon-sglang-matrix-A-001/native/round2_16384_1024_c64.jsonl) | 16384 | 64 | 128 | 1433.9009 | 91.4094 | 426527.2897 | 169.9330 | 1600.6205 | 不合格：四个空输出 |

八份 native 文件合计原报 completed=576、output tokens=589824，errors 数组全空、output_lens 全部为1024；其中仅564条有非空生成文本，**这个计数不是正确率**。严格整格检查仍只计五格、304 请求、311296 个**客户端报告**输出 token，第二轮合格格为两格。空输出分布为：R2 16384/c8 的零基 index=8；R1 16384/c64 的 indices=2,26,47,70,95,97,119；R2 16384/c64 的 indices=0,23,50,95。十二条均 `generated_text=""`、TTFT=0、ITL=[]、errors=""，但 output_lens=1024。三格原报完成数和吞吐保留披露，不能用于“输出完整”合格结论；修复 Infinity 或删除坏样本不会改变这些事实。

本轮另以 [只读 collector 原件](../evidence/session004/remote-final-001/project/evidence/session004/recon-server-metrics-001/collector-final.json) 保存服务 `/metrics`。采集从 UTC `11:49:32.913679` 开始，到 `12:11:58.461899` 因连接拒绝结束，1340 个样本身份保持同一 run/PID/start token；最后有效采样为 UTC `12:11:56.998927`。collector 不发模型请求、不改服务，原始全文仅保留 first/latest/last-good，其余历史记录保存所选数值和全文 SHA。最终审计重新核对 collector 自带 manifest、1340 个样本以及首末原文的 SHA 和解析值。

最后有效服务计数为 `request_success_total{finished_reason="length"}=584`、`request_generation_tokens_count=584`、`generation_tokens_total=request_generation_tokens_sum=590080`，在数值上与 576 个测量请求加八次 warmup、`576×1024+8×32` 相同；原版 warmup 输出上限为32。**这只是服务聚合计数与理论目标的一致性支持**：collector 从最后一格运行期间才开始，没有零基线和逐请求隔离，不能证明每条请求长度，也不能挽救十二条空正文。`length` 是本轮 ignore_eos/输出上限设置下的服务标签，不能与 smoke 所需的 stop 混用。

原版 sender 会合并 reasoning_content/content，usage 缺失时可使用请求长度 fallback，native JSONL 也不单独认证 SSE 终态。结合聚合计数仍不能把每请求 exact output token、raw usage/SSE 独立认证或正式性能验收改为 true。

## 8. 注意事项

- 模板允许披露没有完成的负载；11/12 不能写成完整通过，也不自动表示整题不许提交。旧末格受总预算截断，尚未证明显存 OOM 或天然不支持。
- session004 最终为八格都正常退出、五格严格合格、三格共十二条空正文，c1 四格未跑；不能混入历史 003 拼成“12/12”，也不能把进程退出0改写为全部输出合格。中间原件与最终原件分别保留。
- 用户选择保留版权范围说明和上游许可；模板没有强制全仓开源许可证，内部 `license_ready=false` 不是额外赛事障碍。
- 原三仓库已由 [003 closure](../evidence/final-delivery-closure-003.json) 核验保存；004 是否装配、同步和可供评审访问是另一次核验。
- [本轮规则对照](session004-requirements-and-gaps.md)、[CPU 汇总](../evidence/session004-cpu-summary.json) 和 [analyzer 最新验证](../evidence/session004/sglang-native-reanalysis-validation-20261023/validation-003.json) 只支持各自范围，不替代正式精度或镜像复现。官方材料、实际可取镜像、干净环境复跑和最终访问方式仍须补齐。
- 后续最小补跑建议见 [离线计划](recon-session004-minimal-next-run-plan.md)。该文件列尚缺的 c1 两轮和三个空输出格的诊断/选择性复查条件；本轮没有按此重新开卡。
- 最终提交由用户手动完成；本报告不代写提交回执或当前平台已关机状态。
