# Recon2Reason-Reasoning-4B × BW × vLLM × BF16

## 1. 简介

本任务已在 SCNet 的 BW 单卡上，以原生 vLLM、BF16 启动 `BAAI/Recon2Reason-Reasoning-4B`。2026-10-20 的 session003 通过七项图文功能检查；最终原矩阵与独立补跑合计核验 **11/12 格、480 个完整请求、491520 个输出 token**。全部480条均恰为1024输出 token、正文非空、errors为空；最后一格超时且没有 native JSON，完整覆盖仍为 false。原矩阵完成10格／352请求，独立补跑仅第一格128请求完整，两份 run 都保持 `incomplete_or_failed`。详见 [session003 结果审计](../../docs/session003-recon-results.md) 与 [最终结构化审计](../../evidence/recon-matrix-final-audit-003.json)。

最终68条 ZIP 产物已逐字节核验并零覆盖合并。maxseq16 的原服务与独立补跑均已停止，最终 run ID 为 `bb6a20bae32640c0b924511637d6a222`；[清理快照](../../evidence/session003/recon-final-runtime/cleanup-check.json) 确认自有进程 token 均不再存活、端口关闭。[checkpoint003](../../evidence/platform-checkpoint-20261020-003.json) 实际确认三个 Notebook 均已关机，本轮运行的A/B均选择保存环境。2026-10-19 的首次功能及 [session002 快照](../../evidence/platform-checkpoint-20261019-002.json) 继续作为历史证据保留。

当前为开发交付候选，尚无正式精度或六档性能成绩。2026-10-20 的公开材料核查结果见 [评测材料与提交核查](../../docs/submission-readiness-audit-20261020.md)。该模型适用的正式指标、固定数据、GPU baseline、评测器和图像负载仍需确认；无需等待单独认领回执才能继续开发。

任务的唯一服务入口为 `run.sh`。保留本目录与根 `scripts/` 的相对结构，其他入口均读取同一份 `env.sh`；不要直接从另一题复制展开的命令。

```bash
cd tasks/recon2reason-bw-vllm
# 本地只检查命令展开，不导入 Torch/vLLM、不连接服务、不检查 GPU。
bash run.sh --dry-run
```

选好镜像后的运行顺序为：

```bash
bash env_check.sh
bash run.sh
bash smoke.sh --image /实际图片1.png --image /实际图片2.png
bash stop.sh
```

`stop.sh` 只停止本任务进程。释放 Notebook 和结束平台计费需另外在 SCNet 停机；历史午夜期限不是下一次启动的计费保护。

## 2. 验证环境

以下版本由历史 [环境探针](../../evidence/vllm-probe-001.json) 记录，session003 的 [实际 env.log](../../evidence/recon-functional-env-003.log) 再次记录 BW、BF16、Python 3.10.12 和相同原生 vLLM/Torch 栈；该 env.log 中 SGLang 为 null。新建干净镜像仍需重新自检，不能根据镜像展示元数据推定实际包版本。

| 项目 | 已观察值 |
|---|---|
| 私有镜像展示名 | `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423` |
| 完整镜像地址与 tag | 尚未取得；`IMAGE` 默认为空，展示名不能替代评审可获取的完整地址 |
| 镜像 digest | 尚未取得；可作为复现补充，官方报告附件未明示为必填项 |
| 实际实例 / 最终状态 | `2112186448839770114`；checkpoint003 于03:49观察到关机，保存环境开启 |
| 区域 / 资源组 | 华中一区【A区】；`113 / hx1hgbwnormal` |
| 设备 | BW × 1；`gfx936:sramecc+:xnack-`；68,702,699,520 字节显存 |
| Python | `3.10.12`，`/usr/bin/python3` |
| Torch | `2.10.0+das.opt1.dtk2604.20260325.g6b060a` |
| TorchVision | `0.25.0+das.opt1.dtk2604.torch2100.20260325.g860505` |
| vLLM / vLLM-HCU | `0.18.1+das.fa71803.dtk2604` / `0.18.1+das.dtk2604` |
| Transformers / Triton | `5.5.0` / `3.6.0+gitc73250c4.staging` |
| FlashAttention | `2.8.3+das.opt1.dtk2604.torch2100.20260330.g3f0061` |
| LightOp / AITER | `0.6.0+das.dtk2604.torch2100.2604101656.g193111` / `0.1.2+das.opt1.dtk2604.torch2100.2604212021.gd52133` |
| HIP / 平台类 | `6.3.26093` / `vllm_hcu.platforms.hcu.HCUPlatform` |
| 模型结构 / 精度 | `Qwen3VLForConditionalGeneration` / BF16 |
| 模型挂载 | `/root/private_data/SothisAI/model/Aihub/Recon2Reason-Reasoning-4B/main/Recon2Reason-Reasoning-4B` |
| 服务监听 | `127.0.0.1:30000` |

模型权重已做静态预检与哈希记录，见 [recon-preflight-hashed-001.json](../../evidence/recon-preflight-hashed-001.json)。`env_check.sh` 采集实际版本与设备并生成 `env.log`，不会安装包；没有可用 GPU 时返回 2。`run.sh --dry-run` 可以在不开卡时运行，但它不证明镜像与模型可用。

## 3. 适配优化方案

当前成功路径没有新增 Python 依赖，没有修改权重、结构、tokenizer 或采样策略，也没有项目专用 attention patch。[requirements-extra.txt](requirements-extra.txt) 仅说明这个事实；不自动安装或升级原生 Torch、vLLM 和 HCU 插件。

vLLM 原生日志显示 GELU 从 tanh 近似切换为 exact none；这是当前框架行为。功能通过不能证明这项数值路径差异满足正式精度门槛，仍需在固定数据上比较。

默认参数保持历史成功配置：BF16、TP=1、上下文 32768、显存比例 0.85、eager、最多 4 个序列、每请求 image=2 / video=0。session003 的 maxseq16 是独立的吞吐候选，尚未改为默认，也不能与 maxseq4 的校准混为同一次运行。

所有显式覆盖使用 `RECON2REASON_*` 前缀。即使先前 source 过 ASR 或 RoboBrain 的配置，本目录也会重设通用 `MODEL_PATH`、`PORT`、`PYTHON_BIN`、正式评测路径等变量，不继承另一题的值。模型 ID、框架和精度固定为本题身份。

## 4. 服务启动脚本

`run.sh` 读取 `env.sh` 后调用共享监督器 `scripts/recon2reason_runtime.py`。默认展开命令与 [历史成功命令](../../evidence/vllm-state-001.json) 的参数相同：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --served-model-name recon2reason-reasoning-4b \
  --host 127.0.0.1 --port 30000 --tensor-parallel-size 1 \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --enforce-eager --max-num-seqs 4 \
  --limit-mm-per-prompt '{"image":2,"video":0}'
```

默认加载超时 900 秒，总服务上限 4500 秒，包含加载时间；停止宽限为 30 秒。历史健康就绪约耗时 162.7 秒，这只是启动观测，不是 benchmark。监督器保存 `runtime/config-<run_id>.json`、`state.json`、`supervisor.log` 和 `server.log`。重复启动发现健康端口或本任务存活的监督器时会拒绝，不复用未知进程。失败、超时和中断会清理匹配进程组；`stop.sh` 核对 PID、Linux 启动时间和命令行后停止。

session003 maxseq4 的 [原始状态](../../evidence/recon-calibration-server-maxseq4-state-003.json) 显示实际监督上限3600秒、01:13:50健康就绪、01:33:07停止，其 [profile](../../evidence/recon-calibration-server-maxseq4-server-003.log) 为44.95 GiB／327,296 KV tokens。maxseq16 的 [原服务日志](../../evidence/session003/recon-original-runtime/server.log) 和 [补跑日志](../../evidence/session003/recon-final-runtime/server.log) 均已下载，独立确认 **44.93 GiB／327,168 KV tokens**、实际 chunked prefill 的 max_num_batched_tokens=2048、prefix caching 开启。原服务02:56:41停止；新服务03:01:32 ready、03:46:43停止。原runtime04与新runtime66源文件分别保全，不能将旧/新 run 或 maxseq4/16 的结果合并成一次实验。

| 可覆盖项 | 环境变量 |
|---|---|
| 模型路径、服务名、解释器 | `RECON2REASON_MODEL_PATH`、`RECON2REASON_SERVED_NAME`、`RECON2REASON_PYTHON_BIN` |
| 端口、TP、上下文、显存比例 | `RECON2REASON_PORT`、`RECON2REASON_TP`、`RECON2REASON_MAX_LEN`、`RECON2REASON_MEM_FRAC` |
| 加载、总运行和停止时限 | `RECON2REASON_LOAD_TIMEOUT`、`RECON2REASON_MAX_RUNTIME_SECONDS`、`RECON2REASON_STOP_GRACE_SECONDS` |
| 卡名、完整镜像地址、展示名 | `RECON2REASON_CARD`、`RECON2REASON_IMAGE`、`RECON2REASON_IMAGE_CANDIDATE_NAME` |
| 额外参数数组 | `RECON2REASON_EXTRA_ARGS_JSON` |
| 正式脚本、baseline、数据与图片规格 | `RECON2REASON_OFFICIAL_VERIFY_SCRIPT`、`RECON2REASON_OFFICIAL_BENCH_SCRIPT`、`RECON2REASON_OFFICIAL_BASELINE`、`RECON2REASON_OFFICIAL_DATA_DIR`、`RECON2REASON_IMAGE_RES` |

例如更换模型挂载位置并预览：

```bash
RECON2REASON_MODEL_PATH=/data/Recon2Reason-Reasoning-4B bash run.sh --dry-run
```

`RECON2REASON_EXTRA_ARGS_JSON` 会替换整个额外参数数组。若单独研究图模式，需显式保留 `--max-num-seqs 4` 和 image/video 限制，并作为另一组配置记录；不要把改过参数的输出归入历史成功运行。核心模型、网络、精度、TP、上下文、显存参数不能通过额外参数重复覆盖。

## 5. Smoke 测试

```bash
bash smoke.sh --image /data/test/red-square.png --image /data/test/blue-circle.png
```

脚本要求两张内容不同的真实本地图片，检查健康、模型名、单图、多图、多轮和 SSE 流式；保存图片 SHA-256 和原始回答，不记录图片 base64。流式必须有非空正文和 `[DONE]`，错误事件会失败。默认写本目录 `smoke.log`，不同轮次通过 `run_id` 区分。

历史红色方块、蓝色圆形测试的七项功能检查均通过，见 [JSONL 原始输出](../../evidence/vllm-smoke-001.jsonl) 和 [文本输出](../../evidence/vllm-smoke-001.txt)。

| 检查 | 历史结果 | 生成 token / 结束情况 |
|---|---|---|
| `/health` | 通过 | HTTP 成功 |
| `/v1/models` | 通过 | 目标模型可见 |
| 单图 | 通过 | 9 / stop |
| 多图 | 通过 | 40 / stop |
| 多轮首轮 | 通过 | 9 / stop |
| 多轮追问 | 通过 | 9 / stop |
| 图像流式 | 通过 | 正文、stop、`[DONE]` |

回答区分了首图红色与次图蓝色；位置追问只重复了物体描述，没有充分定位。这证明接口链路工作，不能当作空间推理正确性或正式精度验收。

session003 使用重新生成的红方块／蓝圆形图片于 01:17:16（北京时间）完成 [新七项功能](../../evidence/recon-functional-smoke-003.jsonl)，四个非流式回答与图像流式均以 stop 结束、未截断；这仍是开发功能检查。测试图与原始记录均做逐字节保全，来源和 SHA 见 [审计记录](../../evidence/recon-completed-audit-003.json)。

## 6. 精度测评

尚未取得本模型的赛事专属评测包。需要明确适用指标、固定样本与版本、GPU baseline 或逐样本参考、原版判分器、最大输出、思考开关及图像预处理。发布方 `verify_release.py` 是发布完整性检查，不计算 OCRBench、MMMU 或空间推理精度。`task.json` 的正式指标当前明确标为待确认，不预选 OCRBench 代替官方口径。

`verify.sh` 是外部正式脚本执行桥，不能独立完成测评。桥不假设 baseline 是 JSON 或含 `model/metrics`，也不会改官方文件来适配自身。实际运行会先保存脚本与 baseline 原始副本、SHA-256 和旧结果，再以当前解释器执行 `verify.py --selfcheck`；Shell 脚本使用 Bash。自检失败、超时，或脚本/baseline 内容改变时，禁止进入正式步骤。原始自检/评测日志、新旧产物分别保存在 `runtime/official-verify-<id>/`。

接入路径示例仅描述配置方式，不代表这些文件现已存在：

```bash
export RECON2REASON_OFFICIAL_VERIFY_SCRIPT=/已核实的正式材料/verify.py
export RECON2REASON_OFFICIAL_BASELINE=/已核实的正式材料/baselines/模型.json
export RECON2REASON_OFFICIAL_DATA_DIR=/已核实的正式材料/data
bash verify.sh
```

可在不开卡时先执行 `bash verify.sh --dry-run`，只读检查文件并输出命令、缺项和产物计划，不查服务、不执行脚本、不写结果。缺材料时返回 2，不生成 baseline 或 `verify_result.json`。

未知正式结果格式只记为 `executed_uninterpreted`、`official_acceptance=false`。执行桥返回 0 仅代表执行和新产物保全完成，不能作为精度 PASS；需拿到真实分题评分契约后再接入解释。结果若明确写了错误的 `MODEL_ID/CARD/FRAMEWORK/PRECISION`，或明确 `passed=false`，会失败。正式精度、自检和对应原始预测均尚未运行。

可用 `RECON2REASON_OFFICIAL_VERIFY_ARGS_JSON` 传评测参数，用 `RECON2REASON_OFFICIAL_SELFCHECK_ARGS_JSON` 单独指定自检参数（默认 `["--selfcheck"]`），全部进入记录；不经过 shell eval。默认等待自检 120 秒、外部脚本 7200 秒，分别由 `RECON2REASON_OFFICIAL_SELFCHECK_TIMEOUT_SECONDS` 与 `RECON2REASON_OFFICIAL_TIMEOUT_SECONDS` 控制。Linux 超时会清理该脚本的整个进程组。改变正式接口参数应依据真实材料，不借此省略自检。

## 7. 性能参考

T02 通用模板规定 4096 / 16384 输入 × 1024 输出 × 并发 1 / 8 / 64，共六档、各连续两轮，第二轮计分；每档请求数为并发的两倍且至少 8。需要保留两轮 JSONL、十二份档位日志和汇总表。具体图像分辨率、取图集与分题脚本仍待确认。

`bench.sh` 桥接外部正式脚本，本身没有实现六档客户端。`bash bench.sh --dry-run` 会输出六档两轮纯计划、每轮请求数、计分轮标记和当前缺项，共 608 请求；不连接服务、不执行、不写文件。缺正式脚本时仍输出计划并返回 2。

后续设置 `RECON2REASON_OFFICIAL_BENCH_SCRIPT`；如脚本要求其他参数，使用 `RECON2REASON_OFFICIAL_BENCH_ARGS_JSON`。benchmark 不硬性依赖 baseline、data 或 IMAGE_RES；若显式提供则检查文件，实际图像规格仍须按分题材料确认。默认保全两轮 JSONL、汇总和十二份档位日志，并将结果标为未解释，不因为文件齐全就宣称成绩或负载完成。模板外的真实输出文件名可通过 `RECON2REASON_OFFICIAL_{VERIFY,BENCH}_OUTPUTS_JSON` 提供相对路径数组，禁止路径逃逸。新旧产物分开保留，不覆盖历史证据。详细离线复现见 [Recon 离线就绪记录](../../docs/recon-offline-readiness.md)。

已完成以下开发校准；它们不是正式性能成绩，也没有与官方 SGLang 客户端建立同口径等价关系。当前镜像未安装 SGLang。

| 服务配置 / 客户端并发 | 请求 / 每条输出 | 原生 duration | 客户端 wall | 输出 token/s | mean TTFT ms | mean TPOT ms | mean E2EL ms |
|---|---|---|---|---|---|---|---|
| maxseq4 / c1 | 8 / 1024 | 380.702 s | 414.879 s | 21.518 | 812.167 | 45.724 | 47587.538 |
| maxseq16 / c8 | 16 / 1024 | 107.452 s | 141.348 s | 152.478 | 3510.727 | 48.847 | 53480.968 |

两次都是4096文本目标＋一张1024×1024随机图、第一轮、输出目标1024，失败数均为0、逐请求完整性均通过；客户端并发和服务 maxseq 同时变化，不能宣称这是单一改动带来的7倍优化。原始 [c1 JSON](../../evidence/recon-calibration-native-round1_4096_1024_c1-003.json)、[c8 JSON](../../evidence/recon-maxseq16-calibration-native-round1_4096_1024_c8-003.json) 和 [初始校准审计](../../evidence/recon-completed-audit-003.json) 保留其当时范围。

最终 [原矩阵](../../evidence/session003/recon-matrix/run.json) 按4000秒总预算完成10/12格；16384/c64第一轮在745.997秒时超时，第二轮未尝试。独立 [retry](../../evidence/session003/recon-matrix-retry/run.json) 使用新服务和2697秒总预算：第一轮128×1024完整，native duration1335.164秒、输出98.169 token/s、client wall1388.661秒；第二轮1307.366秒超时、无 native JSON。因此当前是11/12格、480个可核完整请求，最后128个计划请求的完整结果未知，不能记为0或丢弃失败。

全部命令使用 fixed seed=1，并在各自持续开启 prefix cache 的服务中连续执行；第二轮及跨并发可能复用缓存，独立retry又重启了服务。输入目标只描述随机文本，视觉/chat额外 token 与重编码差异未认证；这是一图1024×1024的 native vLLM开发负载，没有与正式 SGLang 客户端建立同口径等价关系。详见 [最终汇总](../../evidence/recon-final-results-summary-003.json) 和 [逐格表](../../docs/session003-recon-results.md)。

## 8. 注意事项与交付缺项

- 正式评测仍缺本模型固定数据、GPU 参考、判分器和适用口径；没有把认领回执或 digest 当作开工条件。
- 开发矩阵最后一格仍未完整核验；正式精度、同口径六档两轮性能、完整可取镜像地址/tag、仓库最终同步与下载核验、干净镜像一键复跑仍待完成。现有相对脚本目录必须完整包含在独立提交件内。
- 新启动应实际保存 `env.log`、`smoke.log`、完整服务日志与 `platform/` 证据；不要复制历史日志冒充干净复跑。八章报告最终数值必须与新产物一致。
- 项目监督器不控制平台计费。checkpoint003 已实际确认三个 Notebook 全部关机；A 于03:48:43左右手动确认、03:49观察停止，早于03:55兜底期限。可见余额493.5055555555卡时可能包含延迟结算，不是最终账单。checkpoint002 只保留为历史快照。
- 模型静态检查、Transformers 参考、vLLM 功能、正式精度和性能分别保存证据；目前只完成其中部分，不宣称整题验收通过。
- 不在任务文件、日志或报告保存账号、密码、token 或 cookie。当前文档更新是对已下载实际结果的本地审计，不执行新的平台操作；赛事正式提交尚未进行。
