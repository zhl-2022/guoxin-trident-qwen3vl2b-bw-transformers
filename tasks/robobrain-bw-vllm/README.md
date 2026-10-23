# RoboBrain2.5-8B-NV × BW × vLLM × BF16

## 1. 简介

session004 已回读当前 head72 compat 在 BW/native vLLM 的七项检查通过：其中 health/models 两项，实际生成请求五次，均完整结束；没有新增官方精度、性能矩阵或 GPU 数值对照。服务 `68f2ba449dbd41629b40cca27aa07553` 于 10:15:51 UTC ready、10:25:49 UTC 因 stop requested 进入 stopped，stop 输出确认匹配监督器/进程组；独立端口检查和平台计费终态仍需另核。原始输入/回答/脚本 seal 与状态见 [session004 技术报告](../../docs/report-004-robobrain-bw-vllm.md)。以下历史数值、IoU和81次Math调用仍只属于其各自 run。

本目录是 `BAAI/RoboBrain2.5-8B-NV` 的独立 T02 开发任务。默认 `run.sh` 启用受源码哈希约束的 head72 分块 Math 兼容候选，`run_compat.sh` 只是同一入口的别名；`run_baseline.sh` 保留显式原生框架对照。当前状态为 `head72_compat_gpu_numerical_and_functional_revalidated_official_pending`，`adaptation_ready=false`。baseline001 将红方块回答为人群、坐标不符及多图截断的历史失败完整保留。

历史版本 `cc6591b5c99bf3cef5090e88d8893e83e6db77e8cbc6bf97c24d42380a92c3af` 已通过 18 项 BW 数值对照；`robobrain-heldout-004.jsonl` 保存 640×448 红方块、512×512 蓝圆的七个新图片功能请求。实际 vision 长度 1120/1024，合成框 IoU 为 0.88396594/0.90244166，点选均在框内；81 次 Math 调用全部完成，服务已停止。该次启动显式使用 1024×1024 dummy profiling，默认 65536-patch profiling 的 Run003 因计算开销主动停止并保留日志。历史结果只绑定历史源码哈希。

2026-10-20 的当前版本修复了 `Hq=1 / Hkv>1` 头维广播时 score 预算低估的问题：按实际广播后的 batch/head 上界分块，并在不支持的几何关系下明确拒绝。当前源码 SHA256 记录于 [task.json](task.json) 的 `compatibility_candidate` 和[新 CPU 验证记录](../../evidence/rules/robobrain-vision-math-compat-cpu-002.json)。6 项真实 CPU 数值与行为测试通过。随后在第二个 BW Notebook 完成当前 SHA 的 24 项 GPU 数值对照（含 6 项广播预算检查），并完成七个新尺寸图片功能请求；两轮相同数值集的重复检查均通过，不能计作 48 个不同 case。最大绝对误差为 0.0009765625。服务 ready/stopped 的 run ID 一致，监督器确认清理完成。详见 [session003 实测与源码审计](../../docs/session003-robobrain-results.md)。正式大图、精度、性能和干净环境复跑尚未完成。

SCNet 模型静态预检已完成：750 个 BF16 张量、2 个分片，payload 为 17,534,247,392 字节；这不是实际服务显存。两分片的全文件 SHA256 及字节数已与固定 revision 的上游 LFS 元数据一致。原生 Transformers 5.5 + SDPA Math GPU 参考已保存；固定图对照将问题定位到视觉 SDPA 后端分派路径，尚未确认具体 fused kernel 根因，官方精度、性能或正式认领也未确认。

模型为标准 `Qwen3VLForConditionalGeneration`。架构已由 BAAI 公共元数据和实际下载配置双重确认；本任务使用自己的 config，不复用 Recon 结果。来源、精确四类 prompt 与输出协议记录在[准备记录](../../docs/robobrain-preparation.md)和本目录 [task.json](task.json)。[BAAI 模型卡](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV)、[FlagOpen 官方推理代码](https://github.com/FlagOpen/RoboBrain2.5/blob/main/inference.py)

保持根 `scripts/` 与 `tasks/` 的相对结构。所有 wrapper 使用共享监督器，并设置本任务的模型身份、端口和日志根。

## 2. 验证环境

下表综合实际原生 probe、模型静态报告、历史服务快照与第二实例 session003 的环境输出。当前版本在第二实例完成 GPU 数值与图片功能复验；完整环境字段分别保留各自采集时间，不能把历史 probe 伪装为本轮重新采集。

| 项目 | 当前证据 |
|---|---|
| 区域 / 资源组 | 华中一区【A区】，`113 / hx1hgbwnormal` |
| 实际设备 | 单卡，名称 `BW`，架构 `gfx936:sramecc+:xnack-`，设备报告显存 `68702699520` 字节 |
| 私有镜像展示名 | `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`；完整地址及实际 digest 仍未知 |
| Python | `3.10.12` |
| Torch | `2.10.0+das.opt1.dtk2604.20260325.g6b060a`；HIP `6.3.26093` |
| vLLM / HCU | `0.18.1+das.fa71803.dtk2604` / `0.18.1+das.dtk2604`；实际激活 `HCUPlatform` |
| Transformers / Triton | `5.5.0` / `3.6.0+gitc73250c4.staging` |
| torchvision | `0.25.0+das.opt1.dtk2604.torch2100.20260325.g860505` |
| flash-attn | `2.8.3+das.opt1.dtk2604.torch2100.20260330.g3f0061`；import 成功 |
| 实际模型目录 | `/root/private_data/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV` |
| 模型 config SHA256 | `5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661` |
| 实测服务地址 | `127.0.0.1:30001`，served name 为 `robobrain2-5-8b-nv` |
| session003 服务快照 | 2026-10-20 01:52:23 创建，01:56:24 ready，01:56:49 stopped（UTC+8）；创建到 ready 约 240.5 秒，仅初始化/健康就绪间隔 |
| session003 设备环境 | 实例 `2112237945233408001`，Torch 模块版本 `2.10.0`、HIP `6.3.26093`、vLLM `0.18.1+das.fa71803.dtk2604`；实际数值设备 `BW` |

静态证据位于根目录 `evidence/robobrain-preflight-001.json`、`evidence/robobrain-preflight-hashed-001.json`、`evidence/robobrain-model-config.json`。初次报告未 hash 权重，后续 hashed 报告为 `weights_hashed=true`；不能覆盖两轮的历史含义。第一/第二分片 SHA256 分别为 `ddcbb8b2c0edd0f7707b5c459c9981ef9c842293e5d520324dddc8c66996fba9`、`374a816964fc361fece15f7f8f661521a367181bdd3a5d6f75f59fa0b0c328b5`，大小分别为 9,914,019,048 和 7,620,320,504 字节，均与 [BAAI 发布的 LFS 元数据](https://huggingface.co/api/models/BAAI/RoboBrain2.5-8B-NV?blobs=true) revision `7ae187cc62e45cfa1c5904903a8f017f11b2037b` 完全一致。它们证明分片文件身份，不证明推理语义正确。默认卡型写实际 `BW`，不补造型号后缀。

环境证据为 `evidence/vllm-probe-001.json`、`evidence/vllm-native-001.txt`；baseline [服务快照](../../evidence/robobrain-vllm-state-001.json)与 Math002 [服务快照](../../evidence/robobrain-vllm-state-math-002.json)均为历史 ready 记录。另有 [baseline 停止状态](../../evidence/robobrain-vllm-stopped-state-001.json)和 [Math002 停止状态](../../evidence/robobrain-vllm-stopped-state-math-002.json)，两次均 `status=stopped`、`reason=stop requested`；这不证明平台 Notebook 停止计费或当前 GPU 空闲。

## 3. 适配优化方案

当前保留模型权重、结构、tokenizer 和官方模板，没有自动安装或升级框架。原生导入、模型服务及请求链路已经实际执行。[原生参考报告](../../evidence/robobrain-reference-gpu-001.json)使用 Transformers 5.5.0、BF16、SDPA Math 和正常 `AutoProcessor`，没有强制 image-only processor；红图、蓝图、双图三个 case 的生成片段能读出相应文字、识别红方块或开始圆形描述，但都达到 64 token 上限，不能称为完整生成或官方精度通过。官方 upstream 的完整 requirements 含 CUDA 组件，不能直接覆盖国产卡 DTK 栈。

随后仅对匹配的 head72、序列长度 384/768 调用强制 Math 的[固定形状诊断](../../scripts/robobrain_vision_math_diagnostic.py)恢复了同图同 prompt 的红方块与蓝圆识别，见[实际服务日志](../../evidence/robobrain-vllm-server-full-math-002.log)；worker PID 7046 共 54 次 eligible Math 调用完成。由此将异常定位到视觉 SDPA 后端分派路径，尚不能确认具体 fused kernel 根因。[实际平台与源码证据](../../evidence/robobrain-runtime-source-002.json)显示 HCU 的 `_enum=PlatformEnum.ROCM`、`is_rocm=true`、`is_out_of_tree=false`，原 wrapper 会执行 contiguous；“平台判断导致跳过 contiguous”的假设已排除。

当前[通用 head72 兼容候选](../../scripts/robobrain_vision_math_compat.py)包装实际 `vit_attn_wrappers.apply_sdpa`，每块 Q 保留当前图完整 K/V，原 `torch_sdpa_wrapper` 保持 `cu_seqlens` 分图及 ROCm contiguous。默认 query chunk 上限 128、单个 score 张量上限 16,777,216 元素；这不是总显存峰值限制。预算使用广播后的 batch/head 上界，覆盖 `Hq=1 / Hkv>1`，并拒绝不兼容的 head、batch、Q/K 特征维和 K/V 长度。候选在 Torch `2.10.0+cpu` 上的 [6 项真实测试](../../tests/test_robobrain_vision_math_compat.py)通过，覆盖 float32/BF16、GQA、头与 batch 广播、不同 Q/K 长度、`Vdim=40`、逐块预算和分图隔离。历史版本已完成 [18 项 GPU 数值比较](../../evidence/robobrain-math-compat-numerical-001.json)和[新图片请求](../../evidence/robobrain-heldout-004.jsonl)，当前版本以独立 [24 项 GPU 数值报告](../../evidence/robobrain-numerical-003.json)和[七次新图请求原文](../../evidence/robobrain-heldout-003.jsonl)完成复验，不沿用旧哈希的通过状态。六项新增广播检查逐块满足 score 元素预算；服务的 81 次 eligible Math 调用全部完成，1323 个块，日志最大单块 score 为 8,388,608 元素。

本目录 `env.sh` 使用 `ROBOBRAIN_*` 变量覆盖候选参数，并固定模型 ID；官方材料默认空，避免继承 Recon baseline。共享 `scripts/recon2reason_runtime.py` 虽保留旧文件名，其模型、served name、端口与任务路径可配置；本目录不会复制该实现。所有 runtime 文件、PID、锁、`env.log` 和 `smoke.log` 都属于本任务。

历史实测使用 BF16、TP=1、上下文 32768、显存比例 0.85、eager，并限制最大并行序列 4、每请求最多 2 图/0 视频。Math002 使用的固定形状 `VisionMathDiagnostic` 仅作历史对照；当前默认入口使用通用 `VisionMathCompat`，两者不能同时加载。该路径仍是兼容候选，不是已完成的性能优化。未启用 `trust_remote_code`。

## 4. 服务启动脚本

```bash
cd tasks/robobrain-bw-vllm
bash run.sh --dry-run
bash env_check.sh
# 核验当前资源与预算后，执行真实服务步骤：
bash run.sh
bash env_check.sh
# 结束本轮：
bash stop.sh
```

`run.sh --dry-run` 只打印配置，不加载模型、不启动服务。默认真实启动会启用通用兼容候选；`run_compat.sh` 与 `run.sh` 等价，不重写用户覆盖变量。模型路径如有变化，设置 `ROBOBRAIN_MODEL_PATH`；其他覆盖变量见 `env.sh`。默认 1024×1024 是历史及当前 session003 已实测的初始化 dummy profiling bound，不能据此保证更大真实图片的显存或性能；它也不是强制将真实输入缩放为 1024×1024 的图像处理策略。

如需开发其他 profiling bound，可显式设置以下变量；这些替代配置没有 BW 实测记录：

```bash
ROBOBRAIN_PROFILE_WIDTH=1536 ROBOBRAIN_PROFILE_HEIGHT=1024 \
  ROBOBRAIN_MAX_NUM_SEQS=4 bash run.sh --dry-run
```

`ROBOBRAIN_EXTRA_ARGS_JSON` 可添加普通框架参数；默认入口拒绝用它覆盖 worker extension、`--limit-mm-per-prompt` 或 `--max-num-seqs`，对应覆盖应通过上述独立变量完成。要运行原生框架对照，显式使用 `bash run_baseline.sh --dry-run`，确认后才去掉 `--dry-run`。原生对照入口默认额外参数与 baseline001 相同，不会调用兼容入口；两个入口共用任务 runtime，必须顺序运行并确认前一服务停止。

baseline001 的历史实际启动命令包含：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port 30001 --dtype bfloat16 \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --enforce-eager \
  --max-num-seqs 4 --limit-mm-per-prompt '{"image":2,"video":0}'
```

`env.sh` 默认加载上限 1800 秒、总时限 7200 秒；baseline001 的实际总时限为 3600 秒，包含加载，不能把默认值写成实际值。监督器保存 `runtime/config-*.json`、`state.json`、`supervisor.log`、`server.log`；该机制不会关闭 SCNet Notebook 或停止卡时计费。不同端口不保证单卡可以同时容纳两个模型，切换前应确认资源释放。session003 在实际 BW 环境执行 `env_check.sh`、24 项数值检查、默认 `run.sh`、七次新图请求和 `stop.sh`；本轮实际加载上限 600 秒、总时限 1800 秒。此次结果登记只审计已下载证据，不重新开卡。第二个 Notebook 保存环境并关机的事实来自根代理实际平台 UI 观察；服务停止记录本身不等于平台关机证明。

## 5. Smoke 测试

通用图文功能检查使用共享实现，包含健康、模型身份、单图、多图、多轮与流式：

```bash
bash smoke.sh --image /实际图片1.png --image /实际图片2.png
```

独立协议诊断使用一张图，四类请求分别为 general、grounding、pointing、trajectory，精确模板从本任务 JSON 读取：

```bash
bash protocol_smoke.sh --image /实际单图.png \
  --target "the object actually visible in this image" \
  --trajectory-instruction "an image-related manipulation instruction" \
  --max-tokens 128
```

建议先加 `--dry-run` 检查 prompt 和图片指纹；诊断 token 上限仅允许 128 或 256，temperature 为 0，不代表正式 baseline。每次新建 `protocol-smoke-*.jsonl`；显式 `--log` 也拒绝覆盖已有文件。日志保存图片/脚本/task hash、有限长度原始回答、reasoning、finish_reason 和语法提示，不保存图片 base64。

独立协议脚本把 reasoning-only、`finish_reason=length`、空答案及 HTTP 错误视为未完成；其坐标正则仅表示出现语法，不检查范围、深度单位或正确性。共享通用 smoke 的 `passed` 则主要表示传输/非空输出，须额外读取 `answer_truncated` 和 finish reason。不能混淆这两种通过范围。

baseline001 的历史结果保存在根目录 [通用 smoke](../../evidence/robobrain-vllm-smoke-001.jsonl) 和 [单图协议 smoke](../../evidence/robobrain-protocol-smoke-001.jsonl)，保持失败原文：

| 检查层 | 原始结果 | 当前解释 |
|---|---|---|
| `/health`、模型身份、单图/多图/多轮/流式 | 共 7 项 `passed=true`，流式含 `[DONE]` | 传输与非空响应通过；没有语义验收 |
| 多图回答 | `finish_reason=length`，输出 256 token，`answer_truncated=true` | 原始 passed 保留，回答未完整结束 |
| 四个单图协议 | general / grounding / pointing / trajectory 均 `stop`，4 项功能协议通过 | 后三项只观察到坐标语法 |
| 红色方块图片，SHA256 `46b3dc7b5e2de94b4b35f1ed4a3947c0575186c5fe852c6d5939c60a08a9f5b7` | 单图中文称“一组穿着不同服装的男性模特”；general 英文称一群站成排的人 | 与已知功能图内容明显矛盾，语义阻断 |
| 指定 `the red square` | grounding 为 `[10, 10, 1000, 1000]`，pointing 为 `[(964, 966)]` | 人工核图发现定位不符；没有计算官方定位指标 |
| trajectory | 输出 3 个带深度点的 tuple | 只有格式，缺参考轨迹和深度单位，不判正确 |

原始日志不修改、不把 `passed=true` 重新包装为适配成功。协议日志里的 task hash 对应当时配置；当前文件后续更新了状态和证据，因此不能要求当前 task hash 与历史值相同。

Math002 对照结果另存于[通用 smoke](../../evidence/robobrain-vllm-smoke-math-002.jsonl)和[单图协议 smoke](../../evidence/robobrain-protocol-smoke-math-002.jsonl)：

| 对照检查 | 已保存输出 | 验证范围 |
|---|---|---|
| 单图与双图 | 单图回答红方块；双图回答第一图红方块、第二图蓝圆，均 `finish_reason=stop` 且未截断 | 固定功能图识别恢复；没有新图片集或官方正确率 |
| 通用传输 | 7 项 `passed=true`，流式包含 `[DONE]`；多轮 follow-up 也完成 | 传输和非空响应；多轮位置措辞未获得语义真值评分 |
| general | 英文识别红方块及图中文字，`stop` | 与该功能图主要对象一致，不代表所有空间描述都经评测 |
| grounding / pointing | `[60, 233, 350, 780]` / `[(237, 516)]`，均 `stop` | 对比 baseline001 的错误输出已明显恢复；未计算官方坐标指标 |
| trajectory | `[(225, 10, 0.439), (230, 200, 0.529), (250, 250, 0.610)]` | 仅保留原始格式结果；没有参考轨迹或深度真值 |

固定形状 Math002、历史通用分块 Math004 与当前 session003 使用各自证据及源码身份。session003 的七次请求均非空、`finish_reason=stop`、无截断；红/蓝图 grounding 的合成框 IoU 分别为 0.88396594 / 0.90244166，两次点选均在已知框内，双图回答识别红方块与蓝圆。没有机器人轨迹或深度真值，也不是正式语义验收。[图片 manifest](../../evidence/robobrain-heldout-manifest-003.json) 的 `processor_grid_actually_verified=false` 保留原样；1120/1024 是独立从服务 Math 日志观察到的实际 vision 长度，不能改写为 processor grid 已直接验证。

## 6. 精度测评

官方图集、GPU baseline、坐标/动作解析器、固定解码参数及评测范围尚缺。本任务不能未经分题确认就把 VLA 解析后一致率 98% 的通用规则作为正式适用要求，也不能自动套用 OCRBench / MMMU。

取得本模型原版官方材料后，设置独立变量：

```bash
export ROBOBRAIN_OFFICIAL_VERIFY_SCRIPT=/实际官方脚本路径/verify.py
export ROBOBRAIN_OFFICIAL_BASELINE=/本模型官方baseline.json
export ROBOBRAIN_OFFICIAL_DATA_DIR=/本模型官方数据目录
bash verify.sh
```

包装器会校验 baseline 模型身份并记录哈希，缺材料时拒绝启动。正式验收仍需官方脚本自检与完整输出。当前 baseline 为 null、metric 待确认、`max_drop=0` 为禁用阶段脚手架。baseline 失败、原生参考片段、Math002 恢复和历史通用候选的合成图检查都不是官方分数。当前源码的 GPU 数值和七次合成图片功能复验已完成，仍需正式材料验证；`adaptation_ready=false`，官方精度与性能状态保持 false。

## 7. 性能参考

尚未取得 RoboBrain 官方图像分辨率、负载与完整 benchmark 材料，因此没有吞吐、TTFT、TPOT、端到端延迟、显存峰值或提升比例。模型 payload、包 import 或静态预检耗时均不能作为服务性能。

取得适用的官方材料并满足精度门槛后，使用独立入口：

```bash
export ROBOBRAIN_OFFICIAL_BENCH_SCRIPT=/实际官方脚本路径/bench.sh
export ROBOBRAIN_OFFICIAL_BASELINE=/本模型官方baseline.json
export ROBOBRAIN_OFFICIAL_DATA_DIR=/本模型官方数据目录
export ROBOBRAIN_IMAGE_RES=官方规定分辨率
bash bench.sh
```

通用规则的六档为输入 4096 / 16384、输出 1024、并发 1 / 8 / 64，各两轮、第二轮计分；是否适用于本具身分题及具体图片要求仍需确认。不得把诊断脚本 128/256 token 请求作为正式负载。

## 8. 注意事项

- 当前为 `head72_compat_gpu_numerical_and_functional_revalidated_official_pending`，`adaptation_ready=false`；当前源码已完成 BW 数值与合成图功能验证，官方精度与性能均未验证。
- 本模型静态/文件 hash、基础服务、原生参考和固定图 Math002 已有独立证据；baseline001 历史错误仍保留，不能被后续成功覆盖。
- 原生 Transformers 5.5 + BF16 Math 的正常 AutoProcessor 参考已保存，三次均达到 64 token 上限；不能当完整生成或官方通过。
- 异常已定位到视觉 SDPA 后端分派路径，尚未确认具体 fused kernel 根因；HCU 具有 ROCm 身份，跳过 contiguous 的假设已排除。
- 历史 `cc6591...` 版本保留原 18 项 BW 数值与七个新图请求；修复广播预算的当前 `083dc53...` 版本已通过 6 项 CPU 测试、24 项 BW 数值及七个图片请求，证据分开登记。
- 默认 1024×1024 profiling 已由当前源码复验；更大输入、其他 profiling bound、正式精度和性能不能由此推断。
- 分题材料、完整镜像地址及 tag、正式结果尚缺；可记录 digest 增强复现，但不能自加为赛事强制项，也不等待独立认领回执才能开发。
- 本任务预算字段来自历史 checkpoint，不是 session003 后的实时余额；第二实例已由根代理保存环境并关机，再次启动前应核对实时余额。
- source 中 x/y 绘图尺度为 0–1000，深度 d 的单位未在包装器规定；不要补造真实机械臂坐标、深度单位或动作正确性。
- 保留失败与截断日志，不修剪成通过；不修改官方 baseline、数据或模型输出以规避检查。
- 本目录没有账号、密码、key 或 cookie，不自动执行 upstream 代码或提交成果。
