# RoboBrain2.5-8B-NV 独立任务准备

核对日期：2026-10-19。组合为 `BAAI/RoboBrain2.5-8B-NV × BW × vLLM × BF16`，任务目录为 `tasks/robobrain-bw-vllm/`。当前状态 `general_head72_math_compat_functional_verified_official_pending`，`adaptation_ready=false`。初次服务把红方块描述为人群、定位不符的结果保留为历史失败；原生 Transformers 参考、固定图 Math 诊断和通用分块 Math 候选均恢复了测试图形识别。通用候选通过 4 项真实 CPU 测试、18 项 BW 数值对照以及两张新尺寸图片的七个功能请求。问题与视觉 SDPA 分派路径有关，具体 fused kernel 根因、官方精度与性能尚未验证。八章记录见[任务 README](../tasks/robobrain-bw-vllm/README.md)。

最终 Run004 使用 1024×1024 的初始化 dummy profiling；真实图片 640×448、512×512 的视觉 patch 长度实际为 1120、1024，超出旧 fixture guard。红方块/蓝圆及双图比较正确，合成框 IoU 约 0.884/0.902，点选均在目标框内。81 次 math 调用全部完成，服务随后停止。见 `robobrain-heldout-004.jsonl`、`robobrain-heldout-manifest-004.json`、`robobrain-vllm-stopped-state-compat-004.json`。大图和正式性能仍待验证；下文记录候选建立过程中的 pending 状态，以此段最终实测结果为准。

## 1. 已确认的来源与尚缺的运行证据

| 项目 | 已确认内容 | 来源或限制 |
|---|---|---|
| 模型身份 | `BAAI/RoboBrain2.5-8B-NV` | [BAAI 官方模型卡](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV) |
| 架构 | `Qwen3VLForConditionalGeneration`，`model_type=qwen3_vl` | [Hugging Face 公共模型元数据](https://huggingface.co/api/models/BAAI/RoboBrain2.5-8B-NV)；不是根据 Recon 模型推测 |
| 来源版本 | HF revision `7ae187cc62e45cfa1c5904903a8f017f11b2037b` | 公开元数据在本轮读取；不能据此证明 SCNet 克隆内容是同一 revision |
| 权重元数据 | BF16，8,767,123,696 参数 | 该行来自公开 safetensors 元数据；实际分片核验见本模型静态预检行，不表示已验证实际显存或服务能力 |
| 许可证与访问 | 模型卡为 Apache-2.0；HF 文件需要访问授权 | 本轮只读取公开模型卡与元数据，没有接受访问条款或获取受限文件 |
| 官方推理包装器 | [FlagOpen/RoboBrain2.5/inference.py](https://github.com/FlagOpen/RoboBrain2.5/blob/main/inference.py) | 只作为文本及 AST 静态解析，没有 import 或执行 |
| 包装器内容指纹 | SHA256 `69511729e252c2cc7b8ca7c24a790099139502b479a1299a66ae2260663affb5` | 读取 `main` 内容的字节指纹；没有声称已经获取 Git commit，`task.json` 中 commit 保持 null |
| SCNet 模型克隆 | clone ID `2112187999273320450`，cluster `20091`；已完成 | 以 `evidence/robobrain-preflight-001.json` 实际文件和 header 验证为证 |
| 本模型静态预检 | 750 个 BF16 张量、2 个分片，payload `17534247392` 字节 | header/index 一致；两分片全文件 SHA256 和大小与 [BAAI LFS 元数据](https://huggingface.co/api/models/BAAI/RoboBrain2.5-8B-NV?blobs=true) 固定 revision 一致，不代表推理正确 |
| 本模型 config SHA256 | `5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661` | 远端静态报告与本地 `evidence/robobrain-model-config.json` 一致 |
| 官方任务与验收 | 尚未确认该组合认领及分题评测范围 | baseline、evaluator、图集、动作/坐标解析器和镜像 digest 都未获得 |

SCNet UI 报告的存储路径为 `/public/home/ac3p4mowhz/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV`。容器内 `/root/private_data/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV` 已由本模型静态报告验证；换实例后仍须重新核对挂载。

`task.json` 的 `model.config_path` 已填写本模型的 `evidence/robobrain-model-config.json`，没有复用 Recon 配置。顶层 dtype 在该配置中缺省；BF16 来源是 text config、实际分片 header 及公开权重元数据，不应把顶层 null 称为精度错误。

同实例的 `evidence/vllm-probe-001.json` 与 `evidence/vllm-native-001.txt` 显示原生 Torch / torchvision / flash-attn / vLLM import 成功、HCUPlatform 激活，设备实际报告 `BW`、`gfx936:sramecc+:xnack-`。有效包版本为 Torch `2.10.0+das.opt1.dtk2604.20260325.g6b060a`、vLLM `0.18.1+das.fa71803.dtk2604`、vLLM-HCU `0.18.1+das.dtk2604`、Transformers `5.5.0`、Triton `3.6.0+gitc73250c4.staging`。初次服务曾出现输出语义异常，后续 Math 诊断的固定图恢复见下文；原生 import、HTTP 层或固定图通过都不能推广成通用适配完成。卡型按观察值写 `BW`，不补造后缀。

### 实际对照结果与当前边界

`evidence/robobrain-vllm-state-001.json` 记录服务北京时间 22:40:34 创建、22:42:24 ready，约 109.9 秒；这是历史健康就绪快照，不是实时状态或模型性能。实际命令为 BF16/TP1、上下文 32768、eager、`--max-num-seqs 4`、每请求最多 2 图/0 视频，实际总时限为 3600 秒，和 env 默认 7200 秒不同。

通用 smoke 日志 `evidence/robobrain-vllm-smoke-001.jsonl` 保存 7 项传输功能 `passed=true`，但 `multiple_images` 的 finish reason 为 `length`，输出 256 token 且标记截断。协议日志 `evidence/robobrain-protocol-smoke-001.jsonl` 的 general/grounding/pointing/trajectory 均完成并观察到需要的语法，但同一红方块功能图（SHA256 `46b3dc7b5e2de94b4b35f1ed4a3947c0575186c5fe852c6d5939c60a08a9f5b7`）被描述为一群站成排的人；grounding 为 `[10, 10, 1000, 1000]`、pointing 为 `[(964, 966)]`，人工核图发现不符。trajectory 只有三点格式，没有参考轨迹或深度真值可判正确。

所有原始 passed 与回答保持不变，明确分别记录“传输通过”和“语义异常”。不能据此宣布官方一致率、定位精度或框架适配完成。后续证据如下：

| 对照层 | 已保存证据 | 观察与限制 |
|---|---|---|
| 原生模型参考 | [Transformers GPU 参考](../evidence/robobrain-reference-gpu-001.json) | Transformers 5.5.0、BF16、SDPA Math、正常 `AutoProcessor`，没有强制 image-only processor。三个 case 分别为红图、蓝图和双图；生成片段读出相应图中文字，红图识别红方块，蓝图读出 `BLUE` 并开始圆形描述，双图开头正确识别红方块。三次均生成到 64 token 上限，不能称为完整生成或正式精度通过。 |
| vLLM 固定图 Math 诊断 | [服务日志](../evidence/robobrain-vllm-server-full-math-002.log)、[通用 smoke](../evidence/robobrain-vllm-smoke-math-002.jsonl)、[协议 smoke](../evidence/robobrain-protocol-smoke-math-002.jsonl) | 同图、同 prompt 下，单图回答红方块，双图回答红方块与蓝圆且 `finish_reason=stop`；grounding 为 `[60, 233, 350, 780]`，pointing 为 `[(237, 516)]`。这些是功能图观察，没有官方坐标指标；trajectory 仍只有格式证据。 |
| 诊断覆盖 | [固定形状诊断脚本](../scripts/robobrain_vision_math_diagnostic.py)与上述服务日志 | 仅为 head72、序列长度 384/768 的匹配调用强制 Math；实际 worker PID 7046 记录 54 次 eligible、54 次 completed。不能把固定形状诊断推广为所有图片均已修复。 |
| 服务结束 | [baseline 停止状态](../evidence/robobrain-vllm-stopped-state-001.json)、[Math 诊断停止状态](../evidence/robobrain-vllm-stopped-state-math-002.json) | 两次均保存 `status=stopped`、`reason=stop requested`；它们证明对应历史服务结束，不是当前 GPU 空闲或平台停止计费的实时证明。 |

[捕获的实际运行源码与平台](../evidence/robobrain-runtime-source-002.json)显示 HCU 的 `_enum=PlatformEnum.ROCM`、`is_rocm=true`、`is_out_of_tree=false`，原 `torch_sdpa_wrapper` 会执行 contiguous 转换。因此“由于平台未识别为 ROCm 而跳过 contiguous”的假设已排除。只改变视觉 Math 分派后固定图输出恢复，支持将异常定位到该 SDPA 后端分派路径；日志中的 FlashAttention 接口提示不足以确认具体 fused kernel 或底层算子的根因。

新的[通用 head72 兼容候选](../scripts/robobrain_vision_math_compat.py)直接包装实际 `vit_attn_wrappers.apply_sdpa`，只对 head72 使用 Math；沿 query 序列分块，每块使用当前图的完整 K/V，原 wrapper 继续负责 `cu_seqlens` 分图及 ROCm contiguous。默认 query chunk 上限 128、单个 score 张量上限 16,777,216 元素，后者不等于总显存峰值。该候选已冻结供验证；[CPU 数值测试](../tests/test_robobrain_vision_math_compat.py)在 Torch `2.10.0+cpu` 上 4 项通过，覆盖 GQA、不同 Q/K 长度、`Vdim=40`、分块预算和分图隔离。[GPU 数值验证器](../scripts/validate_robobrain_vision_math_compat.py)及未参与诊断的图片验证仍 pending；`adaptation_ready=false`，正式精度与性能状态均为 false。

## 2. 四种任务的输入与输出协议

官方包装器使用的通用任务名是 `general`。`task.json` 的 `upstream_protocol.prompt_templates` 保存从官方 AST 提取的完整原文模板，以 `{text}` 表示调用方文本；不是中文改写或自行设计的提示词。对应实际下载源码行号为 pointing 52、trajectory 55、grounding 58，网页格式化行号可能不同。

| task | 图片约束 | prompt 处理 | 原始输出与绘图解析 |
|---|---|---|---|
| `general` | 包装器允许单图和多图 | 原文透传，不添加任务后缀 | 自由文本 |
| `grounding` | 严格一张图 | 用目标描述构造 bbox 指令 | 绘图器匹配 `[x1, y1, x2, y2]`，可匹配多框 |
| `pointing` | 严格一张图 | 在目标描述后追加 2D 点格式要求 | 模板表示 `[(x, y)]`；绘图器匹配圆括号中的整数点，可匹配多点 |
| `trajectory` | 严格一张图 | 用动作描述构造 3D end-effector-centric waypoint 指令 | 模板表示 `[(x1, y1, d1), (x2, y2, d2), ...]`；绘图器匹配整数 x/y 和有小数点的有符号深度 |

以上格式来自[官方推理代码的任务分支和绘图分支](https://github.com/FlagOpen/RoboBrain2.5/blob/main/inference.py)。包装器统一返回 `{"answer": "解码文本"}`，没有返回已校验的坐标 JSON。Python tuple 风格的回答不能直接交给 `json.loads`；正式解析需要使用官方分题解析器，不能用可执行的 `eval`。

绘图函数按 0–1000 的相对尺度解释 x/y，以图像宽高换算像素后裁到边界。该裁边只服务可视化，不能把越界预测修正后拿去冒充原始正确输出。源码只把 d 称为 depth，没有在该接口中规定单位、相机参数、坐标系变换或真实机械臂动作范围；这些字段保持未知。图片解析成功也不能证明机器人轨迹正确。

包装器先放全部 image content，再放 text content，使用模型自己的 chat template，并通过 Qwen 视觉处理工具生成输入。它会截掉输入 token 后解码新生成部分。默认 `max_new_tokens=768`、采样打开、温度 0.7；模型卡的部分结构化示例显式关闭采样。共享 smoke 使用温度 0 做功能检查，不能将它称为与官方参考逐 token 等价。正式解码参数、思考开关和 token 上限以该模型官方 baseline 为准。[模型卡 Quickstart](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV#-quickstart)

## 3. 与现有 Recon 任务的隔离

共享脚本的文件名虽然仍含 `recon2reason`，实际 runtime 会读取环境中的模型 ID、路径、served name、框架和端口；OpenAI smoke 读取 served name 与 base URL。因此本任务复用共享 `scripts/recon2reason_runtime.py` 和 `scripts/recon2reason_smoke.py`，不复制或修改它们。

| 配置 | 本任务默认值 | 隔离范围 |
|---|---|---|
| 模型 ID | `BAAI/RoboBrain2.5-8B-NV` | `env.sh` 固定任务身份 |
| served name | `robobrain2-5-8b-nv` | `/v1/models` 及请求中使用独立身份 |
| 地址 | `127.0.0.1:30001` | 与 Recon 默认 30000 区分 |
| runtime / PID / 锁 | `tasks/robobrain-bw-vllm/runtime/` | 每个任务各自监督和停止 |
| 功能 / 环境日志 | 本目录 `smoke.log`、`env.log` | 不写入 Recon 的日志 |
| 官方材料 | `ROBOBRAIN_OFFICIAL_*` | 默认空，避免继承 Recon baseline、evaluator 或 data |
| 覆盖方式 | `ROBOBRAIN_MODEL_PATH`、`ROBOBRAIN_PORT` 等 | 先前 source 的通用 `MODEL_ID` / `SERVED_NAME` / `PORT` 不会串入本任务 |

端口不同不代表一张卡能同时容纳两个模型。首次执行应在 Recon 资源已经停止或确认具备独立资源后进行；本轮没有安排并行 GPU 运行。

## 4. 后续执行顺序

模型静态预检、原生参考、baseline 服务和固定图 Math 诊断已执行，后两次服务均有停止证据。通用候选的 GPU 数值与新图片语义仍待验证。以下命令用于复查基础流程；它们未启用兼容候选，先由 root 核对当前进程与资源并选择实验配置。不要用 Recon 的预检或参考输出填充本任务结果。

```bash
cd /实际项目路径
export ROBOBRAIN_MODEL_PATH=/已核验的RoboBrain模型目录

# 只读取版本、索引、header和小型配置；没有权重加载或torch import。
# --output 必须使用新的证据文件名，避免覆盖已有结果。
python scripts/scnet_preflight.py \
  --model "$ROBOBRAIN_MODEL_PATH" \
  --output evidence/robobrain-scnet-preflight-001.json

cd tasks/robobrain-bw-vllm
bash run.sh --dry-run
bash env_check.sh
# 在确认平台到期停止、余额、有效卡型和模型内容后才启动。
bash run.sh
bash smoke.sh --image /实际图片1.png --image /实际图片2.png
bash protocol_smoke.sh --image /实际单图.png \
  --target "图中实际存在的目标描述" \
  --trajectory-instruction "与该图对应的动作描述" --max-tokens 128
bash stop.sh
```

env 初始默认为 BF16、TP=1、上下文 32768、显存比例 0.85、eager、加载上限 1800 秒、总时限 7200 秒；baseline 实际服务把总时限覆盖为 3600 秒并加入上述序列/图像数量限制，Math 诊断还显式加载了 `robobrain_vision_math_diagnostic.VisionMathDiagnostic`。固定图诊断有效，通用候选尚不能称为已完成适配或优化。实际镜像完整地址和 digest 仍缺。进程总时限不会关闭 Notebook 或停止平台计费，必须另外核验平台到期停止。

`run.sh --dry-run` 只展开命令，不检查远端模型目录。`env_check.sh` 不安装包。官方 [requirements.txt](https://github.com/FlagOpen/RoboBrain2.5/blob/main/requirements.txt) 包含 CUDA 依赖及不同版本的 torch / vLLM，不能整份安装进国产卡 DTK 镜像；本任务没有新增依赖安装步骤。

本地计划检查：

```powershell
$env:PYTHONPATH = "src"
python -m guoxin inspect-config --task tasks/robobrain-bw-vllm/task.json
python -m guoxin plan --task tasks/robobrain-bw-vllm/task.json
```

当前 plan 应返回 `ready=false`。`accuracy.metric` 仍为待官方材料确认，baseline 为 null；`max_drop=0` 只是禁用状态下的保守脚手架，不是官方阈值。不得据此评判 RoboBrain 精度。`required_metrics` 与 `upstream_protocol` 是描述数据，现有 CLI 不自动执行它们。

独立 `scripts/robobrain_protocol_smoke.py` 从 task config 读取原文模板，分别发送 general、grounding、pointing、trajectory 的四个单图请求；temperature 为 0，诊断输出上限为 128 或 256 token。它先检查 `/v1/models`，只向 loopback 服务发送图像，新建唯一 `protocol-smoke-*.jsonl`，记录图片/脚本/task hash、原始回答、reasoning 字段、finish_reason 和截断信息。HTTP 或 JSON 错误后停止追加请求；token limit 和 reasoning-only 不算完成。

结构化正则只给出与官方绘图器类似的语法提示，不检查坐标范围、深度单位或答案正确性；即使功能协议检查通过也不是官方一致率。脚本不使用 `eval`、不执行 upstream 模型代码、不在日志写图片 base64。`--dry-run` 可先保存四个请求的 prompt 和图片指纹而不发 HTTP。

## 5. 尚待 root 集成的内容

- [x] 完成克隆、实际路径与模型配置 SHA 核验，保存专属静态报告；两分片全文件 hash/大小与 BAAI 固定 revision 的 LFS 元数据一致，推理正确性仍未证明。
- [ ] 确认该组合是否可认领及正式适用的 T02 / 具身指标，取得官方 GPU 输出、图集、解析器和固定解码配置。VLA 解析后一致率 98% 的通用规则不能未经分题确认就套用。
- [x] 在实际镜像运行本模型 serving、7 项通用功能与 4 类协议 smoke，原始结果已保存；这只证明传输和语法层。
- [x] 保留 baseline001 红方块被描述为人群、错误定位以及多图截断的原始历史记录。
- [x] 保存原生 Transformers 5.5 + BF16 SDPA Math、正常 AutoProcessor 的三 case GPU 参考；三次均达到 64 token 上限，不是完整生成或官方评测。
- [x] 保存 vLLM Math002 同图同 prompt 的服务、通用 smoke、协议 smoke 和停止证据；固定图识别与坐标输出恢复，54 次匹配 Math 调用完成，定位到 SDPA 后端分派路径，尚无具体 fused kernel 根因。
- [x] 准备并冻结通用 head72 分块兼容候选，Torch `2.10.0+cpu` 的 4 项数值测试通过。
- [ ] 完成通用候选 GPU 数值比较和未参与诊断的图片验证；当前状态为 `fixture_math_workaround_verified_general_compat_pending`，`adaptation_ready=false`，官方精度与性能均未验证。
- [ ] 正式评测入口使用本模型独有的 `ROBOBRAIN_OFFICIAL_VERIFY_SCRIPT` / `ROBOBRAIN_OFFICIAL_BENCH_SCRIPT` / `ROBOBRAIN_OFFICIAL_BASELINE` / `ROBOBRAIN_OFFICIAL_DATA_DIR`。结果须通过共享包装器的模型身份检查。
- [ ] 刷新余额与平台到期停止；本 task 余额 `496.9866666666` 来自既有平台 checkpoint，后续 Recon 会话会继续消耗，不能当实时额度。

本文件是开发准备记录，不是提交报告。来源架构、静态预检、参考推理、vLLM serving、结构化输出格式、官方一致率和正式性能必须分别保留证据。
