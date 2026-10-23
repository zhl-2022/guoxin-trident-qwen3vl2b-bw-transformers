# Qwen3-ASR-1.7B × BW × vLLM × BF16

## 1. 简介

session004 已回读新隔离 venv 准备、中英短 WAV 和完整文件输入后 SSE 原件：两个转写请求与一个 SSE 请求功能通过，12 个 protected native 包版本/位置不变，七个锁定 CPU 包仅安装到任务 venv。服务 run `1a02b2e4e0c944cabdb195dd54dc8348` 的终态为 `expired`（900 秒预算），后续 stop 未发送信号；独立端口/后代清理与 Notebook 关机另核。正式 WER/CER、性能和镜像验收仍未完成。见 [session004 技术报告](../../docs/report-004-qwen3-asr-bw-vllm.md)。以下 session003 / 2026-10-19 记录保留为历史，不冒充本轮新结果。

本目录适配原始 `Qwen/Qwen3-ASR-1.7B` checkpoint，属于 T04。2026-10-20 的 session003 已在另一 BW 单卡实例新建任务隔离 venv、安装锁定依赖，并完成发布方中英示例派生的三条 WAV 转写和整文件上传后的 SSE 输出；任务服务随后停止并通过进程/端口清理核验。长样本为 39.260375 秒人工中英拼接，不能称自然连续长语音。正式 WER/CER、性能、最终交付镜像的干净环境验收仍未完成，`formal_ready=false`。完整结果见 [session003 ASR 实证报告](../../docs/session003-asr-results.md)。

2026-10-19 的两条自有文本合成音转写历史仍保留：中文 8.395 秒、英文 6.17 秒，返回与本地原稿一致，参考原稿没有发送给模型。本轮发布方音频没有已核验人工参考，未沿用模型返回作为答案或计算 WER/CER。

原始 `asr-preflight-001.json` 的 tokenizer.json 缺失失败原样保留；后续 `asr-preflight-002.json` 已按实际 legacy BPE 布局校验通过：vocab.json、merges.txt、tokenizer_config.json，无需伪造 tokenizer.json。两权重分片的 SHA256 与发布方 revision `7278e1e70fe206f11671096ffdd38061171dd6e5` 一致，共 708 个 BF16 张量、4,698,434,816 字节 payload。模型名称中的 1.7B 不代表全模型参数数，API 统计为 2,349,217,408 个 BF16 参数。

上游入口：[模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)、[vLLM v0.18.1 ASR 实现](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/models/qwen3_asr.py)。光合已有同类部署与配置初始化顺序修复，应引用已有工作，不能把该修复视为本项目独创；公开材料的逐项核查见根目录 [提交就绪核查](../../docs/submission-readiness-audit-20261020.md)。

## 2. 验证环境

| 项目 | 已观察结果 |
|---|---|
| 平台 | SCNet 华中一区【A区】，`113 / hx1hgbwnormal`，BW × 1，`gfx936:sramecc+:xnack-` |
| 基础镜像展示名 | `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`；完整 registry/repo:tag 尚未取得，不能用展示名代替 |
| 原生 Torch | `2.10.0+das.opt1.dtk2604.20260325.g6b060a` |
| 原生 vLLM / HCU | `0.18.1+das.fa71803.dtk2604` / `0.18.1+das.dtk2604` |
| Transformers / NumPy / SciPy | `5.5.0` / `1.25.0` / `1.15.3` |
| 模型实际目录 | `/root/private_data/SothisAI/model/Aihub/Qwen3-ASR-1.7B/main/Qwen3-ASR-1.7B` |
| 模型配置 SHA256 | `2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f` |
| 实际服务 | `127.0.0.1:30002`，`qwen3-asr-1.7b`，BF16、TP=1、MAX_LEN=32768、显存比例 0.85、eager、max-num-seqs=4 |
| 静态文本位置上限 | 嵌套 `thinker_config.text_config.max_position_embeddings=65536`；32768 已初始化，长音频尚未验证 |

当前 native ASR 路径使用 librosa + soundfile，不要求为短 PCM WAV 安装 torchaudio / ffmpeg / PyAV。session003 的新任务 venv 实际安装了 7 个锁定 CPU 包，12 个受保护原生包的版本与路径与 native-before 完全一致，WAV 解码、重采样和音频依赖闭包检查通过，见 [prepare 结果](../../evidence/asr-clean-prepare-003.json)、[安装记录](../../evidence/asr-audio-pip-install-003.log)和[原生前状态](../../evidence/asr-native-before-003.json)。历史 002 的准备记录没有改写。

`native-stack.json` 固定实际原生版本；`requirements-extra.txt` 固定审核过的 7 个官方 PyPI wheel URL 与 SHA256。不安装 `qwen-asr[vllm]`，其 extra 会引入另一套框架版本，也不切换到 `-hf` 模型变体。

## 3. 适配优化方案

当前 Transformers 5.5 在父类初始化期间提前读取 thinker 子配置，原生 ASR 类在之后才设置该字段，因此原始启动失败。项目局部 `vllm.general_plugins` 插件仅提前构造同值 child，再执行完整原生构造函数；不会关闭 token-ID 校验、替换权重、改 tokenizer 或改 site-packages。插件同时校验 Transformers 版本和原生 ASR 配置源码 SHA256；发生漂移时停止并要求复核。父进程及 EngineCore 的实际安装事件保存在第二轮服务日志。

`run.sh` 现在统一启用已验证插件；`run_compat.sh` 为兼容别名。第一轮默认入口的失败日志仍保留，入口改变不表示旧失败已经通过。003 实际执行的监督器身份与原始快照已保留；后续增加显式 run ID 和清理校验的控制补丁仅完成 CPU 回归，见 [控制补丁记录](../../evidence/rules/session003-run-identity-cpu-004.json)。后续若官方镜像修复原生类，需要复核并更新配置，不自动绕过源码保护。

## 4. 环境准备与显式启动

prepare 可单独运行且不会启动服务或加载模型；默认 run.sh 在新 venv 缺失时先安全准备再启动，已有目录只做校验，失败不自动覆盖。默认计划可在本地 Windows / WSL 查看，不下载、不建 venv、不检测 GPU：

```bash
cd tasks/qwen3-asr-bw-vllm
bash prepare.sh --dry-run
bash run.sh --dry-run
```

目标原生 Linux x86_64 / Python 3.10 镜像中，确认算力使用安排后，才执行准备：

```bash
# 原生解释器；目标目录默认本任务 runtime/audio-venv，必须尚不存在。
SCNET_BASE_PYTHON=/usr/bin/python3 bash prepare.sh --apply
# 这是独立一步，只有显式 run 才启动 GPU 服务。
bash run.sh
# 在另一终端做检查与请求，然后停止本项目服务：
bash env_check.sh
bash smoke.sh --audio /实际短语音.wav
bash stop.sh
```

准备时仅在新建 `--system-site-packages` venv 内以 `--no-deps --require-hashes --only-binary=:all:` 安装缺失的锁定 wheel；不会动态解析或升级原生包。已存在的音频包如版本不符，或原生栈版本不同，将在安装前拒绝。安装后检查包版本与路径、依赖闭包、WAV 解码和重采样，以及兼容插件所需配置源码身份。原始记录写入 venv 的 `asr-prepare-evidence/`；失败目录保留供检查，不自动覆盖、删除或重试。session003 已在另一实例的同一原生栈实际执行新增准备流程并用新 venv 完成服务请求；这不是最终交付镜像的从零验收。

已有历史 venv 可先做 CPU 验证，再显式使用；不要求重新安装：

```bash
bash prepare.sh --check-existing /root/private_data/venvs/qwen3-asr-audio-001
export QWEN_ASR_PYTHON_BIN=/root/private_data/venvs/qwen3-asr-audio-001/bin/python
bash run.sh
```

自定义新 venv 使用 `prepare.sh --venv /新的绝对目录 --apply`，随后显式设置对应 `QWEN_ASR_PYTHON_BIN`。`run.sh` 实际启动前会再验证该 venv；未准备时 `--dry-run` 只用基础 Python 展开启动参数，并明确提示没有已准备的 Python，不表示服务可启动。

全部模型、端口和预算覆盖变量使用 `QWEN_ASR_*`。默认加载上限 600 秒、总服务上限 900 秒，仅用于短轮诊断；正式长评测需要按已批准预算另设。监督器停止服务不会关闭 Notebook 或停止平台计费，平台关机必须单独完成。2026-10-19 的历史实例已关机，历史 2 卡时额度不构成本轮新开卡授权。

## 5. Smoke 测试

独立标准库客户端上传本地 WAV 到 `/v1/audio/transcriptions`。默认输入要求单声道、16 kHz、PCM16、时长大于 0 且不超过 30 秒、文件最多 2 MiB；显式 `--long-audio` 开发模式放宽至 120 秒 / 4 MiB，编码要求不变，不自动裁剪或重采样。

```bash
bash smoke.sh --audio /实际短语音.wav --dry-run
bash smoke.sh --audio /实际短语音.wav
# 仅在本地比较，参考文本不会放进请求。
bash smoke.sh --audio /实际短语音.wav --expected-text "人工核对的文本"
# 明确标记长音频开发模式，不能将拼接样本称为自然连续录音。
bash smoke.sh --audio /已准备的长语音.wav --long-audio
```

multipart 只有 `model`、`file`、`response_format=json`、`temperature=0`，不发送语言、prompt 或参考答案。先确认 `/v1/models`，仅向 loopback 请求，不跟随重定向或重试。JSONL 使用新文件名，拒绝覆盖；日志保存 SHA256 和音频元数据，不保存 WAV payload。非空转写只是接口功能证据，精确文本比较不等于 WER/CER；API 未提供 finish reason，不能据此保证没有截断。

session003 的三条样本时长为 4.2039375、15.05125、39.260375 秒，均返回非空转写，见[中文](../../evidence/asr-vllm-smoke-zh-003.jsonl)、[英文](../../evidence/asr-vllm-smoke-en-003.jsonl)、[拼接长输入](../../evidence/asr-vllm-smoke-long-003.jsonl)。样本来源及 PCM24/48k→PCM16/16k、拼接过程见[fixture manifest](../../evidence/asr-fixture-manifest-003.json)。没有人工参考，响应未提供 finish reason，日志未截断不等于已证明模型没有截断。

独立 `scripts/qwen3_asr_stream_smoke.py` 的默认模式只做计划，显式 `--run --log 新文件` 才向 loopback 上传完整 WAV 并读取 SSE。实际中文请求收到 12 个事件、`finish_reason=stop`、`[DONE]` 和 usage（65 prompt + 11 completion），原文保留 `language Chinese<asr_text>` 前缀，见[SSE 原始记录](../../evidence/asr-vllm-sse-zh-003.jsonl)。这是整文件音频上传后的输出流式，不是分块实时音频输入；自然连续长语音、噪声、多语种和实时输入仍待验证。

## 6. 精度测评

另有纯离线开发工具 `scripts/asr_development_eval.py`，接收严格逐 ID 匹配的 reference/prediction JSONL，默认 identity 规范化，输出原始 WER/CER 与 S/D/I、文件和文本 SHA256；不调用模型、不生成官方 baseline 或 PASS。WER 明确仅按空白分词，不暗中对中文、数字、标点或大小写作处理，CER 保留原文全部 Unicode code point。manifest 默认只验证计划，显式 `--check-audio` 才检查本地短 WAV；格式、CPU 限额、主来源和使用方式见 [离线开发准备说明](../../docs/asr-offline-readiness.md)。它没有接管 `verify.sh` 或替代赛事评测器。

尚缺赛事固定音频清单、参考转写、GPU baseline、WER/CER 规范化、语言策略、解码配置及正式判分入口。发布方成绩和通用 HYGON 评测工具可用于开发参考，不能未经核对就当作本分题官方验收结果。

`verify.sh` 已启用真实官方桥：`--selfcheck` 只运行外部原版 selfcheck，完整 eval 需真实数据/baseline/脚本；缺材料返回 2，未知 schema 保持 executed_uninterpreted。`task.json` 中 `max_drop=0` 是禁用阶段的占位值，不是官方门槛。官方表述需要完整镜像地址和 tag；digest 可作为可复现补充证据，未把它设为额外硬性验收条件，也不把单独认领回执设为开发前置条件。

## 7. 性能参考

历史两次 HTTP 往返约 5.13 秒、0.82 秒，未预热，缓存和初始化条件不同，不能据此报告吞吐或 RTF。`bench.sh` 已启用 T04 显式外部音频脚本和输出 contract，缺材料返回 2，不会将 ASR 送进 T02 图像六档压测。正式音频时长、batch、并发、流式延迟、预热和计分方式仍需完整 T04 方法。

session003 中文、英文和拼接长音频 HTTP 往返分别约 4.889、1.807、4.204 秒；SSE 第一非空 delta 约 0.074 秒、输出流用时约 0.424 秒。它们是单次开发观测，未构成正式延迟、RTF 或吞吐评测。ASR 监督器实际从创建到停止约 168.077 秒；合并 RoboBrain+ASR 开发驱动的 547.572 秒不能全部算作 ASR 推理或卡时。

## 8. 待完成交付

- [x] 原始权重、legacy BPE、BF16 加载和两条短 WAV 转写证据。
- [x] 审核的 CPU wheel 锁、默认纯本地准备计划、显式 prepare + run 入口。
- [x] 纯离线 development_only WER/CER 工具与短 WAV manifest 计划/检查；不是正式精度结果。
- [x] 在另一实例同原生栈新建隔离 venv，执行新增准备流程并完成服务请求。
- [x] 三条发布方派生 WAV（含人工拼接长输入）及完整 WAV SSE 输出验证。
- [ ] 最终交付镜像的干净环境从零复跑。
- [ ] 自然连续长语音、噪声、多语种和实时音频输入验证。
- [ ] 正式 WER/CER、性能原始记录与适用门槛判定。
- [ ] 保存最终镜像，取得完整 registry/repo:tag，整理依赖锁、报告和代码仓库，核对评审访问权限；当前仓库保持 private。
- [ ] 核对提交表材料并按用户最终提交安排完成提交。

本目录仍标记 `adaptation_ready=false`、`formal_ready=false` 和 `official_accuracy=false`。准备脚本不会替代缺失的官方数据、baseline 和评测器，也不会自动发布或提交。原始 runtime 快照已逐字节复制为扁平 evidence 文件，来源/目标 SHA256 与核查记录见 [copy provenance](../../evidence/asr-copy-provenance-003.json)，原件没有修改。仓库更新是否落库与赛事提交状态另行核对，不用本地报告推定提交成功。
