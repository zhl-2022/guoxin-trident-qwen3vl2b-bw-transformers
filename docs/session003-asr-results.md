# Session003 ASR 实际运行审计

## 结论与证据身份

2026-10-20，原始 `Qwen/Qwen3-ASR-1.7B` 在另一 BW 单卡实例的同原生栈中，新建任务 venv 并实际完成准备、BF16 服务启动、三条发布方派生 WAV 转写和一次完整 WAV 的 SSE 输出。服务随后主动停止，驱动核验所拥有的进程不存活且端口关闭。当前仍为 `formal_ready=false`、`official_accuracy=false`、`official_performance=false`；没有正式验收结论。

下载包 `dist/session003-completed-intermediate.zip` 的实际 SHA256 为 `61dae05b974e3b0151e90885818493bfe27f461c58125f529eae1c79cbcba9fd`。本次重新核对导出清单中全部 121 个 payload 的大小与 SHA256。该导出是本轮已完成材料的中间快照，不能据此认为同时运行的其他题性能评测已经结束。

为让打包规则继续排除 runtime 目录，选取 16 份 ASR 原始证据逐字节复制到 `evidence/asr-*-003`。完整源路径、目标路径、双 SHA256、实际样本与执行源码身份见 [copy provenance 与审计记录](../evidence/asr-copy-provenance-003.json)。原 session 文件没有修改，历史 001/002 的失败或成功也没有改写。

## 新隔离环境的真实准备

新解释器为 `/root/private_data/guoxin-hackathon-20261020-session003/tasks/qwen3-asr-bw-vllm/runtime/audio-venv/bin/python`。这次确实执行了 `prepare.sh --apply`，不再仅为本地 dry-run；[pip 日志](../evidence/asr-audio-pip-install-003.log)记录七包成功安装：[实际 missing wheel 锁](../evidence/asr-audio-missing-wheels-003.txt)与任务 requirements 中七条审核 URL/SHA256 完全一致。

安装内容为 librosa 0.11.0、soundfile 0.13.1、audioread 3.0.1、lazy-loader 0.4、pooch 1.8.2、scikit-learn 1.6.1、soxr 0.5.0.post1。安装前后的 12 个受保护原生包版本与目录逐项相同，见 [native-before](../evidence/asr-native-before-003.json)和[prepare ready](../evidence/asr-clean-prepare-003.json)。后者同时证明音频依赖闭包、1600 点 WAV 解码与重采样到 800 点的检查完成；validation stderr 为 0 字节。

新准备不导入模型或占用 GPU；后续 driver 又执行 `--check-existing`，然后才单独启动服务。新增准备流程已在另一实例同镜像原生栈运行，尚不等于最终交付镜像的从零验收。受保护版本仍为 DAS Torch 2.10.0、native vLLM 0.18.1、Transformers 5.5.0、NumPy 1.25.0、SciPy 1.15.3。

## 服务与三条实际输入

[ready 状态](../evidence/asr-vllm-state-003.json)、[启动配置](../evidence/asr-vllm-config-003.json)与[完整服务器日志](../evidence/asr-vllm-server-full-003.log)对应同一 run ID `6b00889feafc4ea1af2133b5c4a3317a`。实际为 BF16、TP=1、32768 context、显存比例 0.85、eager、最多 4 个序列、`127.0.0.1:30002`。本轮 driver 将任务服务时限显式设为 1800 秒，默认入口的 900 秒诊断值没有被修改。

| 输入 | 音频时长 | HTTP 往返 | 返回字符数 | 结论 |
|---|---:|---:|---:|---|
| [中文](../evidence/asr-vllm-smoke-zh-003.jsonl) | 4.2039375 秒 | 4.889441 秒 | 14 | 非空转写，未提供人工参考 |
| [英文](../evidence/asr-vllm-smoke-en-003.jsonl) | 15.05125 秒 | 1.807355 秒 | 188 | 非空转写，未提供人工参考 |
| [中英人工拼接](../evidence/asr-vllm-smoke-long-003.jsonl) | 39.260375 秒 | 4.204148 秒 | 393 | 显式 long-audio 开发模式，非自然连续长录音 |

三条实际 WAV 的 SHA256、帧数、单声道/PCM16/16kHz 布局与客户端输入记录逐一相同。来源和转换见 [fixture manifest](../evidence/asr-fixture-manifest-003.json)：中文保持原 PCM16 样本；英文从发布方原始 PCM24/48kHz 经已记录的 `scipy.signal.resample_poly(1,3)` 转为 PCM16/16kHz；长输入为 `zh,en,zh,en`，相邻段插入 0.25 秒静音。原发布方文件没有修改。

中英文原文件来自 Qwen 发布方 README 示例。本报告没有确认其自然录音身份，也没有将它们当作赛事固定数据。所有请求均未发送语言提示、参考答案或热词。三条记录的参考文本为 `null`；返回文本只作为实际预测保留，不反过来生成 reference，不计算 WER/CER。以后仅有明确、独立人工参考时才运行离线开发评分器。

普通转写接口未返回 finish reason/token 数；日志中的 `raw_text.truncated=false` 仅表示客户端日志没有截短该返回，不足以证明模型输出完整或准确。

## SSE 的已验证范围

[中文 SSE 原始记录](../evidence/asr-vllm-sse-zh-003.jsonl)显示：同一 4.2039375 秒 WAV 先以完整文件上传，随后读取输出 SSE；收到 12 个事件、一个 `finish_reason=stop`、`[DONE]`，以及 usage：65 prompt、11 completion、76 total token。首个非空 delta 约 0.074383 秒，输出流总用时约 0.423636 秒。

原始流文本为 `language Chinese<asr_text>` 加中文转写，客户端明确 `asr_prefix_stripped=false`。没有悄悄删去协议前缀再声称原生返回干净文本。单次输出流中的 stop/DONE 已观察到；这不延伸为三条普通转写的截断证明，也不等于实时音频输入、逐块音频上传、端点检测或实时延迟已验证。

这些 HTTP 与 SSE 用时是单次开发观测，未构成正式 RTF、吞吐、延迟分布或精度测试，不能与发布方成绩表直接比较。

## 停止、警告与剩余工作

北京时间 2026-10-20 01:57:35.943807 创建 ASR 监督器，02:00:07.789453 ready，02:00:24.020834 stopped，总历时 168.077 秒，见[停止状态](../evidence/asr-vllm-stopped-state-003.json)和[监督器日志](../evidence/asr-vllm-supervisor-003.log)。驱动的 ASR `cleanup_verified=true` 同时检查所拥有 run 的进程及端口，不只查看一个 stopped 字段。

合并 RoboBrain+ASR 驱动历时 547.572277 秒，包含两题的环境检查、数值检查、加载、请求及清理，不含更早的新 venv 准备和平台 provisioning，不能全部归为 ASR 推理或实际计费卡时。[原始合并驱动报告](../evidence/asr-session-driver-report-003.json)原样保留。根代理已报告第二实例保存环境并关机；服务进程清理不单独证明平台计费结束，平台关闭与余额以相应 UI 证据为准。

服务器日志保留通用 model_type 警告，以及停止时一个 leaked semaphore 的资源追踪警告；没有删除这些行。API 父进程及 EngineCore 均记录受原配置 SHA256 保护的 TF5.5 兼容插件安装。驱动已验证本 run 进程和端口清理，但正式长运行及验收仍需继续观察。

尚缺正式 T04 固定音频、独立人工参考、GPU baseline、原版判分器、WER/CER 规范化、解码设置与音频性能口径，以及最终交付镜像的完整地址/tag和干净环境从零复跑。自然连续长录音、噪声、多语种、实时音频输入也未完成。本报告不改变光合仓库是否落库、GitHub 推送或赛事提交状态；尤其不把一次尚未落库的 WebIDE 更新记录成新提交。
