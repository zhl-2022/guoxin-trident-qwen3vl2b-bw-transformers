# Qwen3-ASR-1.7B 原生 HCU 环境准备依据

核查日期：2026-10-19。本次只读研究公开上游和已保存的环境证据，没有下载权重或音频、安装软件、克隆仓库或操作平台页面。

## 当前判断与进入条件

当前原生 Torch 环境具有可继续验证的技术路径：上游固定 vLLM `v0.18.1` 已内置 `Qwen3ASRForConditionalGeneration`、配置、processor 和转写接口，不要求先安装 Qwen 的 Python 包。现有 HCU 镜像已经实测原生 Torch 2.10、Transformers 5.5、vLLM/HCU 0.18.1 及 FlashAttention 导入通过，见 [vllm-probe-001.json](../evidence/vllm-probe-001.json)。

但该 probe 尚未检查本机 ASR 架构注册、音频依赖、processor 和声学 encoder，也未运行 ASR。因此当前结论是“保留原生 Torch、先补 ASR 专项只读检查”，不是“本模型已可用”。HYGON 官方现有 ASR cookbook 只列 BW1100 / vLLM 0.21 与 0.28.1；没有给出当前 `gfx936` / 0.18.1 构建的直接验证记录。

## 固定模型与上游版本

| 项目 | 已核验信息 | 来源 |
|---|---|---|
| 本阶段模型 | `Qwen/Qwen3-ASR-1.7B`，原始发布格式，不带 `-hf` | [原模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| 模型修订 | `7278e1e70fe206f11671096ffdd38061171dd6e5` | [模型 API](https://huggingface.co/api/models/Qwen/Qwen3-ASR-1.7B) |
| 模型类 / 类型 | `Qwen3ASRForConditionalGeneration` / `qwen3_asr` | [固定 config.json](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/7278e1e70fe206f11671096ffdd38061171dd6e5/config.json) |
| 结构 | `thinker_config` 下的音频 encoder 与 Qwen3 文本 decoder；24 层音频 encoder，28 层文本 decoder | 同上 |
| 参数与文件 | API 统计 2,349,217,408 个 BF16 参数，2 个 Safetensors 分片；“1.7B”不是全模型参数总数 | 模型 API 与文件清单 |
| Qwen 源码修订 | `7c6daf77a2421100f5fb066495372c00129d39ff` | [固定官方仓库](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff) |

BF16 参数按每项 2 字节估算约 4.70 GB，这不是运行显存需求；音频特征、encoder 激活、KV cache 和服务预分配需要额外空间。

不要把 `Qwen3-ASR-1.7B-hf` 当成同一目录替代品。其 [官方模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) 要求 Transformers 5.13.0 起的原生接口，当前镜像是 5.5.0；本阶段使用 vLLM 内置适配的原始 checkpoint，不为切换该变体升级现有框架。

## vLLM 0.18.1 中已经存在的代码路径

| 固定上游文件 | 已确认能力 | 本机仍需验证 |
|---|---|---|
| [models/registry.py](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/models/registry.py) | 注册 `Qwen3ASRForConditionalGeneration` | 当前 DAS fork 是否保留该注册 |
| [models/qwen3_asr.py](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/models/qwen3_asr.py) | 实现音频编码、Qwen3 解码、`SupportsTranscription`、转写 prompt 与输出处理 | HCU encoder/attention 的真实执行与权重映射 |
| [configs/qwen3_asr.py](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/transformers_utils/configs/qwen3_asr.py) | 内置原始 Qwen3-ASR 配置 | 本地 config 解析与实际 checkpoint 一致 |
| [processors/qwen3_asr.py](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/transformers_utils/processors/qwen3_asr.py) | 内置 `WhisperFeatureExtractor` + Qwen2 tokenizer 的 processor；默认采样率 16 kHz | 当前 Transformers 5.5 下导入、特征抽取、模板与 tokenizer |

HYGON 的 [固定 ASR cookbook](https://github.com/HYGON-AI/inference-cookbook-das/blob/0a5aa29b5720c36d2e033e220834edfe7ed6a6ce/docs/model-deployment/vllm/qwen3-asr.md) 提供 BF16 单卡服务与音频 API 示例，但其 0.21 / 0.28.1 配方不能直接变成当前 0.18.1 的验证结果。尤其不要复制新版本 AITER、FlashAttention unified 或新 runner 开关；当前构建应先沿用本机默认，再依据错误和源码决定最小调整。

## 音频输入和 API

首轮使用短的真实语音 WAV，先做单条、低并发。输入应明确记录文件 SHA-256、字节数、采样率、通道数、采样点数和时长，保留原始文件；用于正式评测时不要私自截断或改变音频。

Qwen 官方 [音频工具](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/utils.py) 将音频规范化为单声道、16,000 Hz、`float32` 波形，范围为 `[-1, 1]`，并处理声道合并与重采样。vLLM 原生 processor 的默认采样率同为 16 kHz。初始功能样本优先选已符合该格式的 WAV，减少格式转换变量；HTTP 上传的是带容器头的音频字节，不能将任意 PCM 字节冒充 WAV。

原始 checkpoint 的 [preprocessor_config.json](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/7278e1e70fe206f11671096ffdd38061171dd6e5/preprocessor_config.json) 使用 128 个 mel bin、`n_fft=400`、`hop_length=160`、`chunk_length=30`。Qwen Python 包支持自行分块长音频，不代表当前原生 HTTP 路由的长音频分块、流式和时间戳行为已经验收；首轮先使用不超过 30 秒的样本。

| 接口 | 本阶段可准备的请求 | 返回注意事项 |
|---|---|---|
| `/v1/chat/completions` | user 的 content 含 `audio_url`；可使用可达 HTTPS URL，或已核验格式的音频 data URL | 原始文本可能含语言前缀和 `<asr_text>`；保存 raw 和解析结果 |
| `/v1/audio/transcriptions` | multipart 音频文件、`model` 和 `response_format=json` | v0.18.1 模型实现支持转写，并会处理输出标记；本机路由仍需实际请求验证 |
| Python `Qwen3ASRModel.transcribe` | 官方包可接收路径、URL、base64 或 `(np.ndarray, sr)` | 本阶段不安装该包；不能把它的参数直接当成 HTTP 字段 |

请求结构示意，以下别名需要与实际服务的 served model name 对齐：

```json
{
  "model": "qwen3-asr-1.7b",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,<WAV_BASE64>"}}
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 1024
}
```

以上 API 形式分别见 Qwen [官方部署说明](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/README.md#deployment-with-vllm) 与 HYGON cookbook；此示意尚未在当前实例提交。

## 模板、语言与解码基线

保留 checkpoint 原始 `chat_template.json`。其格式收集 system context 与音频占位符，然后生成 assistant 前缀；不要照搬视觉模型的“描述图片”等 user 文本。初始请求使用空 context 与自动语言识别，不注入语言、热词或提示答案。

Qwen 官方 [推理实现](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/qwen3_asr.py) 对强制语言使用 assistant 前缀 `language {Language}<asr_text>`。而上游 vLLM 0.18.1 的 `get_generation_prompt` 使用 `to_language` 决定是否加入语言前缀，不能假定 HTTP `language` 字段与 Qwen Python 包的 `language=` 具有同样行为。正式基线不指定语种，避开该差异；需要强制语种时再逐接口核验。

发布方 [评测说明](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/README.md#evaluation) 使用 BF16、vLLM、greedy、最多 1,024 个新 token，未指定语言。功能阶段可沿用 `temperature=0`、`max_tokens=1024` 并检查 finish reason，不能把输出截断当成成功。正式成绩仍以赛事下发的精确解码要求为准。

## 包依赖边界

先读元数据并分别验证音频依赖；缺失依赖只在新增隔离环境中补齐，不运行原生环境升级命令。

| 路径 | 一手依据 | 当前处理 |
|---|---|---|
| vLLM 原生音频依赖 | [v0.18.1 setup.py](https://github.com/vllm-project/vllm/blob/v0.18.1/setup.py) 的 audio extra 含 librosa、scipy、soundfile、mistral_common[audio]、av | 检查是否已在镜像中，逐项隔离导入、解码 WAV；不要直接重新安装整个 vLLM extra |
| Qwen Python 包 | [固定 pyproject.toml](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/pyproject.toml) 版本 0.0.6 固定 Transformers 4.57.6、Accelerate 1.12.0，optional vLLM 固定 0.14.0 | 与当前 Transformers 5.5 / 原生 vLLM 0.18.1 不匹配，不直接安装 `qwen-asr[vllm]` |
| FlashAttention | 当前镜像已实测导入原生 DAS build，Recon 已实际使用原生 attention | 不安装通用 wheel 覆盖；ASR 声学 encoder 仍需单独验证 |
| ForcedAligner | 单独的 checkpoint 和时间戳处理路径 | 首轮仅转写，不因 ASR 加载自动下载对齐模型 |

`qwen-asr-serve` 属于 Qwen 包包装命令；本阶段优先验证已有 vLLM 原生服务路径。若原生路径缺少组件，先保存明确错误和依赖要求，再决定隔离安装最小组件或使用另一个完整原生镜像，不替换当前 Torch。

### 已观察到的缺包与最小 WAV 路径

实际 [asr-probe-001.json](../evidence/asr-probe-001.json) 确认原生 ASR 注册、模型、配置、processor 文件均存在，NumPy 为 1.25.0、SciPy 为 1.15.3，`python-multipart` 为 0.0.22；`librosa`、`soundfile`、`torchaudio` 和 FFmpeg 缺失。文件存在不代表模型已加载成功。

上游 vLLM v0.18.1 的 [speech_to_text.py](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/entrypoints/openai/speech_to_text/speech_to_text.py#L193-L222) 先调用 `librosa.load(BytesIO(audio_data), sr=model_sample_rate)`；只有 libsndfile 无法识别容器时才走 PyAV 分支。其 [audio media 实现](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/multimodal/media/audio.py) 也用 librosa/soundfile。所检查的转写入口和 Qwen3-ASR 实现没有调用 torchaudio。因此首轮限定单声道 16 kHz PCM WAV，可以仅补 librosa/soundfile 及 CPU 依赖闭包；不因此安装 torchaudio、FFmpeg、PyAV 或 Qwen 包。这是固定版本源码给出的依赖结论，仍需当前 DAS fork 实际转写验证。

| 缺包时选用的固定版本 | PyPI 一手元数据与兼容依据 |
|---|---|
| librosa 0.11.0 | [PyPI JSON](https://pypi.org/pypi/librosa/0.11.0/json)：Python ≥3.8、NumPy ≥1.22.3、SciPy ≥1.6；依赖 soundfile、numba、soxr、scikit-learn 等 |
| soundfile 0.13.1 | [PyPI JSON](https://pypi.org/pypi/soundfile/0.13.1/json)：依赖 CFFI 和 NumPy，有带 libsndfile 的 manylinux x86_64 wheel |
| numba 0.61.2 / llvmlite 0.44.0 | [Numba JSON](https://pypi.org/pypi/numba/0.61.2/json)、[llvmlite JSON](https://pypi.org/pypi/llvmlite/0.44.0/json)：Python ≥3.10、NumPy ≥1.24 且 <2.3、llvmlite ≥0.44 且 <0.45，均有 CPython 3.10 Linux wheel |
| soxr 0.5.0.post1 / scikit-learn 1.6.1 | [soxr JSON](https://pypi.org/pypi/soxr/0.5.0.post1/json)、[scikit-learn JSON](https://pypi.org/pypi/scikit-learn/1.6.1/json)：Python ≥3.9；后者要求 NumPy ≥1.19.5、SciPy ≥1.6，均兼容已观察版本 |

[prepare_audio_env.sh](../scripts/prepare_audio_env.sh) 创建新的 `--system-site-packages` venv，保留现有发行包。它先输出 pip dry-run JSON，只允许缺失 CPU 音频依赖的固定版本；若解析计划要覆盖任何已存在包、引入未审阅包或下载非 PyPI wheel，立即停止。安装使用报告中的 wheel URL 与 SHA256，并带 `--no-deps --require-hashes`。公共依赖已存在且满足要求时直接复用；若不满足则停止，不擅自替换。

```bash
# 路径必须不存在；省略 --apply 则仅生成计划，不安装。
bash scripts/prepare_audio_env.sh \
  /root/private_data/venvs/qwen3-asr-audio-001 --apply
```

脚本仅针对已观察的 Linux / Python 3.10 / NumPy 1.25.0 / SciPy 1.15.3 原生环境，记录原生包版本和路径，验证依赖闭包、内存 WAV 解码及 CPU 重采样，再核对原生包未变。该检查不加载模型、不分配 GPU、不证明识别精度。启动服务时必须显式使用新 venv 的绝对 `bin/python` 路径；原生 shell Python 不会自动切换。新目录中的 `audio-env-evidence` 保存解析计划、固定 wheel 清单和验证结果；失败时保留目录供检查，脚本不会覆盖或递归清理它。

## 当前实测结果

`asr-preflight-002.json` 已通过实际 legacy BPE 检查；两分片的原始哈希报告 001 保留。隔离 venv 已实际完成 WAV 解码/重采样，七个新增包的 wheel URL 与 SHA256 记录在 `asr-audio-reviewed-plan-002.json`，原生包未变，见 `asr-audio-validation-002.json`。实际运行安装脚本为 SHA256 `5ea10013a1d31ffeef8d6ef8e15a7bc106b9e6975d5e19ac1febcadd210d7ed8`；当前源码另有重复发行包元数据取首个可见路径的小改动，不能把新 hash 写成实际已执行。

首轮服务在加载权重前因 TF5.5 的配置初始化顺序失败，完整日志保留。显式项目插件 `scripts/qwen_asr_tf55_compat.py` 先创建同值 thinker，再调用原构造函数，保留父类校验；带源码与版本 guard，通过任务局部的 general_plugins 同时进入 API 和 EngineCore。第二轮服务成功加载 BF16、开放 loopback 转写接口，返回与两条自有中英文合成音原稿一致的文本，随后停止。输入、返回、时间和退出状态见 `asr-vllm-smoke-*-002.jsonl`、`asr-vllm-state-002.json`、`asr-vllm-stopped-state-002.json`。没有降级或覆盖原生框架，没有安装 Qwen Python 包。

此配置兼容插件是当前镜像的显式候选方案，尚需干净环境和正式数据复跑；TF5.5 还打印了 RoPE 参数兼容告警，两条短样本不能排除长音频或精度问题。

## 样本、正式数据与验收边界

Qwen 官方发布了 [中文示例 WAV](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav) 和 [英文示例 WAV](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav)，均在官方 README 和 HYGON API 示例中使用。它们是后续功能检查的候选，本次未下载。

赛事 T04 的数据集、split、参考转写、WER/CER 文本规范化规则、语言策略和性能压测口径仍待官方材料确认。公开示例音频、合成音、静音或任意公共数据子集都不能替代正式评测集，也不据此填报成绩。

后续进入顺序：确认当前安装包包含 ASR 注册与原生配置/processor → 验证音频解码和本地 checkpoint 配置/模板 → 小批量 BF16 加载 → 中文、英文单条转写及接口差异检查 → 再按官方材料进行精度与性能测量。任何一层通过都不自动代表下一层通过。
