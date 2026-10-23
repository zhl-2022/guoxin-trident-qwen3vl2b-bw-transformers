# ForcedAligner 原生框架路线与实际镜像身份核查

## ForcedAligner 当前结论

固定 `Qwen/Qwen3-ForcedAligner-0.6B` 不可以直接按标准 vLLM Qwen3-ASR 生成模型运行。外层 architecture 名称相同，内部输出头和推理协议不同。官方 vLLM 0.18.1 尚未在核查源码中实现这个分类分支；DAS 分支是否另外增加了支持，仍需读取当前实际安装文件确认。

本轮没有修改现有代码。官方 config、generation config、权重头及固定源码保存在 `evidence/forcedaligner-native-route-20261023/`；权重仅读取 86,752 B 的长度与 JSON 头，没有读取 tensor payload。

| 项目 | 固定 ForcedAligner 实际内容 | 标准 vLLM 0.18.1 Qwen3-ASR |
|---|---|---|
| 外层 architecture | `Qwen3ASRForConditionalGeneration` | 相同名称，不能独立证明语义兼容 |
| 内层 model_type | `qwen3_forced_aligner` | 使用 ASR thinker 的 text_config 构造 Qwen3ForCausalLM |
| 输入词表 | `thinker.model.embed_tokens.weight=[152064,1024]` | 按 text vocab 构造输入 embedding |
| 输出头 | `thinker.lm_head.weight=[5000,1024]`；config `classify_num=5000` | Qwen3ForCausalLM 的生成头按 `text_config.vocab_size=152064` 构造 |
| 计算 | 一次完整 thinker forward，在已有输入的 timestamp 位置分类 | 常规生成 logits/sampling 路径 |
| 时间戳 | 输入 token ID 151705；分类索引乘以 80 ms | 核查文件中没有 classify_num、timestamp 位置收集或时间修正分支 |
| 后处理 | 官方分词/按字切分，每个单位有两个 timestamp 占位；对输出作单调修正，再转秒 | 不能用 token 文本解码替代 |

固定 config 来源：[模型配置](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/blob/c7cbfc2048c462b0d63a45797104fc9db3ad62b7/config.json)。分类输出头与完整输入前向见 [官方模型实现](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)，位置选择与时间戳修正见 [官方 aligner](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/qwen3_forced_aligner.py)。原生 vLLM 对照：[Qwen3-ASR 实现](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/models/qwen3_asr.py)、[Qwen3 语言模型头](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/models/qwen3.py)。

不要将 text vocab 改成 5000 来迎合输出头：这会破坏 152064 行的真实输入 embedding 和包括 timestamp/audio 在内的高 ID token。也不要通过 ordinary generate 输出的类别 ID 伪装成文本时间戳；正确结果来自输入中的全部 timestamp 位置。

## 可以形成闭环的两条路线

现有发布方 Transformers runner 可验证真实时间戳，适合作为功能产物与后续数值参考；赛事是否接受 TF-only 仍未闭合。官方 `Qwen3ASRModel.LLM(forced_aligner=...)` 也会另行调用 `Qwen3ForcedAligner.from_pretrained`，形成 vLLM ASR 加 Transformers aligner 的组合，不能据此声称 aligner 已在 vLLM 原生运行。来源：[官方组合入口](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/qwen3_asr.py)。

若要求 aligner 本身走 vLLM，最小真实开发范围是一个独立、注册明确的 token-classification 模型适配：

1. 复用已验证的 ASR 音频 encoder 和 Qwen3 因果文本 backbone，完整保留输入词表与位置编码。
2. 单独加载真实的 5000×1024 分类头，和 152064 行输入 embedding 分离；不能沿用生成 head 的尺寸或 LM sampling。
3. 为完整 prefill 保留每个有效 timestamp 输入位置的 hidden state，分类而非续写；多模态 token 展开后的位置索引必须与官方 processor 一致。
4. 返回每个单位两个时间类别，复用固定发布方的切分、80 ms 换算和修正规则。token-classify/pooling 的选择、输出形状、分批与 chunked-prefill 行为需要专门实现与核查，不能只添加一个 architecture override。
5. 在同一音频/文本上与固定 TF 实现对比原始类别和最终时间戳，再执行独立 timestamp reference 的 AASms 与性能评测。ASR 文本正确不证明对齐时间准确。

这是一项新增模型适配工作；现阶段只有明确技术路线，没有已通过的原生插件。当前 GPU 窗口应先保存真实 TF 结果和原件，保持框架资格 pending，避免把包装 API 计作原生完成。

## 当前 native 源码的无 GPU 读取命令

在平台原生 Python 下执行下列命令。只读取安装元数据和包源码，不导入 torch/vLLM，不读取环境变量或认证文件。

```bash
python3 -B - <<'PY'
import hashlib, importlib.metadata as m, json
from pathlib import Path
d = m.distribution('vllm')
p = Path(d.locate_file('vllm/model_executor/models/qwen3_asr.py'))
b = p.read_bytes()
lines = b.decode().splitlines()
needles = ('classify_num', 'forced_aligner', 'timestamp', 'Qwen3ForCausalLM', 'compute_logits', 'lm_head', 'pooler')
print(json.dumps({'version': d.version, 'source': str(p), 'sha256': hashlib.sha256(b).hexdigest(),
                  'matching_source_lines': [{'line': i+1, 'text': s} for i,s in enumerate(lines)
                                            if any(x in s for x in needles)]}, indent=2))
PY
```

若 DAS 文件中出现专用 ForcedAligner 分类/pooling 分支，继续精确读取其实现与测试；若没有，则本轮上游阻碍也适用于当前安装的这份文件。

## 实际 Jupyter 镜像身份

既有 SCNet 原件记录实际私有镜像 ID `2071293281204580354`、版本 `2082674417745739777`、名称 `vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`、tag `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`，显示大小 51.64 GB。现有 task 元数据的 registry/digest 仍为空。

官方 ModelZoo 可确认公开 base 引用 `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`；公开说明不能推出实际私有 Jupyter 派生镜像所在 registry/repo。来源：[官方 ModelZoo 固定 README](https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/blame/2c86d258f40afe2d6698f3702663e1c82b6a3eec/README.md)。

当前 session004 native 安装元数据已记录：JupyterLab 4.6.2、Jupyter Server 2.20.0、Notebook 7.6.1、jupyter 1.1.1、jupyter_core 5.9.1、jupyter_client 8.9.1；位置均为 `/usr/local/lib/python3.10/dist-packages`。这些是环境实测版本，不能反推镜像 digest。依据：`evidence/session004/sglang-client-native-context-004.json` 的 native 部分；原镜像定位记录见 `docs/session003-delivery.md` 第 6.2 节。

可以在实际 Notebook 再读取版本和元数据 SHA，不输出密码、token、完整环境、进程参数或 Jupyter server 配置：

```bash
python3 -B - <<'PY'
import hashlib, importlib.metadata as m, json, sys
rows = {}
for name in ('jupyterlab','jupyter_server','notebook','jupyter','jupyter_core','jupyter_client','vllm','vllm-hcu','torch','transformers'):
    try:
        d = m.distribution(name)
        data = d.read_text('METADATA')
        rows[name] = {'version': d.version, 'root': str(d.locate_file('')),
                      'metadata_sha256': hashlib.sha256(data.encode()).hexdigest() if data else None}
    except m.PackageNotFoundError:
        rows[name] = None
print(json.dumps({'python':sys.version,'executable':sys.executable,'distributions':rows},indent=2))
PY
```

要闭合实际 OCI 引用，优先从 SCNet 当前 Notebook 的镜像版本详情读取后端返回的 image URL/repository/tag/digest，仅保存这些字段。若当前容器可访问自己的 Docker metadata，可只针对当前 hostname 读取：

```bash
docker inspect --format '{{json .Config.Image}} {{json .Image}}' "$(hostname)"
```

此命令可能因平台使用 Kubernetes/containerd、Docker 不可用、hostname 不是 Docker container ID 或权限不开放而失败；失败不会推出镜像不存在。不要为读取镜像信息提权、安装 Docker、枚举其他用户容器或读取 Docker/Jupyter 认证配置。容器内包版本、tag 名、OS release 和 container ID 都不足以重建完整 registry/repo:tag；平台详情没有给出时，完整引用继续明确标为缺口。
