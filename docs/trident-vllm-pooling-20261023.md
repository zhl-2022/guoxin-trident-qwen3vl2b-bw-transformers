# Trident 的 vLLM pooling 候选核查与执行

## 当前结论

Trident-Qwen3VL-2B 存在有源码和真实 checkpoint 元数据支持的 vLLM 0.18.1 embedding 转换路线。已新增独立 probe、配置与 CPU 契约测试；没有改动既有 Transformers runner，没有改 checkpoint，也没有把静态兼容判断写成 GPU 通过或赛事通过。

本轮只从固定权重文件读取 76,248 B safetensors 长度与 JSON 头，未读取 tensor payload；元数据缓存于 `evidence/trident-vllm-probe-20261023/`。Windows 与 WSL 的 14 项契约测试分别通过，GPU 运行、数值对照与官方评测待执行。

## 已核实的映射

| 项目 | 固定 Trident 实际内容 | vLLM 0.18.1 对应行为 |
|---|---|---|
| 模型身份 | `openbmb/Trident-Qwen3VL-2B`，revision `a5aedf8733f2107ab75b42214c78496d85759077` | 仍加载同一份本地权重和 tokenizer |
| 架构名称 | `Qwen3VLForEmbedding`，没有注册在 vLLM 原生 registry | 仅在运行配置中用 `hf_overrides` 指向 `Qwen3VLForConditionalGeneration` |
| 配置内容 | 与官方 Qwen3-VL-Embedding-2B 当前 config 的区别仅为 `architectures` 和顶层 `use_cache` | 采用已有 Qwen3-VL 实现；不改尺寸、RoPE、token ID 或 DeepStack 配置 |
| 权重 | 625 个 BF16 tensor；仅 `model.language_model.*` 与 `model.visual.*`；没有额外 projection、adapter 或 lm_head | Qwen3-VL mapper 分别映射至语言与视觉模块；embed conversion 跳过生成头 |
| 向量维度 | hidden size 2048，`embed_dim=null` | 2048 维完整向量，不做 Matryoshka 截断 |
| 池化 | `pooling=last_token` | `runner=pooling`，`convert=embed`，`seq_pooling_type=LAST` |
| 归一化 | 先转 FP32，再 L2 normalize | `head_dtype=float32`，embedding pooler 的 `use_activation=true` 调用 L2 normalize |
| 输入 | query/doc 有不同 system 指令；图像先于文本；保留 assistant generation prompt 的最后位置 | 从固定 `mmemb_model.json` 读取指令，调用原 checkpoint AutoProcessor，逐项保存 prompt 和 token IDs |
| 图像 | `min_pixels=262144`、`max_pixels=1310720`，patch size 16 | 固定 qwen-vl-utils 0.0.14 的官方 vision 处理，再向 vLLM 传 PIL；`do_resize=false` 避免二次缩放 |

原始 Trident 模型定义明确使用标准 Qwen3VLModel，未增加 decoder 或 projection 层；发布方 backend 从 checkpoint 的 `mmemb_model.json` 建立 encoder。来源：[固定 Trident 源码](https://github.com/OpenBMB/Trident/tree/7fbf4b2033e1b3a43a99185fc125d765a549c52d)、[固定模型 config](https://huggingface.co/openbmb/Trident-Qwen3VL-2B/blob/a5aedf8733f2107ab75b42214c78496d85759077/config.json)、[固定 encoder 配置](https://huggingface.co/openbmb/Trident-Qwen3VL-2B/blob/a5aedf8733f2107ab75b42214c78496d85759077/mmemb_model.json)。官方 Qwen config 的 `main` 是本轮捕获的对照资料，不替代 Trident 的固定 revision 身份。

vLLM 支持表列有经 conversion 的 Qwen3-VL embedding，源码中的 mapper 与 pooler 支持上述转换。来源：[v0.18.1 支持表](https://docs.vllm.ai/en/v0.18.1/models/supported_models/#embedding)、[pooling 配置](https://docs.vllm.ai/en/v0.18.1/models/pooling_models/)、[Qwen3-VL loader](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/models/qwen3_vl.py)、[conversion adapter](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/models/adapters.py)、[FP32 pooling head](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/layers/pooler/seqwise/heads.py)、[embedding normalization](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/layers/pooler/seqwise/poolers.py)。

## 独立执行命令

下载完成并核验后，先在原生 base Python 中准备 Trident native-first 的独立 CPU 闭包。该环境复用平台 Torch/vLLM/DTK，不把 Transformers 4.57.1 安装进原生环境。不要用 publisher-pinned 的 TF4.57.1 环境启动本 vLLM probe。

```bash
python3 -B /root/private_data/guoxin-candidate-tools-20261023/scripts/bootstrap_candidate_envs_20261023.py apply --task trident --profile native-first --label 001 --max-seconds 900
```

将本 probe 补充 ZIP 解压到独立工具根目录后执行：

```bash
/root/private_data/guoxin-candidates/envs/trident-001/bin/python -B /root/private_data/guoxin-candidate-tools-20261023/scripts/trident_vllm_pooling_probe_20261023.py preflight --model-dir /root/private_data/guoxin-candidates/models/trident --hash-weights

/root/private_data/guoxin-candidates/envs/trident-001/bin/python -B /root/private_data/guoxin-candidate-tools-20261023/scripts/trident_vllm_pooling_probe_20261023.py run --model-dir /root/private_data/guoxin-candidates/models/trident --output /root/private_data/guoxin-candidates/results/trident-vllm-pooling-001 --max-seconds 600
```

若同一组输入已取得发布方 TF runner 的成功结果，可在新 output 目录增加参数：

```bash
--reference-result /root/private_data/guoxin-candidates/results/trident-tf-pilot-001/worker-result.json
```

这里的 reference 是工程数值对照。脚本要求 reference 为 `publisher_embedding_backend` 并且输入文本、图像本地绝对路径完全一致。它保存 reference 的一次读取快照、SHA256、checkpoint/run-start/environment/run-final 等已有 sidecar 的 SHA 和内容；若 checkpoint sidecar 存在但模型/revision 不匹配则拒绝。原 TF runner 未在产出时保存图片 SHA 和独立源码核查原件，因此这两项明确记为 unverified；相同文件路径不能证明当时字节相同，不能将这个参考直接称为来源完整的精度基线。每个向量保存 cosine 与最大绝对差异，工程 cosine 阈值为 0.999；这个自定阈值不是赛事精度线。未提供 reference 时，结果明确记录未作数值对照。

probe 分别覆盖文本 query、文本 document、图像 document、图像加文本 document。逐项比较 vLLM 实际 prompt token IDs 与同一官方 processor 的展开 token IDs，任何不同均使开发契约失败，不静默忽略。该 pilot 没有实现视频、任意远程图片、服务端 API 或完整 benchmark；输入超长或格式超出范围时拒绝执行。输出目录必须新建，父进程将完整权重 hash、语义/输入预检、依赖导入及 GPU worker 一并纳入时限，并仅清理自建进程组。失败保存结构化 checkpoint、worker error 和 run-final。单独的 `preflight` 是显式只读检查，不受 `run --max-seconds` 的子进程时限控制。没有 pip 安装动作。

## ForcedAligner 共用工程与工具根的命令

工具在独立根 `/root/private_data/guoxin-candidate-tools-20261023`，原 task 和固定源码 ZIP 在 session004 工程根。不要将两个根混为同一路径。

```bash
python3 -B /root/private_data/guoxin-candidate-tools-20261023/scripts/bootstrap_candidate_envs_20261023.py apply --task forcedaligner --label 001 --max-seconds 900
```

随后在 session004 工程根运行：

```bash
python3 -B tasks/qwen3-forcedaligner-bw-transformers/entry.py prepare-source --upstream-root /root/private_data/guoxin-candidates/source/qwen3-asr

/root/private_data/guoxin-candidates/envs/forcedaligner-001/bin/python -B tasks/qwen3-forcedaligner-bw-transformers/entry.py run --model-dir /root/private_data/guoxin-candidates/models/forcedaligner --upstream-root /root/private_data/guoxin-candidates/source/qwen3-asr --requests tasks/qwen3-forcedaligner-bw-transformers/requests.example.json --output /root/private_data/guoxin-candidates/results/forcedaligner-pilot-001 --max-seconds 600
```

源码根已存在并且身份已验证时，复用即可；`prepare-source` 不覆盖已有目录。bootstrap 固定 `with_pip=False`，因为 SCNet 的系统 Python 没有 ensurepip；它使用新 venv Python 调用系统可见 pip，不改变 native 包。

## 尚未闭合的条件

- GPU 需验证平台 vLLM 分支实际包含这些上游接口、完整权重成功加载、图像处理行为和 token 展开一致。
- 需要 TF 对照与公开 DocVQA 或赛事固定子集的完整检索结果，不能用一个合成图片的排序充当精度成绩。
- T02 明确为 SGLang/vLLM，embedding 在范围内；本转换是合规框架的候选工程路径，但尚未完成赛事验收。
- T04 的 TF pilot 仍未获得 TF-only 交付资格的明确依据；不能因为能输出时间戳就直接计正式完成件数。
- 缺官方 evaluator、固定数据、GPU baseline 时，所有 `official_accuracy_verified`、`official_performance_verified` 与 `submission_ready` 保持 false。

本轮保留原始许可：Trident/MIT；上游 vLLM/Apache-2.0。新编脚本只有版权范围说明，没有替用户选择全仓开源许可证。
