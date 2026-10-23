# 超算互联网 × 光合开发者社区"国芯国模"黑客松（2026 Q4 赛季）· 技术报告模板

> 【】为占位符，提交前全部替换；阈值、容差、负载档位以任务池公示为准。
> 本报告作为提交仓库的 `README.md`，与 `env.sh`、`run.sh`、`smoke.sh`、`verify.py`、`bench.sh` 及其产出的原始文件一起提交。
> 报告中的版本号、精度分数、性能数字必须与 `env.log`、`verify_result.json`、`bench/round2.jsonl` 一致，原始文件不得删减。
> 红线：改权重 / 结构 / tokenizer / 采样语义规避计算、硬编码答案、伪造日志 → 取消资格。



````markdown
# 【模型名称】 on 【卡型】 + 【框架 版本】

> 示例标题：Kimi-Linear-48B-A3B-Instruct on BW1000 + SGLang 0.5.12

## 1. 简介

本文档记录 `【MODEL_ID】` 在 `【卡型】 + 【框架 版本】`（精度 `【bf16 / int8 / int4】`）环境的适配与验证结果。

【模型一句话介绍：参数量、架构特点（如 MoE / 线性注意力 / 视觉塔）】。

【与官方部署方式的核心差异，一两句话说清：如需升级 xx 包、打 xx patch、改用 xx 参数】。

相关获取地址：

- 权重下载（超算互联网 AI 社区 / 魔搭）：【URL】
- 镜像：`【镜像完整地址:tag】`
- 提交仓库：【URL】

参考文档：

- 【官方部署文档 / cookbook 链接】

仓库内容：

| 文件 | 作用 | 产出 |
| --- | --- | --- |
| `env.sh` | 模型、镜像、框架、卡型、并行方式与启动参数，其余脚本读这里的变量 | — |
| `env_check.sh` | 采集卡型与重点组件版本 | `env.log` |
| `requirements-extra.txt` / `patches/` | 在镜像基础上升级的包、代码改动（无改动则不放） | — |
| `run.sh` | 模型启动脚本：装依赖 + 重放 patch + 拉起服务，轮询 `/health` | `server.log` |
| `smoke.sh` | Smoke 测试：健康检查、单轮 / 多轮 / 流式问答 | `smoke.log` |
| `verify.py` | 精度验证脚本：按官方基线文件跑评测并判定 | `verify_result.json` |
| `bench.sh` | 性能测试脚本：6 档负载，每档连测两次 | `bench/` |

一键复现：

```bash
git clone 【URL】 && cd 【repo】
bash run.sh                             # 装依赖 + 重放 patch + 起服务（重复执行自动跳过已完成的步骤）
bash env_check.sh                       # 记录验证环境 → env.log
bash smoke.sh                           # Smoke 测试 → smoke.log
source env.sh && python verify.py       # 精度测评 → verify_result.json
bash bench.sh                           # 性能测试 → bench/
```

## 2. 验证环境

镜像：`【镜像完整地址:tag】`

重点组件（取自 `env.log`，在第 3 节升级过的包写升级后的版本，并在备注里注明原版本）：

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| DTK | `【26.04】` | 镜像自带 |
| `torch` | `【2.x.x+das.dtk2604】` | 镜像自带 |
| `triton` | `【版本】` | 镜像自带 |
| `【sglang / vllm】` | `【版本】` | 镜像自带 |
| `【sglang-das / vllm-plugin-das】` | `【版本】` | 镜像自带 |
| `transformers` | `【版本】` | 【镜像自带 / 由 x.x 升级】 |
| `【flash-attn / 模型特有依赖】` | `【版本】` | 【镜像自带 / 新装】 |

- 卡型：`【BW1000 / K100-AI】`
- 卡数：`【N】`（单机 TP=【N】 / 多机 【M】 节点 × TP=【N】）
- 模型路径：`【/root/private_data/...】`
- 服务端口：`【30000】`

环境自检（贴 `env.log` 关键部分原文）：

```text
image: 【镜像地址:tag】
dtk: 【版本】
python: 3.10.x
torch: 2.x.x
HIP: 6.3.xxxxx
Device: 【K100_AI】 x 【N】
【pip list 中的重点组件】
```

超算互联网平台（参赛算力统一使用超算互联网，必填）：

| 项 | 值 |
| --- | --- |
| 区域 | 【西南一区 / 其他】 |
| Notebook 规格 | 【CPU / 内存 / 系统盘 / 文件存储】 |
| 加速卡与显存 | 【K100_AI / BW1000】×【N】；【64GB / 其他】 |
| 开发镜像 | 【社区镜像名称与 tag】 |
| 模型获取方式 | 【AI 社区模型库克隆 / 下载】 |
| 服务映射 | 后端【127.0.0.1:30000】→ 代理【0.0.0.0:8001】→ 外部【https://...】（未使用可写「无」） |
| 平台证据 | `platform/`（Notebook 创建页、环境自检、模型管理、服务映射截图） |

## 3. 适配优化方案

> 写清在镜像基础上做了哪些软件包与代码层面的改动，**必须做到照着能复现**。
> 没有任何改动时，本节只写一句：「无代码改动，直接使用镜像内版本」。

### 3.1 基础镜像

基于 `【镜像完整地址:tag】`，【为什么选这个镜像，如：它是目前唯一带 sglang 0.5.12 的 DTK 26.04 镜像】。

### 3.2 软件包升级

| 包 | 镜像内版本 | 升级后版本 | 来源 | 原因 |
| --- | --- | --- | --- | --- |
| `【transformers】` | `【4.51.3】` | `【4.57.1】` | PyPI | 【模型 config 要求 ≥ 4.57】 |
| `【fla-core】` | 无 | `【0.4.1】` | PyPI | 【KDA kernel 依赖】 |

已写入 `requirements-extra.txt`，由 `run.sh` 以 `--no-deps` 安装（避免依赖解析把镜像里的 DCU 版 torch 换掉，连带依赖需逐个列出）：

```text
transformers==【4.57.1】   # 镜像内 【4.51.3】，【原因】
【tokenizers==0.22.1】     # transformers 的连带依赖
【fla-core==0.4.1】        # 新装，【原因】
```

DCU 专用包（whl）写明下载地址，或放入仓库并在 `requirements-extra.txt` 中写相对路径。

### 3.3 代码改动

每个 patch 写清四项：**改了哪个文件 / 改了什么 / 为什么改 / 是否影响精度**。

#### `patches/0001-【简述】.patch`

- 作用于：`【sglang/srt/layers/attention/fla/chunk.py】`（路径相对 `PATCH_ROOT`，默认为 site-packages）
- 改了什么：【如：HIP 平台下把 chunk 分块大小 BT 由 64 改为 32】
- 为什么改：【如：DCU 的 LDS 容量小于 A100 共享内存，BT=64 时 kernel 申请超限，报 `out of resource: shared memory`】
- 精度影响：【如：只改变分块大小，不改变计算结果；见第 6 节精度结果】

```diff
--- a/sglang/srt/layers/attention/fla/chunk.py
+++ b/sglang/srt/layers/attention/fla/chunk.py
@@ -88,3 +88,4 @@ def chunk_kda_fwd(q, k, v, g, beta, scale, initial_state):
     B, T, H, K = q.shape
-    BT = 64
+    # DCU 的 LDS 容量小于 A100 共享内存，BT=64 时 kernel 申请超限
+    BT = 32 if is_hip() else 64
     NT = triton.cdiv(T, BT)
```

【有多个 patch 时按序号依次列出】

### 3.4 升级步骤

`run.sh` 启动前会自动完成以下步骤（重复执行时跳过已装的包和已打过的 patch）；手动复现时按顺序执行：

```bash
docker pull 【镜像完整地址:tag】                     # 或在超算互联网 Notebook 中选择该镜像
pip install --no-deps -r requirements-extra.txt     # 3.2 软件包升级
PATCH_ROOT=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
for p in patches/*.patch; do patch -p1 -d "$PATCH_ROOT" < "$p"; done   # 3.3 代码改动
【其他必要步骤，如 kernel 预编译：python -c "from fla.ops import chunk_kda"】
```

## 4. 服务启动脚本

`env.sh` 中与启动相关的配置：

```bash
export MODEL_PATH="【/root/private_data/models/xxx】"
export SERVED_NAME="【xxx】"
export FRAMEWORK="【sglang / vllm】"
export PORT=【30000】
export TP=【4】
export EP=【1】
export PP=【1】
export MAX_LEN=【32768】       # 须容纳 16K 输入 + 1K 输出的压测
export MEM_FRAC=【0.85】
export EXTRA_ARGS="【--reasoning-parser xxx --tool-call-parser xxx ...】"
【必要的环境变量，如 export HIP_VISIBLE_DEVICES=0,1,2,3】
```

启动：

```bash
bash run.sh
```

`run.sh` 展开后实际执行的完整命令（取自 `server.log` 首行）：

```bash
python -m sglang.launch_server \
  --model-path 【/root/private_data/models/xxx】 \
  --served-model-name 【xxx】 \
  --tp-size 【4】 --ep-size 1 --pp-size 1 \
  --context-length 【32768】 \
  --mem-fraction-static 【0.85】 \
  --host 127.0.0.1 --port 【30000】 \
  --trust-remote-code \
  【EXTRA_ARGS 展开的参数】
```

重点参数说明（只列与默认值不同、或对能否跑通 / 性能有影响的参数）：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--tp-size` | `【4】` | 【权重 98 GB，单卡 64 GB，需 4 卡切分】 |
| `--context-length` | `【32768】` | 【覆盖 16K + 1K 压测；开到模型最大长度时 KV cache 不足】 |
| `--mem-fraction-static` | `【0.85】` | 【0.9 时 64 并发下 OOM】 |
| `【--reasoning-parser】` | `【xxx】` | 【思考内容从 reasoning_content 返回】 |
| `【图模式 / 调度参数】` | `【xxx】` | 【说明】 |
| `【环境变量】` | `【xxx】` | 【说明】 |

多机部署时补充组网方式：节点数、各节点 `NODE_RANK`、主节点 `DIST_ADDR`（须为其他节点可达的 IP）。

## 5. Smoke 测试

```bash
bash smoke.sh
```

`smoke.sh` 依次执行以下检查（任一接口非 2xx 即失败）：

```bash
# 健康检查
curl -sf http://127.0.0.1:【30000】/health -o /dev/null -w "HTTP %{http_code}\n"

# 模型列表
curl -sf http://127.0.0.1:【30000】/v1/models

# 简单问答
curl -sf http://127.0.0.1:【30000】/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "【SERVED_NAME】",
    "messages": [{"role": "user", "content": "用一句话介绍 DCU。"}],
    "temperature": 0,
    "max_tokens": 256
  }'

# 另有多轮（第三轮问「我叫什么名字？」）与流式（"stream": true），见 smoke.sh
```

验证结果（贴 `smoke.log` 关键部分）：

- `/health` 返回 `200`
- `/v1/models` 返回 `【SERVED_NAME】`
- 单轮问答回复：「【模型实际输出】」
- 多轮：正确回答「小海」
- 流式：逐 token 返回，以 `data: [DONE]` 结束

## 6. 精度测评

精度验证脚本 `verify.py` 按官方基线文件 `baselines/【MODEL_ID 中 / 换成 __】.json` 写死的配置调用服务跑评测，与官方 GPU 基线对比，较基线劣化 ≤2% 判 PASS（百分制即 2 pt），提升不罚。

```bash
python verify.py --selfcheck            # 先确认判分逻辑未被改动
source env.sh && python verify.py       # 跑评测，结果写入 verify_result.json
```

评测配置（取自官方基线文件，参赛者不改）：

| 数据集 | 子集题数 | max_tokens | 解码 | 思考开关 | 判分方式 |
| --- | --- | --- | --- | --- | --- |
| GSM8K | `【500】` | `【2048】` | greedy（temperature=0） | `【关闭】` | 取 `\boxed{}` 或末尾数字，与答案数值相等 |
| MMLU | `【1000】` | `【32】` | greedy（temperature=0） | `【关闭】` | 取首个 A–D 选项字母 |

结果（`verify_result.json` 原文）：

```json
{
  "MODEL_ID": "【MODEL_ID】",
  "CARD": "【卡型】",
  "FRAMEWORK": "【框架】",
  "PRECISION": "【精度】",
  "detail": {
    "gsm8k": {"got": 【】, "ref": 【】, "worse": 【】, "tol": 2.0, "pass": 【true】},
    "mmlu":  {"got": 【】, "ref": 【】, "worse": 【】, "tol": 2.0, "pass": 【true】}
  },
  "passed": 【true】
}
```

| 数据集 | 官方 GPU 基线 | 本次提交 | 劣化 | 容差 | 判定 |
| --- | --- | --- | --- | --- | --- |
| GSM8K | 【】 | 【】 | 【】 pt | ≤2 pt | 【PASS】 |
| MMLU | 【】 | 【】 | 【】 pt | ≤2 pt | 【PASS】 |

结论：【PASS / FAIL；未通过直接写原因】

> T02–T04 的评测器、数据集与判定口径随分题材料下发，表格列按实际指标调整。

## 7. 性能参考

性能测试脚本 `bench.sh` 使用 SGLang 自带压测客户端（SGLang / vLLM 服务通用），随机 token 输入、固定输入输出长度，负载如下：

| 输入 | 输出 | 并发 |
| --- | --- | --- |
| 4K（4096） | 1K（1024） | 1 / 8 / 64 |
| 16K（16384） | 1K（1024） | 1 / 8 / 64 |

每档连测两次，第一次预热，**取第二次数据**；请求数为并发的 2 倍（至少 8 个）。

```bash
bash bench.sh                           # 文本模型
BENCH_MODE=image bash bench.sh          # 多模态理解模型，每个请求带 1 张图
```

单档命令（`bench.sh` 内部对 6 档 × 2 次循环执行）：

```bash
python -m sglang.benchmark.serving \
  --backend 【sglang / vllm】 --host 127.0.0.1 --port 【30000】 \
  --model 【SERVED_NAME】 --tokenizer 【MODEL_PATH】 \
  --dataset-name random-ids \
  --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 \
  --max-concurrency 8 --num-prompts 16 --seed 1 \
  --output-file bench/round2.jsonl
```

第二次结果（`bench/summary.md` 原文）：

| random_input_len | random_output_len | max_concurrency | completed | duration | request_throughput | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 1 | | | | | | | | | | |
| 4096 | 1024 | 8 | | | | | | | | | | |
| 4096 | 1024 | 64 | | | | | | | | | | |
| 16384 | 1024 | 1 | | | | | | | | | | |
| 16384 | 1024 | 8 | | | | | | | | | | |
| 16384 | 1024 | 64 | | | | | | | | | | |

- 原始数据：`bench/round1.jsonl`、`bench/round2.jsonl`、各档日志 `bench/round*_*.log`
- 本结果作为该件「模型 × 卡型 × 框架 × 精度」的性能基准；后续优化在同卡型、同框架、同负载下对比
- T03（生成类）/ T04（语音类）的性能测法后续随分题材料提供

## 8. 注意事项

> 把特殊的地方写清楚；主动说明的问题不扣分，评审复跑时才发现的会打回。

- **不支持的负载**：【如：16K 输入 × 64 并发时 KV cache 不足，请求排队导致超时，该档未出数据，见 `bench/round2_16384_1024_c64.log`】
- **修改的模型参数**：【改了哪个参数（如 `config.json` 的 `max_position_embeddings`、`generation_config.json`）、改成多少、为什么改；并说明不影响精度和性能的依据（如第 6 节精度结果）】
- **已知限制**：【如：暂不支持 xx 特性 / 多图输入上限 xx 张】
- **踩坑与规避**：【报错现象（关键报错原文）→ 原因 → 处理方法】
- **其他**：【如：首次启动需 JIT 编译 kernel 约 x 分钟，`LOAD_TIMEOUT` 已调为 xx 秒】
````
