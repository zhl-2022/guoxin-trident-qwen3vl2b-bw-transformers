# Trident-Qwen3VL-2B on BW + native vLLM（Session010 技术报告）

## 1. 简介

T02 · 已完成发布基线数值对照、完整原始证据独立复算，以及从已保存镜像新建实例的完整复现。技术投递材料已整理；主办方验收和最终赛事表单提交尚未发生。

代码：[光合 session010](https://developer.sourcefind.cn/codes/ghu_30ebea88e/guoxin-trident-qwen3vl2b-bw-transformers-session010)。镜像：[guoxin-publisher-eval-bw009](https://www.scnet.cn/ui/console/index.html#/image-management/detail/2114587646006431746)，版本 `dtk2604-vllm018-asr-trident-20261026`。模型固定 revision：`a5aedf8733f2107ab75b42214c78496d85759077`。源码、上游许可、失败轮次和原始报告均保留。

## 2. 验证环境

华中一区 A 区，BW 64 GB × 1、15 CPU、59 GB 内存；DTK 26.04、Ubuntu 22.04、Python 3.10。新实例实测 Torch 2.10.0、HIP 6.3.26093、vLLM 0.18.1+das.fa71803.dtk2604、Transformers 5.5.0。平台镜像继承的 PyTorch 2.9.0 标签不代表实际运行版本。

独立实例 ID `2114808632296972290`，容器 `crdnotebook-2114808632296972290-ac3p4mowhz-24051`；镜像 ID `2114587646006431746`。模型、环境、数据分别位于镜像的 `/opt/guoxin-trident009`。实例从镜像新建，完整运行日志记录于 UTC 2026-10-26T20:09:34.937352+00:00 至 2026-10-26T20:31:55.923907+00:00。

## 3. 适配优化方案

实际服务是原生 vLLM pooling HTTP。历史目录名含 transformers，不代表本轮使用独立 Transformers 服务。图像参数保持 min_pixels=262144、max_pixels=1310720、image_patch_size=16；评测器使用真实文件路径，修复 file URI 中逗号被转义后读取失败的问题。

本轮逐项核验原始仓库 1147 个清单文件；所有原始 Python / shell 文件与已测 session009 版本一致，见 `evidence/replay010/unchanged-serving-code.json`。

## 4. 服务启动脚本

在指定镜像中进入本仓库根目录执行，输出目录使用新的唯一名称：

```bash
source env.sh
RUN_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
bash env_check.sh
"$PYTHON_BIN" verify.py --selfcheck
bash run.sh --run-id "$RUN_ID"
bash smoke.sh
"$PYTHON_BIN" verify.py --output /root/verify-trident-new-unique
# Trident verify.py 已包含下表四轮性能；bench.sh 可再次执行同一完整流程。
bash stop.sh --run-id "$RUN_ID"
```

服务仅监听本实例回环地址；启动与停止使用同一运行标识。新实例实际执行的环境检查、自检、启动、smoke、全量评测和停止阶段均返回 0；ASR 另有独立 SSE 和 bench 阶段。复现驱动及逐阶段返回值保存在 `evidence/replay010/`。

## 5. Smoke 测试

新实例的单独 smoke 和全部真实 HTTP embedding 请求均成功，保留启动、服务日志、响应与清理记录。四轮性能由根 verify.py 的实际全量执行覆盖；bench.sh 是同一完整评测入口的薄封装。

完整准确率与四轮性能均零请求失败。原件 `results.zip` 中保存实际响应，未使用假模型或固定返回值。

## 6. 精度测评

完整 ChartQA Origin：1,250 query、1,509 document image，共 2,759 个 embedding 请求。

原始轮 NDCG@10 为 **69.930198298914**；独立新实例为 **69.858923604896**。发布基线 70.19，百分制 2 pt 下限 68.19；本轮也通过更严格的相对 2% 下限 **68.7862**（相对劣化 **0.471686%**）。

独立审计全部 2,759 个原始 2048 维向量、图像 SHA、矩阵、1,250 份稳定排序和二值相关性 NDCG，重新计算结果与保存结果一致。首轮文件名失败记录保留，不与成功重跑或本轮复现拼接。

只覆盖论文 Origin 中的 ChartQA 数据集；不把单一数据集称为跨数据集 Mix。评测数据固定 revision 为 15c41335843f8880e24778843b462234b6427e4f。

依据：[Trident 论文](https://arxiv.org/abs/2610.11816)、[OpenBMB 官方代码](https://github.com/OpenBMB/Trident)。独立核验结果为 `evidence/replay010/independent-audit.json`。评测器原始 JSON 中的 `submission_ready=false` 是评测阶段的保守范围字段，原样保留；本次完成的工程检查以 `DELIVERY_STATUS_010.json` 和新实例回执为准，不将它改写成主办方验收。

## 7. 性能参考

固定哈希选取 24 个完整输入，c1/c4 各两轮；共 96/96 请求成功。沿用运行前声明的第二轮作为主要对照，两轮原件全部保留。单位为 embedding 向量/秒，包含端到端 HTTP 开销。

| 并发 | 轮次 | 成功/请求 | 墙钟秒 | 速率 | p50 秒 | p95 秒 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 24/24 | 2.749294 | 8.729515 | 0.150304 | 0.191367 |
| 1 | 2 | 24/24 | 1.851568 | 12.961984 | 0.093263 | 0.120259 |
| 4 | 1 | 24/24 | 0.785775 | 30.543078 | 0.127995 | 0.182104 |
| 4 | 2 | 24/24 | 0.805278 | 29.803388 | 0.131172 | 0.185736 |

第二轮速率与原始轮比较：c1 变化 +0.389%; c4 变化 -2.656%，均在 20% 以内。该比较用于复现一致性，不作为已获得后续优化奖励的声明。

## 8. 注意事项

- 每个模型单独投递本报告及对应光合、镜像链接；当前没有最终赛事提交回执，也没有主办方验收回执。
- 两项代码库保持私有；账号内可访问与评审账号已授权是不同事实，评审访问尚未独立验证。
- 新实例完整复现已完成，服务已停止；本轮平台全部实例也已关机，证据见 `evidence/replay010/all-gpu-stopped.png`。
- 新编代码继续保留版权说明；上游许可保留，不擅自替换许可证。
- 原始 session009 资料以历史版本保存在 `provenance/frozen009/`；本报告及 `publisher-delivery-manifest010.json` 是本轮材料入口。
