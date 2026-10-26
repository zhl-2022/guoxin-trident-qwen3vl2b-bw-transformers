# Trident-Qwen3VL-2B on BW + native vLLM（Session009 发布基线对照）

## 1. 简介

T02 · 本轮完整选定公开评测集的数值对照及独立复算已通过，SCNet 适配环境镜像已保存。最终镜像独立新实例复跑、当前光合版本同步及评审访问尚未闭合；`submission_ready=false`、`official_acceptance=false`、`submitted=false`。当前表单关闭，本报告没有被上传到赛事表单。

权重和固定 revision 见 `evidence/publisher009/SOURCE.json`；原始源码与许可保留。镜像：`guoxin-publisher-eval-bw009:dtk2604-vllm018-asr-trident-20261026`，[SCNet 账号内详情](https://www.scnet.cn/ui/console/index.html#/image-management/detail/2114587646006431746)。当前源码备份采用 GitHub 私有分支 `codex/publisher-eval009`；光合 session007 仓库仍是历史版本，不能冒充本轮内容。

## 2. 验证环境

超算互联网华中一区 A 区，BW 64 GB × 1，15 CPU、59 GB 内存；DTK 26.04、Ubuntu 22.04、Python 3.10。实测共享系统 Torch 2.10.0、HIP 6.3.26093、native vLLM 0.18.1+das.fa71803.dtk2604、vLLM-HCU 0.18.1+das.dtk2604。ASR 的独立音频环境使用 Transformers 5.5.0、soundfile 0.13.1；Trident 的完整具体日志以其原件 ZIP 为准。平台镜像页面的继承 PyTorch 2.9.0 标签不是运行时版本证明。

模型、环境、数据均在 `/opt/guoxin-trident009`。服务只绑定 127.0.0.1，未新增公网模型服务。镜像大小 64.35 GB，平台显示华中一区 A 区可用；账号内可见不代表评审账号已有访问权限。

## 3. 适配优化方案

使用原生 vLLM pooling HTTP 服务。首轮只在 `d_264` 文件名含逗号时失败：`Path.as_uri()` 转为 `%2C`，qwen-vl-utils 的 file URI 读取未 URL 解码。修订评测器改传真实文件系统路径，保持 min_pixels=262144、max_pixels=1310720、image_patch_size=16 不变。修订后从头重跑全部 2,759 请求，没有删样本或拼接首轮成绩。两轮原件均保留。

`scripts/submission009/` 保存真实运行的评测工具；原始工具包和运行清单 SHA 对应。仓库根入口是本轮新增薄封装，CPU selfcheck 与 shell 语法另行检查；没有把这些后加入口记为本轮已在 GPU 执行。

## 4. 服务启动脚本

在上述最终镜像的 Linux 环境中：

```bash
bash run.sh --run-id your_unique_run_id
bash env_check.sh
bash smoke.sh
python verify.py --selfcheck
source env.sh
"$PYTHON_BIN" verify.py --output /root/verify-new-unique-directory
bash bench.sh
bash stop.sh --run-id your_unique_run_id
```

输出目录必须全新，脚本拒绝覆盖原始结果。ASR 启动限制 9,900 秒、加载限制 900 秒；Trident 限制 5,400 秒、加载限制 600 秒。单卡两个模型串行启动。ASR 使用 BF16、TP1、32768 context、0.85 memory、eager、max-num-seqs4；Trident 0.45 memory，完整启动 argv 在原件 `launch.log`。这些设置是已执行配置，不把模板中的六档随机 token 负载伪装成已完成。

## 5. Smoke 测试

原仓库的历史 smoke 证据随原包保留。本轮全量评测均走真实 HTTP 服务，全部请求成功，记录启动、清理、原始响应和独立核验。ASR 两轮性能前有独立 warmup 原件；Trident 在全量评测后测性能。新根 `smoke.sh` 调用镜像中的同一原始任务 smoke，未增加伪造日志或假模型。

## 6. 精度测评

完整 OpenBMB/Trident-Eval ChartQA Origin：1,250 条查询、1,509 张文档图，共 2,759 个 HTTP embedding 请求。

NDCG@10 为 **69.93019829891405**，论文表 1 发布基线 **70.19**，差异约 -0.259802 个百分点，超过下限 68.19。独立核验全部 2,759 个原始 2048 维向量、图像 SHA、矩阵、1,250 份稳定排序与二值相关性 NDCG，复算 69.93019829891416。浮点求和末位差异约 1.1e-13。

数据固定 revision 为 `15c41335843f8880e24778843b462234b6427e4f`。这是 Origin 单数据集结果；论文 Mix 65.82 涉及跨数据集混合，不作 Mix 完成声明。


`verify_result.json` 是实际原件副本。完整原始结果在 `evidence/session009/` 的 ZIP 中，含逐请求响应与失败轮次。ZIP 不删减 payload；Trident 原始 JSONL 超过 GitHub 单文件上限，故保留压缩原件并记录 SHA。独立审计见 `evidence/session009/trident-independent-audit002.json`。数值通过不等于主办方正式验收。

## 7. 性能参考

固定输入以预声明哈希选取 24 个完整请求，c1/c4 各两轮；全部 96/96 成功。第二轮为主要对照，第一轮同样保留。速率单位：embedding 向量/秒。以下为端到端 HTTP 实测，不冒充生成 token/s 或 GPU-only kernel 时间。

| 并发 | 轮次 | 成功/请求 | 总墙钟秒 | 速率 | p50 秒 | p95 秒 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 24/24 | 2.779970 | 8.633186 | 0.159377 | 0.181051 |
| 1 | 2 | 24/24 | 1.858775 | 12.911730 | 0.094835 | 0.120617 |
| 4 | 1 | 24/24 | 0.805696 | 29.787896 | 0.123600 | 0.186752 |
| 4 | 2 | 24/24 | 0.783887 | 30.616677 | 0.114602 | 0.206387 |

原始汇总见 `bench009/summary.json`，逐请求在原始 ZIP。Trident 的冻结评测器集成准确率与性能；bench.sh 会重新完整评测后跑四组性能，不复用旧分数。 语音/嵌入检索的赛道性能适用口径仍应由评审确认，不宣称已完成文本六档负载。

## 8. 注意事项

- 最终表单当前“暂不支持填写”；没有提交回执。
- 镜像保存与同一 Notebook 恢复已观察；尚未从最终镜像新建独立 GPU 实例完整复跑。
- GitHub 当前备份保持私有；本轮新增结果还需同步到光合，并核对评审获取方式。
- 新根入口仅是复现封装，其完整 GPU 重跑尚未执行；真实执行脚本 SHA 与原件另列，不能混淆。
- 仅报告 ChartQA Origin；不把 Mix 或其他检索基准写成已完成。首轮文件名失败原件保留。
- 新编适配代码仍按用户要求保留版权说明；上游许可不变，不替用户新选开放许可。
