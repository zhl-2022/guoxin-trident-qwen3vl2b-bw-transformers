# Qwen3-ForcedAligner-0.6B on BW + Transformers（session004 实测工程参照）

## 1. 简介

本轮已在 SCNet 完成固定 ForcedAligner 的真实 GPU 对齐：4.2039375 秒中文音频与已知文本输入，输出13个汉字的起止时间戳；开发契约通过，退出码0。**这是发布方 Transformers 路径的一次功能验证，TF-only计件资格、正式AAS和性能验收仍未闭合。**

模型 `Qwen/Qwen3-ForcedAligner-0.6B`，revision `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`；源码 QwenLM/Qwen3-ASR `7c6daf77a2421100f5fb066495372c00129d39ff`。见 [model-lock](../tasks/qwen3-forcedaligner-bw-transformers/model-lock.json)、[upstream-lock](../tasks/qwen3-forcedaligner-bw-transformers/upstream-lock.json)。输入已知文本是强制对齐的正常条件，并非ASR自由转写。

## 2. 验证环境

<!-- session004-platform-closure -->

本轮已独立核对[SCNet 平台关机与余额记录](../evidence/platform-checkpoint-20261023-004.json)：A/B 两个本轮实例及旧参考实例均为“已关机”；A 手动关机并选择保存环境，B 按预定时限自动关机。可见余额为 **488.0238888888 卡时**，与开机前可见余额差 4.6686111111 卡时；余额差可能包含延迟结算，不当作最终账单。资源为华中一区 A 区 113 组、BW64GB×1、CPU15核、内存59GB。账户镜像 ID `2071293281204580354` / version `2082674417745739777`，Notebook 可选择标签 `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`；完整实际 registry/repo:tag 和评审获取方式尚缺，digest 仅为可复现补充项。平台配置不替代任务原始环境/驱动实测。

本轮代码仓库目标：[zhl-2022/guoxin-qwen3-forcedaligner-bw-transformers 的 codex/session004 分支](https://github.com/zhl-2022/guoxin-qwen3-forcedaligner-bw-transformers/tree/codex/session004)。仓库保持私有；实际推送 commit 和逐文件回读以独立交付回执为准，评审权限仍须落实。光合既有三仓库的 session003 与本轮 GitHub 分支分别记录，不假称光合已同步新版。

| 项目 | 本轮原件记录 |
|---|---|
| 平台/精度 | SCNet BW64GB×1为平台配置；BF16、SDPA；本任务目录未单独导出hy-smi型号原文 |
| Python | 3.10.12；`/root/private_data/guoxin-candidates/envs/forcedaligner-001/bin/python` |
| checkpoint | `/root/private_data/guoxin-candidates/models/forcedaligner`；全部文件和完整权重SHA校验通过 |
| CPU依赖 | Transformers4.57.6、nagisa0.2.11、soynlp0.0.493、librosa0.11.0、soundfile0.13.1；accelerate1.12.0复用native |
| native | Torch2.10.0+das、vLLM0.18.1+das、DTK26.04；没有替换native GPU包 |
| 镜像 | 实际完整registry/repo:tag及digest尚缺；环境版本不能代替镜像身份 |

逐包版本/路径见 [environment](../evidence/session004/remote-intermediate-001/candidates/results/forcedaligner-pilot-001/environment.json)，身份见 [checkpoint核验](../evidence/session004/remote-intermediate-001/candidates/results/forcedaligner-pilot-001/checkpoint-verification.json)。初次安装超时后，partial修复只补audioread、复用42项，native metadata及原record不变。导入时发生GPU context discovery，原件如实记录，没有声称CPU-only。

## 3. 适配优化方案

使用固定 `Qwen3ForcedAligner.from_pretrained(...).align(...)`，保留发布方文本切分与时间修正规则，不改模型/tokenizer。CPU依赖在独立system-site-packages venv；系统无ensurepip，使用无pip venv，再由该venv Python调用可见pip。

真实分类头为 `[5000,1024]`，输入embedding为 `[152064,1024]`。官方在已有输入timestamp位置一次前向分类，再乘80ms并修正。标准vLLM ASR生成入口未被本轮证明支持该分支；ASR+TF aligner组合不等于aligner原生vLLM适配。见 [架构核查](forcedaligner-native-route-20261023.md)。

## 4. 服务启动脚本

当前是离线对齐runner，尚无已验证的SGLang/vLLM对齐服务标准入口。[run-start](../evidence/session004/remote-intermediate-001/candidates/results/forcedaligner-pilot-001/run-start.json)保存准确argv；run ID为 `ec698b4b-f378-4597-b236-bcd523bbb8ad`，上限360秒。未来复跑应使用新结果目录：

```bash
/root/private_data/guoxin-candidates/envs/forcedaligner-001/bin/python -B tasks/qwen3-forcedaligner-bw-transformers/entry.py run --model-dir /root/private_data/guoxin-candidates/models/forcedaligner --upstream-root /root/private_data/guoxin-candidates/source/qwen3-asr --requests tasks/qwen3-forcedaligner-bw-transformers/requests.example.json --output /root/private_data/guoxin-candidates/results/forcedaligner-pilot-NEW --max-seconds 360
```

执行helper SHA为 `8c7ec3978d59e2d8cf69e09fc647b6f2cb5fb1fa7685ee11cdf4e77048dc3bde`，与导出清单的执行文件hash相符。最终stage004新增文件及仓库SHA封存由总装配统一完成。

## 5. Smoke 测试

文本“甚至出现交易几乎停滞的情况。”、language=Chinese；音频SHA256 `46dbc998c9d1d48111267c40741dd3200f2e5bcf4075f8c4c97f4451160dce50`，来自固定发布方示例。[完整worker-result](../evidence/session004/remote-intermediate-001/candidates/results/forcedaligner-pilot-001/worker-result.json) SHA256 `c91746c4e7b826749418b407aaf2e46458af930ed382c70e416229ce8b394184`。

| 字 | 起点(s) | 终点(s) |
|---|---:|---:|
| 甚 | 0.400 | 0.720 |
| 至 | 0.720 | 0.960 |
| 出 | 0.960 | 1.120 |
| 现 | 1.120 | 1.520 |
| 交 | 1.520 | 1.760 |
| 易 | 1.760 | 2.000 |
| 几 | 2.000 | 2.240 |
| 乎 | 2.240 | 2.480 |
| 停 | 2.480 | 2.720 |
| 滞 | 2.720 | 2.880 |
| 的 | 2.880 | 3.040 |
| 情 | 3.040 | 3.360 |
| 况 | 3.360 | 3.680 |

13个有效文本单元保留，标点按官方规则不形成时间单元。时间戳有限、有序且在音频范围内，开发契约通过；没有独立标注判断各个边界是否准确。

## 6. 精度测评

`accuracy_against_independent_timestamps_evaluated=false`。尚缺赛事固定音频/文本、独立字/词timestamp reference、AASms定义/聚合、标点和语言切分规范、官方GPU baseline及原版evaluator/selfcheck。不能将单调性和范围检查换算成AAS或“较GPU基线劣化≤2%”通过。

没有正式verify结果，`official_accuracy_verified=false`；TF-only框架资格没有被功能成功自动解除。

## 7. 性能参考

单样本align调用为12.206611秒；run-final记录的worker监督阶段wall为72.990906秒，包含导入/模型准备等，未计入父进程此前的完整checkpoint hash。[run-final](../evidence/session004/remote-intermediate-001/candidates/results/forcedaligner-pilot-001/run-final.json)为completed、worker_exit_code=0，并记录仅清理自建进程组。

这是一次开发诊断，没有独立预热/重复、长音频/多语言/并发负载和有效峰值显存记录；不作为正式延迟、吞吐/RTF或加速比。对齐性能不能套用输出1024 token矩阵。

## 8. 注意事项

- 当前wrapper音频上限180秒；本轮只验证一个4.20秒中文例子。
- 真实GPU时间戳参考已保存；原生框架服务、正式精度/性能、可取镜像及干净复跑尚缺，`submission_ready=false`。
- 初次安装超时、修复原件及GPU discovery均保留，不能把修复终态改称首次CPU-only成功。
- 发布方Apache-2.0及新代码版权范围说明保留，不替用户选择全仓许可证。
- 原件来自11:30 UTC非原子中间导出；本成功run各文件hash已核对，快照不证明最终关机/费用终态。平台终态、最终封存和仓库访问由总交付记录闭合。
- 最后赛事表单由用户手动提交，本任务没有提交回执。
