# RoboBrain2.5-8B-NV on BW + vLLM（session004 实测报告）

## 1. 简介

`BAAI/RoboBrain2.5-8B-NV` 已在本轮 BW、BF16、native vLLM 上使用受源码约束的 head72 VisionMathCompat 完成七项功能检查。七项包含 health/models，实际生成请求为五次；不将它们当作动作精度或正式性能结果。`formal_ready=false`、`official_accuracy=false`、`official_performance=false`。

权重：[发布方模型](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV)。历史私有仓库：[光合](https://developer.sourcefind.cn/codes/ghu_30ebea88e/guoxin-robobrain-bw-vllm)、[GitHub](https://github.com/zhl-2022/guoxin-robobrain-bw-vllm)。本轮版本同步和评审访问待最终核对。已读取 [remote-intermediate-001 manifest](../evidence/session004/remote-intermediate-001/export-manifest.json)，SHA `2b1543b7bb12c95f3a0aa041e8e3db605f6189921f2819316c5fcfc7ea86ecb2`；原件保留逐文件 hash、CRC 与实际执行脚本前后 SHA。

## 2. 验证环境

<!-- session004-platform-closure -->

本轮已独立核对[SCNet 平台关机与余额记录](../evidence/platform-checkpoint-20261023-004.json)：A/B 两个本轮实例及旧参考实例均为“已关机”；A 手动关机并选择保存环境，B 按预定时限自动关机。可见余额为 **488.0238888888 卡时**，与开机前可见余额差 4.6686111111 卡时；余额差可能包含延迟结算，不当作最终账单。资源为华中一区 A 区 113 组、BW64GB×1、CPU15核、内存59GB。账户镜像 ID `2071293281204580354` / version `2082674417745739777`，Notebook 可选择标签 `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`；完整实际 registry/repo:tag 和评审获取方式尚缺，digest 仅为可复现补充项。平台配置不替代任务原始环境/驱动实测。

本轮代码仓库目标：[zhl-2022/guoxin-robobrain-bw-vllm 的 codex/session004 分支](https://github.com/zhl-2022/guoxin-robobrain-bw-vllm/tree/codex/session004)。仓库保持私有；实际推送 commit 和逐文件回读以独立交付回执为准，评审权限仍须落实。光合既有三仓库的 session003 与本轮 GitHub 分支分别记录，不假称光合已同步新版。

| 项目 | 本轮实证或缺口 |
|---|---|
| 平台 | 根代理 UI 关联实例B `2112237945233408001`、SCNet华中一区A区、BW×1；UI映射不冒充日志自行采集的设备 probe |
| 服务与模型 | `/root/private_data/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV`；`robobrain2-5-8b-nv`；`127.0.0.1:30001` |
| 解释器/框架 | `/usr/bin/python`；server 初始化日志为 vLLM0.18.1，实际 worker 为 `HcuGPUWorker`；本题没有单独新 env.log，完整包路径版本可参考同实例ASR prepare，但不伪称为Robo独立采样 |
| 配置 | BF16、TP1、32768上下文、显存比例0.85、eager、maxseq4、每请求最多2图/0视频、1024×1024初始化 profiling bound |
| 镜像 | 平台展示名已知，最终 registry/repo:tag、获取方式、干净镜像复跑和完整本题设备 probe 仍缺 |

实际 [config](../evidence/session004/remote-intermediate-001/project/tasks/robobrain-bw-vllm/runtime/config-68f2ba449dbd41629b40cca27aa07553.json) 与 [server.log](../evidence/session004/remote-intermediate-001/project/tasks/robobrain-bw-vllm/runtime/server.log) 为准。profile bound 不强制真实输入缩放为1024，也不能保证更大图片可用；KV cache容量和启动日志里的瞬时吞吐不作为性能成绩。

## 3. 适配优化方案

兼容代码对目标 head72 vision SDPA 执行有 score 元素上界的分块 Math，保留逐图 cu boundaries、mask、scale、广播/GQA语义；不改权重/tokenizer，不关闭视觉输入。实际执行源码 SHA `083dc53f3e4f8171b774469bffa724c2269a9d289769f56a9d7a5624cd81659f` 与既有已测版本一致，manifest 前后 seal 及 server 内 `ROBOBRAIN_VISION_MATH_COMPAT installed` 事件共同记录安装；chunk128、score budget16777216。

历史003的24个不同GPU数值case、两次相同集重复检查、合成框IoU与81次Math调用均保留在 [003报告](session003-robobrain-results.md)，未计作本轮新数值实验。本轮是五个新功能生成请求的复验；没有新的数值对照、官方深度/轨迹参考或独立 kernel 根因实验。旧baseline错误和截断记录继续保留，不用当前成功覆盖。

## 4. 服务启动脚本

canonical [run.sh](../tasks/robobrain-bw-vllm/run.sh) 读取 [env.sh](../tasks/robobrain-bw-vllm/env.sh)，默认启用 `robobrain_vision_math_compat.VisionMathCompat`；`run_compat.sh` 为同一路径，`run_baseline.sh` 是独立原生对照。`ROBOBRAIN_*` 覆盖变量隔离，正式 evaluator/baseline/data/args/output contract 不继承其他任务。

```bash
bash tasks/robobrain-bw-vllm/run.sh --dry-run
ROBOBRAIN_PYTHON_BIN=/usr/bin/python ROBOBRAIN_MAX_RUNTIME_SECONDS=1800 bash tasks/robobrain-bw-vllm/run.sh
bash tasks/robobrain-bw-vllm/smoke.sh
bash tasks/robobrain-bw-vllm/verify.sh --selfcheck
bash tasks/robobrain-bw-vllm/stop.sh
```

服务 run ID `68f2ba449dbd41629b40cca27aa07553`，创建 `10:13:15.079736 UTC`，ready `10:15:51.060057 UTC`，`10:25:49.335472 UTC` 因 stop requested 进入 **stopped**，elapsed754.255秒。[终态](../evidence/session004/remote-intermediate-001/project/evidence/session004/robo-state-stopped.json) 与 [stop输出](../evidence/session004/remote-intermediate-001/project/evidence/session004/robo-stop.log) 对应同一服务；stop输出确认匹配监督器和 server process group。该记录不等于另一次独立的端口/后代进程 probe，也不证明 Notebook 计费已关闭。

## 5. Smoke 测试

[本轮原始JSONL](../evidence/session004/remote-intermediate-001/project/evidence/session004/robo-default-smoke-004.log) 为9条记录，客户端run ID `b3a24318692c49229b47ff3c55286bec` 与服务run ID分开。输入为固定红方块与蓝圆，SHA分别为 `d51437c5bb459fb25f47a9e52e650b3f815d1d9900ec63b651f338e6e73c7165`、`13588c097bf637a09c571e441392c40f406b81ca0a50e30b3ee2f336585fec2f`，来源见 [fixture manifest](../evidence/session004/remote-intermediate-001/project/fixtures/smoke-session004/manifest.json)。

| 检查 | 实际结果 | 单次用时 |
|---|---|---:|
| health / models | 两项通过，served name匹配 | 不计生成耗时 |
| single_image | 非空、stop、回答日志未裁剪 | 1.490392秒 |
| multiple_images | 非空、stop、回答日志未裁剪 | 6.735136秒 |
| multiturn_first | 非空、stop | 0.660714秒 |
| multiturn_followup | 带前轮上下文，非空、stop | 0.802132秒 |
| streaming_image | 非空、stop、DONE、无stream_errors | 0.667065秒 |

summary `functional_checks_passed=true`，七项均通过。流式原文为“图像中主要物体是一个位于右侧的红色正方形。”。该客户端只检查协议、非空与完整结束，不按答案正确率或动作/坐标容差判分；不要把旧003合成框IoU挪成本轮成绩。脚本SHA `bec36fb0c85332eb7144c479e9782d57c9b0dc2ad15b610ac81d1af2bd237486` 由导出manifest封存。

## 6. 精度测评

正式具身/VLA精度未执行。需与本题绑定的固定输入（含实际要求的图像/深度）、GPU动作/坐标reference、解析器/单位/容差、原版evaluator/selfcheck与判分公式。通用98%一致性条款或合成图输出不能拼成自制官方评分器。

`verify --selfcheck` 仅执行真实外部selfcheck，不跑full eval；缺材料返回2，未知schema为 `executed_uninterpreted`，不升级为PASS。当前正式分数、baseline、`verify_result.json`均留空，见 [本轮规则与缺项](session004-requirements-and-gaps.md) 和 [字段表](stage004-report-fields.json)。

## 7. 性能参考

本题本轮**没有执行正式或模板形状的SGLang性能矩阵**。当前正在其他服务上执行的Recon矩阵不属于Robo证据，不能挪用。上表五个短请求耗时、加载时间和服务生命周期只作开发观测，没有正式吞吐、TTFT、ITL、稳态预热或两轮第二轮取数成绩。

如果后续执行4096/16384输入、1024输出、并发1/8/64两轮，必须另存本题实际命令、图像策略、来源、warmup、完整成功/失败数和raw文件，并确认分题适用性。当前全部性能档位为未执行，不能补零值或计算加速比。

## 8. 注意事项

- 本轮仅确认当前源码实际安装和固定合成图功能协议；历史数值、当前功能与正式语义精度分别记载。
- 默认1024 profiling 已实跑，其他profile、更大真实输入、深度/轨迹和正式性能仍未证明。
- 模板允许如实披露未完成负载；性能矩阵未完成不自动判整题不可提交，真实官方材料与镜像复现缺项继续列出。
- 服务 stopped 与独立清理核验、平台停机分别核对。当前原件没有新的平台终态截图，不从历史003推断。
- 保留版权范围和上游许可，不增加全仓开源门槛。stage004新版本、完整镜像、干净复跑和评审访问待闭合，最后表单用户手动提交。
