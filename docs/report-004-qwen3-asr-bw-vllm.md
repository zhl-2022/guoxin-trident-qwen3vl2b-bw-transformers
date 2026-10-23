# Qwen3-ASR-1.7B on BW + vLLM（session004 实测报告）

## 1. 简介

原始 `Qwen/Qwen3-ASR-1.7B` checkpoint 已在本轮 BW + native vLLM 完成新的隔离音频环境准备、中文/英文短 WAV 转写与整文件上传后 SSE 输出验证。本报告已读取 session004 原件；功能验证成立，正式 WER/CER、音频性能方法和最终镜像验收仍未完成。`formal_ready=false`、`official_accuracy=false`、`submitted=false`。

权重：[发布方模型](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)。历史私有仓库：[光合](https://developer.sourcefind.cn/codes/ghu_30ebea88e/guoxin-qwen3-asr-bw-vllm)、[GitHub](https://github.com/zhl-2022/guoxin-qwen3-asr-bw-vllm)；session004 的新 commit、同步与评审权限尚需最终核对。

本轮依据 [remote-intermediate-001 导出 manifest](../evidence/session004/remote-intermediate-001/export-manifest.json)，其 SHA256 为 `2b1543b7bb12c95f3a0aa041e8e3db605f6189921f2819316c5fcfc7ea86ecb2`。manifest 封存 payload 大小/SHA/CRC，以及实际执行脚本前后 SHA；原件只读，未把旧 session003 的长音频结果算成本轮新增实验。

## 2. 验证环境

<!-- session004-platform-closure -->

本轮已独立核对[SCNet 平台关机与余额记录](../evidence/platform-checkpoint-20261023-004.json)：A/B 两个本轮实例及旧参考实例均为“已关机”；A 手动关机并选择保存环境，B 按预定时限自动关机。可见余额为 **488.0238888888 卡时**，与开机前可见余额差 4.6686111111 卡时；余额差可能包含延迟结算，不当作最终账单。资源为华中一区 A 区 113 组、BW64GB×1、CPU15核、内存59GB。账户镜像 ID `2071293281204580354` / version `2082674417745739777`，Notebook 可选择标签 `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`；完整实际 registry/repo:tag 和评审获取方式尚缺，digest 仅为可复现补充项。平台配置不替代任务原始环境/驱动实测。

本轮代码仓库目标：[zhl-2022/guoxin-qwen3-asr-bw-vllm 的 codex/session004 分支](https://github.com/zhl-2022/guoxin-qwen3-asr-bw-vllm/tree/codex/session004)。仓库保持私有；实际推送 commit 和逐文件回读以独立交付回执为准，评审权限仍须落实。光合既有三仓库的 session003 与本轮 GitHub 分支分别记录，不假称光合已同步新版。

| 项目 | 本轮实证或明确缺口 |
|---|---|
| 平台 | SCNet BW 单卡；根代理 UI 报告实例 B 为 `2112237945233408001`、华中一区 A 区；该平台映射与进程原件的证据来源分开保留 |
| 原生依赖 | prepare 原件列 Torch `2.10.0+das.opt1.dtk2604.20260325.g6b060a`、vLLM `0.18.1+das.fa71803.dtk2604`、HCU `0.18.1+das.dtk2604`、Transformers `5.5.0`；受保护包均来自 `/usr/local/lib/python3.10/dist-packages` |
| 服务解释器 | project 下 `tasks/qwen3-asr-bw-vllm/runtime/audio-venv/bin/python`，新建 system-site-packages venv |
| 模型与服务 | `/root/private_data/SothisAI/model/Aihub/Qwen3-ASR-1.7B/main/Qwen3-ASR-1.7B`；`qwen3-asr-1.7b`；`127.0.0.1:30002` |
| 实际配置 | BF16、TP1、32768 上下文、显存比例 0.85、eager、max-num-seqs4；加载预算600秒、总服务预算900秒 |
| 镜像 | 展示名 `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`；实际完整 registry/repo:tag、获取方式及最终镜像干净复跑仍缺，不用公开同名 base 镜像冒充 |

实际配置见 [config](../evidence/session004/remote-intermediate-001/project/tasks/qwen3-asr-bw-vllm/runtime/config-1a02b2e4e0c944cabdb195dd54dc8348.json)。本轮未生成单独 env.log，不把历史 Python/设备详细 probe 改标为本次采集；包身份来自本轮 prepare 原件，GPU engine 状态来自 server.log。

## 3. 适配优化方案

局部 `vllm.general_plugins` 插件在 Transformers5.5 父类读取 thinker 前构造同值 child，再运行原始初始化和 token-ID 校验。未修改权重/tokenizer 或关闭验证。实际插件 SHA 为 `84d0d5402c73d21562fb12bb649ace56d6f29b7d1fdf1d3e402b8d4a6e28820a`；[server.log](../evidence/session004/remote-intermediate-001/project/tasks/qwen3-asr-bw-vllm/runtime/server.log) 保存 API 父进程与 EngineCore 两次安装事件，目标配置源码 SHA 为 `56fd1f1ea939dff47533b052962d294375e85482ed310e48ccad402cf5699818`。保留公开同类工作的引用，不把初始化顺序修复宣称为从零独创。

本轮 [prepare 启动输出](../evidence/session004/remote-intermediate-001/project/evidence/session004/asr-start-002.log) 与归档 `asr-prepare-evidence` 确认七个锁定 CPU wheel 安装成功，12 个 protected native 包的版本/位置在 native-before 与 ready 逐项相等；WAV 解码及重采样检查通过。prepare 自身 `model_loaded=false/gpu_checked=false/service_started=false` 仅界定准备阶段；随后真实 serving 已另有 ready 和请求证据。最初错误选择 `/usr/bin/python` 为任务 venv 的失败见 [asr-start.log](../evidence/session004/remote-intermediate-001/project/evidence/session004/asr-start.log)，保留原文，不重写成成功。

## 4. 服务启动脚本

canonical [env.sh](../tasks/qwen3-asr-bw-vllm/env.sh)、[run.sh](../tasks/qwen3-asr-bw-vllm/run.sh) 为默认路径。新 venv 缺失时才准备，已有目录只校验，失败不自动删除或覆盖。`QWEN_ASR_*` 隔离任务身份和预算。原生包/配置源码漂移时拒绝运行。

```bash
bash tasks/qwen3-asr-bw-vllm/run.sh --dry-run
SCNET_BASE_PYTHON=/usr/bin/python bash tasks/qwen3-asr-bw-vllm/run.sh
bash tasks/qwen3-asr-bw-vllm/smoke.sh
bash tasks/qwen3-asr-bw-vllm/verify.sh --selfcheck
bash tasks/qwen3-asr-bw-vllm/bench.sh
bash tasks/qwen3-asr-bw-vllm/stop.sh
```

服务 run ID 为 `1a02b2e4e0c944cabdb195dd54dc8348`；创建 `09:57:35.327930 UTC`，ready `09:59:20.009955 UTC`，终态 **expired**，`10:12:39.030575 UTC` 因900秒预算到期结束，监督器 elapsed903.702秒。见 [终态](../evidence/session004/remote-intermediate-001/project/evidence/session004/asr-state-stopped.json)。后续 [stop 输出](../evidence/session004/remote-intermediate-001/project/evidence/session004/asr-stop.log) 为“recorded supervisor is no longer running; no process was signaled”，不能改写为该 stop 调用主动终止服务，也不单独证明所有后代/端口已独立清理核验。Notebook 停止计费另查平台终态。

## 5. Smoke 测试

默认固定 WAV 的来源与转换见 [fixture manifest](../evidence/session004/remote-intermediate-001/project/fixtures/smoke-session004/manifest.json)。中文 SHA `46dbc998c9d1d48111267c40741dd3200f2e5bcf4075f8c4c97f4451160dce50`，英文 SHA `64e219123047e533c1c1dbb446cb04873ff8c72a2ff20e24ae9d84664d9b455f`。均单声道 PCM16/16kHz；音频按完整文件上传，不发送参考文本、语言或 prompt。

| 本轮输入 | 时长 | 实际结果 | HTTP 往返 |
|---|---:|---|---:|
| [默认中文](../evidence/session004/remote-intermediate-001/project/evidence/session004/asr-default-smoke.jsonl) | 4.2039375秒 | 非空14字符；模型身份匹配 | 7.145111秒 |
| [英文](../evidence/session004/remote-intermediate-001/project/evidence/session004/asr-en-smoke.jsonl) | 15.05125秒 | 非空188字符；模型身份匹配 | 1.786333秒 |

普通转写路由未返回 finish reason 或 token usage；`raw_text.truncated=false` 只说明日志未裁剪，不证明模型输出无截断。两条 summary 的 `semantic_accuracy_verified=false`，WER/CER 未计算。客户端 SHA 为 `da40fff4081c7873a4d276225d4ef990df9b2d666daf3c1d73ee0796def655f1`；客户端自己的 run ID 与监督器服务 run ID 分开记录。

[中文 SSE 原件](../evidence/session004/remote-intermediate-001/project/evidence/session004/asr-sse-zh.jsonl) 有12个 JSON事件、`finish_reason=stop`、`[DONE]`、65 prompt + 11 completion = 76 tokens；原始 `language Chinese<asr_text>` 前缀保留，客户端 SHA 为 `3ae83568801ff05787cef6b959d3cfc28b63d3014e04049d8803cb3cd35a7f05`。第一非空 delta0.077767秒、流读取0.421373秒，仅单次开发观测。整文件输入后的输出流式已经验证，实时分块音频输入没有验证。

## 6. 精度测评

正式 WER/CER 仍未执行。需要官方固定音频/reference、语言/规范化/分词、解码、独立 GPU baseline、2% 劣化公式及原版 evaluator/selfcheck。不能把模型返回反作参考答案。历史开发集、session003 的人工拼接39秒结果和纯离线 identity-normalized WER/CER 工具不代替正式材料。

当前 `verify.sh` 已启用通用真实官方桥；`--selfcheck` 只执行真实外部 selfcheck，不运行 full eval。缺材料返回2；未知 schema 为 `executed_uninterpreted`，不生成 accuracy PASS。T04 bench 必须显式外部脚本和输出 contract，详见 [桥 CPU 验证](../evidence/session004/official-bridge-validation-20261023/validation.json) 与 [字段表](stage004-report-fields.json)。

## 7. 性能参考

上述 HTTP/SSE 用时不构成正式 RTF、吞吐或流式延迟成绩；900秒监督器生命周期也不是音频推理耗时。本轮只完成两条短 WAV 和一次中文整文件 SSE，无音频长度×batch/并发矩阵、稳态预热或正式性能判定。

T04 音频计量方法尚缺，不能套 T02 图像608请求。未来须同时填写总音频秒数、请求数/完整失败数、预热/轮次、实际输入模式、推理与 HTTP 计时边界及 RTF/吞吐公式。

## 8. 注意事项

- 本轮功能通过；正式 WER/CER、性能、自然连续长音频、噪声、多语种与实时输入均不能据短 WAV 推断。
- `expired` 终态、后续 stop 未发送信号、独立进程/端口清理与平台停机分开记载；当前归档不足以把最后两项都标为已核通过。
- 七个 CPU 包在隔离环境安装不等于替换原生 Torch/vLLM。原始失败和历史003均保留，当前代码更新后的 task SHA 不冒充执行时 SHA。
- 最终镜像、干净镜像复跑、stage004 commit/访问与报告装配仍需闭合。保留版权范围和上游许可，模板未强制全仓开源。用户手动提交最后表单。
