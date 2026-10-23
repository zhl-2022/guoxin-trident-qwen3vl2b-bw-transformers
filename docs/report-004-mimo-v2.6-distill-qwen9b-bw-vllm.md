# MiMo-V2.6-Distill-Qwen-9B on BW + vLLM（session004 T01 实测报告）

## 1. 简介

T01 `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` 已在 BW 单卡、BF16、native vLLM 的**文本服务配置**实际加载并完成单轮、多轮和输出SSE功能验证。成功运行显式设置 `image=0/video=0`，因此本报告只声明T01文本范围，不声明视觉/视频接口适配。正式精度、性能矩阵、镜像验收和提交仍未完成。

权重：[发布方模型](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)，固定revision `2367e865d009c13ac81713a2878291d33ab28177`，四片权重指纹见 [model-lock](../tasks/mimo-v2.6-distill-qwen9b-bw-vllm/model-lock.json)。本轮server加载成功证明实际读取checkpoint；本归档没有该次下载器的全权重逐文件SHA终态，不把“加载成功”等同于重新全量校验。独立外部仓库URL、最终commit/评审访问和完整镜像待最终填写。

依据已读 [remote-intermediate-001 manifest](../evidence/session004/remote-intermediate-001/export-manifest.json)，SHA `2b1543b7bb12c95f3a0aa041e8e3db605f6189921f2819316c5fcfc7ea86ecb2`。实际脚本与配置逐文件封存，失败/成功原文分别保留；`official_accuracy=false`、`official_performance=false`、`submission_ready=false`。

## 2. 验证环境

<!-- session004-platform-closure -->

本轮已独立核对[SCNet 平台关机与余额记录](../evidence/platform-checkpoint-20261023-004.json)：A/B 两个本轮实例及旧参考实例均为“已关机”；A 手动关机并选择保存环境，B 按预定时限自动关机。可见余额为 **488.0238888888 卡时**，与开机前可见余额差 4.6686111111 卡时；余额差可能包含延迟结算，不当作最终账单。资源为华中一区 A 区 113 组、BW64GB×1、CPU15核、内存59GB。账户镜像 ID `2071293281204580354` / version `2082674417745739777`，Notebook 可选择标签 `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`；完整实际 registry/repo:tag 和评审获取方式尚缺，digest 仅为可复现补充项。平台配置不替代任务原始环境/驱动实测。

本轮代码仓库目标：[zhl-2022/guoxin-mimo-v2.6-distill-qwen9b-bw-vllm 的 codex/session004 分支](https://github.com/zhl-2022/guoxin-mimo-v2.6-distill-qwen9b-bw-vllm/tree/codex/session004)。仓库保持私有；实际推送 commit 和逐文件回读以独立交付回执为准，评审权限仍须落实。光合既有三仓库的 session003 与本轮 GitHub 分支分别记录，不假称光合已同步新版。

| 项目 | 本轮实证与缺口 |
|---|---|
| 平台 | 根代理UI报告实例B `2112237945233408001`、SCNet华中一区A区、BW×1；该平台关联与进程原件来源分开保留 |
| checkpoint | `/root/private_data/guoxin-candidates/models/mimo9b`，模型架构 `Qwen3_5ForConditionalGeneration`；不是旧MiMo-7B |
| 服务 | `/usr/bin/python3 -m vllm.entrypoints.openai.api_server`；server初始化vLLM0.18.1、BF16、TP1、32768上下文 |
| 资源/调度 | 显存比例0.80、eager、maxseq4、max-num-batched-tokens2048；loopback30109 |
| 成功多模态限制 | `--limit-mm-per-prompt '{"image":0,"video":0}'`，限制image/video输入并跳过对应初始化profiling；未修改checkpoint权重/架构 |
| 时限 | 第二次加载预算900秒、服务总预算1800秒；第一次加载预算600秒失败 |
| 镜像/依赖 | 实际完整registry/repo:tag与评审获取待填；本题无独立env.log，完整包路径/版本不得仅从同实例ASR准备记录复制成MiMo独立采集 |

成功 [config](../evidence/session004/remote-intermediate-001/project/tasks/mimo-v2.6-distill-qwen9b-bw-vllm/runtime/config-813bbcb199394129949b9afe7214b7cd.json) 和 [两次合并server.log](../evidence/session004/remote-intermediate-001/project/tasks/mimo-v2.6-distill-qwen9b-bw-vllm/runtime/server.log) 均保留。本轮没有重装原生Torch/vLLM的步骤；这不代替MiMo独立native前后manifest。

## 3. 适配优化方案

保留checkpoint结构、权重、tokenizer、chat template与generation defaults；标准text smoke不发送猜测的temperature/top_p/top_k或thinking override，不添加未经验证的reasoning parser。原始 `<think>...</think>` 内容保留，单次出现看似正确答案不计精度成绩。

成功配置只把T01服务接受的image/video数量设为0，不声称视觉适配成功，也不通过改模型权重规避计算。首次未限制多模态的运行在600秒health deadline失败；第二次既改变了该配置，也处于不同文件缓存/冷热条件。日志分别记录加载权重308.84秒、4.44秒，不能据此把提升完全归因于关闭profiling或宣称加速比。

实际 [mimo_text_smoke.py](../scripts/mimo_text_smoke.py) SHA `4e45292e6b0af32c52026abbe4723b39477d40cd419036ae35a267627396e5a6`；实际GPU运行时env.sh SHA `660c4e5baf4a10428705937beb810aec837ea05a9df25e0b9c58b20055cc3a8d` 见导出manifest。成功时image0/video0来自显式 `MIMO_EXTRA_ARGS_JSON` override；回读后canonical env默认已选同一T01 text-only配置，并把总服务默认上限设为1800秒。**更新后的env源码没有再次GPU执行**，不把新SHA冒充实际运行SHA。回读后执行目标WSL `test_mimo_text_smoke.py` 15项全通过、三个task env的bash语法检查通过，显式dry-run核到image0/video0、32K、900/1800秒和未加reasoning parser。

## 4. 服务启动脚本

canonical [env.sh](../tasks/mimo-v2.6-distill-qwen9b-bw-vllm/env.sh)、[run.sh](../tasks/mimo-v2.6-distill-qwen9b-bw-vllm/run.sh)、[smoke.sh](../tasks/mimo-v2.6-distill-qwen9b-bw-vllm/smoke.sh) 使用共享supervisor和MIMO_*隔离。旧 `entry.py` / `runtime.json` 的8K单请求pilot作为独立历史候选保存，其prepared状态不覆盖本轮32K标准服务实测，也没有被本轮重新执行。旧manifest记录初始候选快照，当前README/task更新后的完整性以后续stage004新manifest为准。

成功运行时的显式 override 保留如下（实际完整 argv 仍以归档 config 为准）：

```bash
MIMO_LOAD_TIMEOUT=900 MIMO_MAX_RUNTIME_SECONDS=1800 \
MIMO_EXTRA_ARGS_JSON='["--enforce-eager","--max-num-seqs","4","--max-num-batched-tokens","2048","--limit-mm-per-prompt","{\"image\":0,\"video\":0}"]' \
  bash tasks/mimo-v2.6-distill-qwen9b-bw-vllm/run.sh
```

当前默认已包含同一组 serving 参数，常规入口为：

```bash
bash tasks/mimo-v2.6-distill-qwen9b-bw-vllm/run.sh --dry-run
# 当前默认复现成功的T01文本配置，含image0/video0；首次仍需先核checkpoint与预算。
bash tasks/mimo-v2.6-distill-qwen9b-bw-vllm/run.sh
bash tasks/mimo-v2.6-distill-qwen9b-bw-vllm/smoke.sh --log /新的绝对路径/mimo-smoke.jsonl
bash tasks/mimo-v2.6-distill-qwen9b-bw-vllm/verify.sh --selfcheck
bash tasks/mimo-v2.6-distill-qwen9b-bw-vllm/stop.sh
```

| 尝试 | 实际生命周期与终态 |
|---|---|
| 首次 `52e3289d6e9b49778421c75f378b7873` | 10:28:42.853353 UTC创建；`health-check timeout`，10:39:13.245270 UTC failed，elapsed630.392秒；随后旧smoke连接拒绝，原件保留 |
| 第二次 `813bbcb199394129949b9afe7214b7cd` | 10:44:01.162930 UTC创建，10:46:29.361855 UTC ready；总1800秒到期，11:14:10.112657 UTC **expired**，elapsed1808.949秒 |

见 [首次终态](../evidence/session004/remote-intermediate-001/project/evidence/session004/mimo-first-state.json)、[第二次终态](../evidence/session004/remote-intermediate-001/project/evidence/session004/mimo-second-stopped.json)、[后续stop输出](../evidence/session004/remote-intermediate-001/project/evidence/session004/mimo-second-stop.log)。stop调用只报告监督器已不存在、未发送信号，不改写为它主动停止了模型；独立进程/端口清理和平台关闭计费仍需各自证据。

## 5. Smoke 测试

[成功原件](../evidence/session004/remote-intermediate-001/project/evidence/session004/mimo-standard-smoke-002.jsonl) 有32条JSONL，是原始请求/响应、SSE事件和检查记录的条数，**不是32个请求**。客户端run ID `cf2599b885024bb0bbb9409e99959c8b` 与服务run ID不同；11:07:36.176848至11:10:50.059970 UTC，summary `functional_checks_passed=true`。

实际六项为health、models、single、multiturn_first、multiturn_followup、stream，其中四次生成请求。三个普通响应均HTTP200、非空、finish_reason stop、usage自洽；多轮保留完整前文，未通过猜测拆分thinking给出答案评分。SSE有原始 `[DONE]`、stop、usage14 prompt + 12 completion = 26，原文为 `<think></think>Hello! How can I help you today?`，读取wall0.733929秒。max_tokens2048为开发请求上限，不代表实际生成2048或符合正式1K输出负载。

[首次连接失败](../evidence/session004/remote-intermediate-001/project/evidence/session004/mimo-standard-smoke.jsonl) 的两条记录保留 `Connection refused` 与summary false，不合并进成功002。四次生成都是短功能请求，未计算题目答案正确率；检查通过不表示该checkpoint正式T01精度通过。

## 6. 精度测评

正式T01精度仍未执行。需要与本模型绑定的官方verify/selfcheck、GPU baseline、固定GSM8K/MMLU或实际分题数据、thinking/max_tokens/解码/解析及判分契约。不能从模板示例题数或通用2pt样例构造本模型基线。

当前桥 `--selfcheck` 仅执行真实外部selfcheck，缺材料返回2；完整eval另行调用并保存原始预测和 `verify_result.json`。未知schema不生成PASS。正式分数、baseline、评分公式与材料SHA仍见 [待填字段](stage004-report-fields.json)。

## 7. 性能参考

本轮未执行4K/16K输入、1K输出、并发1/8/64、各两轮的性能矩阵。32K实际engine初始化只证明该配置启动成功，四次短问答/SSE与日志瞬时吞吐不构成正式throughput/TTFT/ITL。

两次加载时间存在文件缓存和profiling配置混杂因素；没有受控A/B或加速比结论。全部正式性能档位暂列未执行，正式取数须另存命令、输入长度政策、采样/思考设置、warmup、两轮raw与失败数，不挪用Recon的图像矩阵。

## 8. 注意事项

- 已验证T01文本服务的BF16/32K/native vLLM实际加载与功能；image/video输入为0，视觉适配未验证。
- 第一次failed、第一次smoke连接拒绝、第二次成功与expired终态分别保留。第二次比第一次快不证明具体优化的独立收益。
- 新默认image0/video0和1800秒服务时限是成功配置的工程默认化，只做目标CPU计划回归，尚未再跑GPU；执行时原env SHA与当前源码SHA分别登记。
- 旧8K entry pilot的冻结源码/manifest不改成本轮已运行；standard32K的实测另挂session004记录。
- 全权重重新SHA校验原件、正式精度/性能、完整镜像和干净复跑、stage004访问仍有缺项。版权范围和上游MIT声明保留，许可证选择不是模板额外门槛。
- `submission_ready=false`，最后赛事表单由用户手动提交；Notebook最终计费状态不能从模型expired推断。
