# Recon session004 下一次最小补跑计划

状态：**仅文件建议，未执行，不启动 GPU**。本计划以 [最终审计](../evidence/session004/remote-final-001-audit.json) 为依据：八格进程全部退出0且原件齐全，三格合计十二条空正文，严格整格合格为五格；服务已清理。后续先诊断最低成本的空输出格，再决定 c1 补齐和其他选择性复查，不为了“进程已完成”重跑已有完整原件。

## 1. 已知缺口与最小范围

| 缺口 | 已核事实 | 下一次最小范围 | 能证明什么 |
|---|---|---|---|
| c1 两档未运行 | 本轮实际 controller 只选 4096/16384 × c8/c64 × 两轮；c1 共四格缺失 | 4096/c1 与 16384/c1，各两轮；共 32 个测量请求，按原设置另有四次 warmup | 新 run 的四份原版客户端结果；不回填为旧 run 原件 |
| 三个 16K 格存在空输出 | R2/c8 空1条；R1/c64 空7条；R2/c64 空4条。全部空样本 TTFT=0、ITL=[]、errors=""、output_lens=1024 | 先保存可归因的请求/事件诊断，优先复查 R2/c8 的16请求；c64 两轮各128请求只在诊断确有必要时另行安排 | 判断是否再次出现空文本及对应 usage/finish/DONE；不能删除旧坏样本 |
| 16384/c64 最后两格 | 最终原件已验证退出0、wall1622.878336/1600.620530秒；空输出使两格仍不合格 | 不为补“结果文件缺失”重复运行，它们的文件已齐全 | 最终进程完成、坏样本和服务清理已有证据 |
| 正式验收契约 | evaluator、固定数据、GPU baseline、总输入长度和冷缓存/解码要求未闭合 | 继续查找和固定官方材料；不靠额外随机吞吐补成正式精度 | 正式验收仍独立待完成 |

建议首批最多是一次 c8 空输出诊断复查加四个 c1 格，合计 **48 个测量请求**；若诊断尚未闭合，则先停在诊断阶段。之后若确需复查两格 c64，再增加256请求，全部选择性复查与缺口补齐合计304请求，仍不是重跑整套12格。新的服务/缓存状态必须单独记录，不能把不同启动条件直接拼为“同一轮完整基准”。

## 2. 固定复现条件

- 继续使用同一个已核模型路径、BF16、TP=1、上下文 32768、显存比例 0.85、eager、max-num-seqs=16、image=2/video=0。重新核对实际启动 argv 和 ready 状态；保存完整镜像引用、原生包版本、设备清单和模型身份。
- 原版客户端固定 SGLang 0.5.12，sender alias 为 `sglang-oai-chat`，服务框架为 vLLM。004g driver SHA 为 `ff8eef852e19f336e99884cc031353627ca304c6a7534c22375116758a850000`；不要在旧运行目录覆盖 driver 或结果。
- 保持一张 1024×1024 随机 JPEG、seed=1、文本目标 4096/16384、输出目标 1024、warmup=1、temperature=0、ignore_eos=true、stream usage=true。保存真实请求总数及 warmup 数，不把图像/chat token 当作已包含在文本目标内。
- 记录 prefix-cache 实际设置和是否新启服务。官方若要求独立重启、关闭缓存或其他输入定义，先按照正式契约重新形成可比较的运行方案；不能把当前未明确的条件写成已满足。
- 每个新输出目录必须不存在。使用明确的单格、客户端总时限、服务监督时限和平台自动关机上限；实际完成即停止，不假定上一轮实例仍在计费或可复用。

## 3. c1 四格命令草案

以下命令仅供后续已有选定服务和运行限额时使用。`RECON_MODEL`、`RECON_RUNTIME`、`RECON_RUN_ID`、`SGLANG_PYTHON`、`NEW_C1_OUTPUT` 都必须取自那次实际环境和全新目录；本文件没有填造 run ID，也没有执行这些命令。

```bash
python -B scripts/t02_sglang_bench.py \
  --run --allow-client-device-discovery \
  --client-python "$SGLANG_PYTHON" \
  --model-path "$RECON_MODEL" --served-model-name "$RECON_MODEL" \
  --runtime-dir "$RECON_RUNTIME" --expected-run-id "$RECON_RUN_ID" \
  --output-dir "$NEW_C1_OUTPUT" \
  --load 4096:1 --load 16384:1 --round 1 --round 2 \
  --warmup-requests 1 --timeout-seconds 900 --total-timeout-seconds 3000
```

900 秒单格和 3000 秒客户端总时限是建议的硬上限，不是完成时间估计或成功保证。服务监督时限还须覆盖模型加载与清理，平台自动关机须单独设置。若某格到上限，保留 timeout 与部分产物，不自动扩展预算、降低输入/输出或改采样参数来获得“通过”。

## 4. 空输出复查

先使用现有 native 数组、对应客户端/server 日志定位空输出发生的那次请求。当前 JSONL 没有保存每次原始 SSE usage 和完整 finish/DONE 证据，不能从 output_lens=1024 判断真实生成了 1024 个可见 token。后续诊断需要保存请求标识、输入身份、原始事件或事件摘要、正文/思考片段计数、usage、finish reason、DONE 和错误，明确哪些字节被保留；**该诊断采集尚未实现，不能把本文件当作已有工具**。

诊断可能引入额外 I/O，因此其吞吐不替代未修改原版客户端的性能结果。待诊断入口实际实现并做 CPU 协议验证后，单独以新目录复查整格：

```bash
python -B scripts/t02_sglang_bench.py \
  --run --allow-client-device-discovery \
  --client-python "$SGLANG_PYTHON" \
  --model-path "$RECON_MODEL" --served-model-name "$RECON_MODEL" \
  --runtime-dir "$RECON_RUNTIME" --expected-run-id "$RECON_RUN_ID" \
  --output-dir "$NEW_EMPTY_OUTPUT_RECHECK" \
  --load 16384:8 --round 2 \
  --warmup-requests 1 --timeout-seconds 600 --total-timeout-seconds 600
```

相同 seed 不等于已保存了原样本输入；若要逐请求对比，必须实际固定并保存请求/媒体身份。该次复查成功也只证明新 run；旧 `round2_16384_1024_c8` 和两格 c64 的坏样本、不合格结论继续保留。c64 复查命令需要另外按其128请求规模和新的硬预算形成，不能直接照搬本节600秒的 c8诊断上限。

## 5. 回读与结束条件

使用 [独立 analyzer](../scripts/analyze_sglang_results_004.py) 在新的分析目录读取新导出。只将顶层 `request_rate: Infinity` 规范化为字符串 `inf`，保留原始 SHA 和 producer 状态；其他非有限数值、超时、mock、空输出或证据不闭合继续失败。

若 exporter 按设计只导出脚本 SHA，先独立核对 ZIP CRC、manifest 全部大小/SHA 和导出前后脚本身份，再给 analyzer 显式传入 manifest 与其已核 SHA。不得从本地源码补造一份“远端原件”。保存最终 stopped/expired、owned cleanup 与平台终态证据，然后形成新报告版本。正式精度、官方验收和用户手动提交仍分别记录。
