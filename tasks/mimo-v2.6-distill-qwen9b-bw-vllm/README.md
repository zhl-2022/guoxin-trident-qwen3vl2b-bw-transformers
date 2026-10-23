# MiMo-V2.6-Distill-Qwen-9B × BW × native vLLM：T01 文本服务

## 1. 当前状态

session004 已读取真实 GPU 原件：BF16、TP1、32K 上下文的 native vLLM 服务加载成功，health/models、单轮、两次多轮调用和 SSE 六项检查通过，共四次生成。成功 JSONL 有32条记录，不是32个请求。正式 T01 精度、性能、完整镜像和干净镜像验收仍未完成，`submission_ready=false`。原件、run ID、来源 SHA 和边界见 [session004 技术报告](../../docs/report-004-mimo-v2.6-distill-qwen9b-bw-vllm.md)。

## 2. 模型与实际范围

模型为 `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`，固定 revision `2367e865d009c13ac81713a2878291d33ab28177`，架构 `Qwen3_5ForConditionalGeneration`，原始 BF16 四片权重；指纹见 [model-lock.json](model-lock.json)。本轮实际加载不替代重新全量校验 SHA 的报告，下载器最终校验原件仍需完整归档。

成功服务显式 `--limit-mm-per-prompt '{"image":0,"video":0}'`。当前标准默认采用同一 **T01 text-only** 配置，限制 image/video 输入并跳过对应 profiling，不修改 checkpoint 权重/结构，不声明视觉/视频适配。保留模型默认采样与思考语义；标准 smoke 不发送 temperature/top_p/top_k/thinking override，不猜测启用 reasoning parser，原始 `<think>` 内容保留。

## 3. 标准入口与默认

[run.sh](run.sh) 使用共享监督器，[env.sh](env.sh) 只读取 MIMO_* 覆盖变量。默认 `/usr/bin/python3`、loopback30109、BF16、TP1、32768上下文、maxseq4、max-num-batched-tokens2048、显存比例0.80、eager、image0/video0。加载预算900秒、总服务预算1800秒，结束后平台关机须另操作。

```bash
bash tasks/mimo-v2.6-distill-qwen9b-bw-vllm/run.sh --dry-run
# 核验固定模型文件、原生环境和预算后执行：
bash tasks/mimo-v2.6-distill-qwen9b-bw-vllm/run.sh
bash tasks/mimo-v2.6-distill-qwen9b-bw-vllm/smoke.sh --log /新的绝对目录/mimo-smoke.jsonl
bash tasks/mimo-v2.6-distill-qwen9b-bw-vllm/stop.sh
```

显式 `MIMO_EXTRA_ARGS_JSON` 替换完整 extra 参数列表；需要其他模态时必须先展开 dry-run、另做有界新实验，不能据文本成功推断其可用。共享控制器拒绝通过 extra 覆盖模型、端口、精度等核心参数。新的默认源码只做 CPU 计划回归，尚未再执行 GPU；实际成功时的 env SHA 和新默认 SHA 在 task.json / 报告分别记载。

## 4. 已保存的失败与成功

首次 run `52e3289d6e9b49778421c75f378b7873` 未限制多模态，600秒 health deadline 后 failed；随后第一次 smoke 连接拒绝。第二次 run `813bbcb199394129949b9afe7214b7cd` 显式 image0/video0，10:46:29 UTC ready，功能验证通过，11:14:10 UTC 到1800秒总预算而 expired；后续 stop 发现监督器已不存在，未发送信号。

首次/第二次权重加载日志308.84秒/4.44秒存在缓存与配置混杂条件，不计算优化加速比。模型服务 expired 与 Notebook 停止计费、独立端口/后代进程清理分别核对。

## 5. 原冻结 pilot 的区别

原 [entry.py](entry.py)、[runtime.json](runtime.json) 和 manifest 的8K单问题 pilot 保留为独立候选，未在本轮重新执行；其 prepared / gpu_verified=false 仅描述旧路径。旧 manifest.json 记录初始候选快照；本轮 README/task metadata 更新后它不再是当前目录的完整性清单，应使用新的 stage004 仓库 manifest。`task.json` 顶层原8K参数有明确 legacy scope，新增 `standard_service_profile` / `session004_observation` 描述本轮32K实际标准服务，不重写旧记录。

```bash
python -B tasks/mimo-v2.6-distill-qwen9b-bw-vllm/entry.py plan
python -B tasks/mimo-v2.6-distill-qwen9b-bw-vllm/entry.py preflight --model-dir /root/private_data/guoxin-candidates/models/mimo9b --hash-weights
```

plan/preflight 不导入 Torch、不下载或启动模型。旧 pilot 的显式 thinking 与8K上下文不代表标准路径的当前默认，也不能容纳模板16K性能档位。

## 6. 功能覆盖

标准 [mimo_text_smoke.py](../../scripts/mimo_text_smoke.py) 保留原始请求/普通响应/完整上下文/SSE事件，验证非空、stop、usage自洽、DONE，无语义评分。三个普通响应 completion tokens 分别10、15、24；SSE12，均未达到2048开发请求上限。不能称已完成正式1024输出负载或答案精度。

## 7. 正式评测缺项

verify/bench 已接真实官方桥，缺材料返回2。`--selfcheck` 仅运行真实外部 selfcheck，不执行完整 eval；T01 bench 要求显式脚本和输出 contract，不借用 T02 图像608请求。仍缺对应本题的 evaluator、固定数据/reference、GPU baseline、思考/解码/评分和性能方法，不能伪造 baseline/PASS。

## 8. 交付边界

正式精度/性能、完整实际镜像 registry/repo:tag、干净镜像复跑、评审访问与最终stage004仓库冻结仍待闭合。保留模型卡MIT和 COPYRIGHT-SCOPE.txt；新代码未选全仓许可证不是模板额外门槛。保留全部失败原件，最后赛事表单由用户手动提交。
