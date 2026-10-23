# Trident-Qwen3VL-2B：vLLM离线GPU通过，HTTP接口本地准备

本轮真实native vLLM已输出1 query + 4 documents的2048维向量，5条token IDs全部与官方processor一致。**新增HTTP服务只完成CPU契约，GPU服务复跑与正式检索精度/性能未完成；submission_ready=false。** 历史目录名称含transformers，实际成功路线是vLLM。见 [八节技术报告](../../docs/report-004-trident-qwen3vl2b-bw-transformers.md)。

## 固定模型与真实结果

- `openbmb/Trident-Qwen3VL-2B`，revision `a5aedf8733f2107ab75b42214c78496d85759077`；4.255GB合并BF16权重完整核验，无额外基座/adapter。
- native Torch2.10/vLLM0.18.1+das、Transformers5.5.0；隔离qwen-vl-utils0.0.14。BF16/TP1/8K、LAST池化、FP32后L2、memory0.45/eager。
- 输入覆盖文本、图像、图文，向量维度2048、范数约1，点积排序 `[0,3,2,1]`；合成例没有qrels，不是检索精度成绩。
- [完整向量/输入/分数](../../evidence/session004/remote-intermediate-001/candidates/results/trident-vllm-pooling-001/worker-result.json)、[prompt审计](../../evidence/session004/remote-intermediate-001/candidates/results/trident-vllm-pooling-001/prompt-audit.json)、[引擎配置](../../evidence/session004/remote-intermediate-001/candidates/results/trident-vllm-pooling-001/engine-options.json)、[终态](../../evidence/session004/remote-intermediate-001/candidates/results/trident-vllm-pooling-001/run-final.json)。结果SHA为 `7de46591cf782cbb50ac6d670f822868f766097b44854d568a2b13f83eec3044`。
- 加载至调用65.617416秒、run总116.630754秒，只是5条开发请求。父进程peak allocator=0不代表EngineCore实际零显存；有效峰值未知。

## 标准HTTP入口

本地新增env/run/smoke/verify/bench/stop/env_check文件，使用native vLLM `/v1/embeddings` messages扩展，默认127.0.0.1:30119，参见 [service-profile.json](service-profile.json)。run沿用有时限、PID/start-token保护的现有supervisor；TRIDENT_LOAD_TIMEOUT/TRIDENT_MAX_RUNTIME_SECONDS从进入supervisor后约束服务，此前同步执行完整权重SHA与模型语义预检没有独立wall deadline。输入为embedding而非chat/completions；客户端保留query/doc指令与末尾assistant位置，图像先经同一官方规则处理再传data URI，server do_resize=false。

```bash
bash tasks/trident-qwen3vl2b-bw-transformers/run.sh --dry-run
bash tasks/trident-qwen3vl2b-bw-transformers/smoke.sh --dry-run
# 下列是未来GPU HTTP复跑命令，本轮没有执行：
bash tasks/trident-qwen3vl2b-bw-transformers/run.sh
bash tasks/trident-qwen3vl2b-bw-transformers/smoke.sh
bash tasks/trident-qwen3vl2b-bw-transformers/stop.sh
```

默认候选venv为 `/root/private_data/guoxin-candidates/envs/trident-001/bin/python`，可用TRIDENT_PYTHON_BIN显式指定；load600秒/max-runtime900秒。smoke独占新日志，发送真实embedding HTTP请求，检查模型、usage、维度/范数并保留完整响应。它没有GPU HTTP成功原件；本地假服务契约结果不能继承离线GPU的向量验证。

TRIDENT_MODEL_PATH相对路径按调用者cwd一次解析为绝对路径，预检和server的--model/--tokenizer使用同一值。stop.sh --dry-run只输出计划，不调用停止流程或发送信号。

HTTP错误响应也按同一socket body deadline/2MB cap读取，日志保留endpoint/status、正文及字节SHA，最终返回失败；错误正文超限或超时则记录捕获失败原因，不算接口成功。

verify/bench实际执行时缺正式embedding schema则返回2；--dry-run仅输出plan_only并返回0，不算正式验证通过，也不生成verify_result或1024-token性能矩阵。official_embedding_evaluator_bridge_implemented=false，当前是显式阻塞入口，真实官方脚本注入/调用桥接尚未实现。原发布方TF entry.py仍保留作未来参考，但没有TF向量实测结果，不把它当作已完成的数值基线。

## 仍缺材料与边界

尚缺HTTP GPU复跑、同输入TF向量数值对照、官方subset/qrels/预处理/指标、GPU baseline和原版evaluator/selfcheck、embedding吞吐/延迟负载、视频/多图边界以及完整镜像引用/干净复跑。日志中的ROCm GELU tanh到none回退也须在精度对照中核查。

公开DocVQA Origin仅有 [benchmark计划](benchmark-plan.json)，没有执行，不能写成赛事或论文Mix结果。原GPU probe已由 [最终导出manifest](../../evidence/session004/remote-final-001/export-manifest.json) 的before/after hash-only seal核对SHA256=2b4030517bd55056595b4caa34506c5f8573d6c3149b81aa81c507a10061ede7；归档未重构源码payload。新HTTP源码仍由总装配另封SHA，旧manifest尚未覆盖本次修改。上游MIT与新代码版权范围保留，最终表单由用户手动提交。
