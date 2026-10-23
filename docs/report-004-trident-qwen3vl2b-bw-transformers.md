# Trident-Qwen3VL-2B on BW + vLLM（session004 离线GPU实测与HTTP准备）

## 1. 简介

本轮真实native vLLM已完成Trident pooling conversion：1个文本query、4个文本/图像/图文document，均输出2048维有限L2向量；5条展开token IDs与同一官方processor全部一致，开发契约通过。**这是离线LLM.embed GPU验证；新增/v1/embeddings服务入口只完成本地CPU契约，没有GPU启动HTTP服务，也没有正式检索评测。**

模型 `openbmb/Trident-Qwen3VL-2B`，revision `a5aedf8733f2107ab75b42214c78496d85759077`。原发布方Trident `7fbf4b2033e1b3a43a99185fc125d765a549c52d`的指令、池化和图像语义保留；已合并权重，不另换基座。见 [model-lock](../tasks/trident-qwen3vl2b-bw-transformers/model-lock.json)、[upstream-lock](../tasks/trident-qwen3vl2b-bw-transformers/upstream-lock.json)。历史目录名含transformers，实际本轮成功路径为vLLM。

## 2. 验证环境

<!-- session004-platform-closure -->

本轮已独立核对[SCNet 平台关机与余额记录](../evidence/platform-checkpoint-20261023-004.json)：A/B 两个本轮实例及旧参考实例均为“已关机”；A 手动关机并选择保存环境，B 按预定时限自动关机。可见余额为 **488.0238888888 卡时**，与开机前可见余额差 4.6686111111 卡时；余额差可能包含延迟结算，不当作最终账单。资源为华中一区 A 区 113 组、BW64GB×1、CPU15核、内存59GB。账户镜像 ID `2071293281204580354` / version `2082674417745739777`，Notebook 可选择标签 `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`；完整实际 registry/repo:tag 和评审获取方式尚缺，digest 仅为可复现补充项。平台配置不替代任务原始环境/驱动实测。

本轮代码仓库目标：[zhl-2022/guoxin-trident-qwen3vl2b-bw-transformers 的 codex/session004 分支](https://github.com/zhl-2022/guoxin-trident-qwen3vl2b-bw-transformers/tree/codex/session004)。仓库保持私有；实际推送 commit 和逐文件回读以独立交付回执为准，评审权限仍须落实。光合既有三仓库的 session003 与本轮 GitHub 分支分别记录，不假称光合已同步新版。

| 项目 | 实测原件 |
|---|---|
| 平台/精度 | SCNet BW64GB×1为平台配置；BF16、TP1、8K；EngineCore记录cuda/DAS执行 |
| Python | `/root/private_data/guoxin-candidates/envs/trident-001/bin/python` |
| native | Torch2.10.0+das、torchvision0.25.0+das、vLLM0.18.1+das.fa71803.dtk2604、vllm-hcu0.18.1+das.dtk2604 |
| CPU依赖 | Transformers5.5.0/Pillow12.1.1复用native；qwen-vl-utils0.0.14位于候选venv |
| checkpoint | `/root/private_data/guoxin-candidates/models/trident`，全部文件/完整权重hash通过 |
| 镜像 | 实际私有Jupyter完整registry/repo:tag及digest待闭合；公开base不能替代 |

见 [environment](../evidence/session004/remote-intermediate-001/candidates/results/trident-vllm-pooling-001/environment.json)、[checkpoint核验](../evidence/session004/remote-intermediate-001/candidates/results/trident-vllm-pooling-001/checkpoint-verification.json)。初次“CPU-only”断言被native设备发现触发；后续修复复用39项、安装0项，真实记录GPU context discovery，native metadata不变。并发repair lock导致的一次失败原件也保留。

## 3. 适配优化方案

625个BF16 tensor使用标准Qwen3-VL前缀，无额外projection/lm_head；仅在运行配置指定 `Qwen3VLForConditionalGeneration`，`runner=pooling`、`convert=embed`、LAST pooling、FP32 head后L2归一化。权重、tokenizer和RoPE尺寸不改，query/doc分别使用发布方system指令及assistant generation prompt。

图像采用固定qwen-vl-utils、patch16、min_pixels262144/max_pixels1310720预处理，再以do_resize=false送入vLLM。5条token IDs实际核对通过。参数为max_num_seqs4、max_num_batched_tokens8192、memory0.45、eager、关闭prefix cache，见 [engine-options](../evidence/session004/remote-intermediate-001/candidates/results/trident-vllm-pooling-001/engine-options.json)、[路线依据](trident-vllm-pooling-20261023.md)。

EngineCore日志提示ROCm的GELU tanh approximation回退为GELU none；这项差异仍需TF数值对照/检索精度验证，token一致和向量范数正确不能替代精度验证。

## 4. 服务启动脚本

成功GPU原件来自 [独立pooling probe](../scripts/trident_vllm_pooling_probe_20261023.py)，实际上限480秒。[run-start](../evidence/session004/remote-intermediate-001/candidates/results/trident-vllm-pooling-001/run-start.json)保存准确argv；[run-final](../evidence/session004/remote-intermediate-001/candidates/results/trident-vllm-pooling-001/run-final.json)为completed/exit0，并记录自有进程组清理。

本地已新增env/run/smoke/verify/bench/stop/env_check标准文件，见 [service-profile](../tasks/trident-qwen3vl2b-bw-transformers/service-profile.json)。run向已有有时限、PID/start-token保护的supervisor提供8K embedding配置；HTTP采用vLLM官方/v1/embeddings messages扩展。客户端保留system指令、图像先于文本、末尾空assistant与continue_final_message，图像先按官方规则处理。[官方HTTP示例](https://github.com/vllm-project/vllm/blob/v0.18.1/examples/pooling/embed/vision_embedding_online.py)。

```bash
bash tasks/trident-qwen3vl2b-bw-transformers/run.sh --dry-run
bash tasks/trident-qwen3vl2b-bw-transformers/smoke.sh --dry-run
# 以下仅供未来GPU HTTP复跑，本轮未执行：
bash tasks/trident-qwen3vl2b-bw-transformers/run.sh
bash tasks/trident-qwen3vl2b-bw-transformers/smoke.sh
bash tasks/trident-qwen3vl2b-bw-transformers/stop.sh
```

默认127.0.0.1:30119、候选venv。进入supervisor后，服务启动上限600秒、服务总运行上限900秒；此前同步执行的完整权重SHA和模型语义预检没有独立wall deadline，不能把900秒写成从hash开始的整条run上限。smoke发送真实embedding请求，保存原始响应、usage、维度和范数；不发送chat/completions。verify/bench在实际执行且缺embedding正式contract时返回2；加--dry-run仅输出plan_only并返回0，不能算正式验证通过。HTTP接口CPU/mock通过不继承离线GPU通过；最终新增源码SHA/stage封存待总装配。

TRIDENT_MODEL_PATH在调用者当前目录解析一次为绝对路径，完整预检与server的--model/--tokenizer共用同一身份，避免server进入任务目录后重解释相对路径。stop.sh --dry-run只输出计划，不能调用停止流程或给进程发信号。HTTP400/500使用与成功响应相同的socket body deadline和2MB cap读取错误正文，保存endpoint、status、原文、字节base64及SHA后保持失败；读取错误也保留status及失败原因。这些边界由新增CPU回归覆盖，不构成GPU验证。

## 5. Smoke 测试

[完整向量原件](../evidence/session004/remote-intermediate-001/candidates/results/trident-vllm-pooling-001/worker-result.json) SHA256 `7de46591cf782cbb50ac6d670f822868f766097b44854d568a2b13f83eec3044`；[prompt审计](../evidence/session004/remote-intermediate-001/candidates/results/trident-vllm-pooling-001/prompt-audit.json)保留输入token与图像SHA。图片为64×64合成红矩形，再经官方规则预处理。

| 文档索引 | 输入 | 与query余弦点积 |
|---|---|---:|
| 0 | 文本a red rectangle | 0.8738650785 |
| 1 | 文本a blue circle | 0.7526463320 |
| 2 | 红矩形图片 | 0.7866230580 |
| 3 | 红矩形图片+文本a red rectangle | 0.7910106371 |

query文本为a red rectangle，实际排序 `[0,3,2,1]`。1+4向量均2048维，范数范围1.0000000189至1.0000000516；5条prompt token identity均true。这些是结构/接口检查，合成例没有独立qrels，排序不是正式Recall/nDCG成绩。

## 6. 精度测评

`numerical_reference_available=false`，尚无同输入发布方TF向量对照。还缺赛事固定检索subset/qrels、预处理/指令约定、指标实现、GPU baseline及原版evaluator/selfcheck。[DocVQA Origin计划](../tasks/trident-qwen3vl2b-bw-transformers/benchmark-plan.json)未执行，不是赛事固定集或论文完整Mix表。

当前official_accuracy_verified=false，没有官方verify结果。official_embedding_evaluator_bridge_implemented=false：verify/bench目前是显式阻塞入口，尚未实现真实官方embedding脚本注入/调用桥接；取得材料后仍需实现并验证。已实证使用vLLM开发路线，但完整服务契约与正式精度验收仍需闭合，不能据一次向量输出计完成正式提交。

## 7. 性能参考

probe记录引擎加载至调用结束65.617416秒；整个run为116.630754秒，包括完整checkpoint hash等。仅5条顺序请求，无预热/重复、并发矩阵或独立latency分解，不能报告正式embedding吞吐或加速比。

原始gpu_peak_allocated_bytes=0是调用进程allocator读数，模型在EngineCore子进程；它不代表实际零显存，有效峰值显存未知。日志KV cache可用21.81GiB/204224 tokens也不是峰值测量。正式性能须采用embedding文本/图像负载和向量吞吐/延迟，不能套生成1024 token或TTFT/TPOT表。

## 8. 注意事项

- 已完成：真实vLLM离线pooling GPU加载、5条向量和token一致性。未完成：HTTP GPU复跑、视频/多图边界、TF数值对照、官方检索精度/性能。
- 新HTTP客户端处理图像并注入发布方query/doc指令；任意原始图像、消息模板不能自动继承本验证。
- GPU probe执行身份已由 [最终导出manifest](../evidence/session004/remote-final-001/export-manifest.json) 的external-script-hashes/004-trident_vllm_pooling_probe_20261023.py前后hash-only seal闭合，SHA256为2b4030517bd55056595b4caa34506c5f8573d6c3149b81aa81c507a10061ede7，与本地已审核probe一致；归档没有重构源码payload。新增HTTP代码另封SHA，不倒填旧GPU身份。
- 11:30 UTC中间导出不证明最终关机/费用终态；仓库访问、完整镜像引用和干净复跑由最终交付闭合。
- MIT上游许可与新代码版权范围说明保留；目录历史名称保留，实际框架明确写vLLM。
- official_accuracy_verified、official_performance_verified、submission_ready均false；最后赛事表单由用户手动提交。
