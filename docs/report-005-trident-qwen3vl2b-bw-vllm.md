# Trident-Qwen3VL-2B on BW + vLLM（session005 HTTP GPU 实测补充报告）

## 1. 简介

本轮真实 GPU 已完成标准 `/v1/embeddings` HTTP 链路，复用 session004 已加载成功的模型、vLLM pooling 配置及固定五条开发输入。最终导出包通过全部路径、大小、SHA、CRC 与安全解包回读；独立读取五份原始 HTTP 响应重新计算后，确认五个向量均为 2048 维有限 L2 向量，排序为 `[0,3,2,1]`，与 session004 离线开发参考逐元素最大绝对差为 `0.0`。服务日志恰有五条 embedding POST 200。

首次 HTTP 成功后，旧停止入口因运行目录软链接造成配置路径比较失败，原 controller 的 `cleanup_verified=false`、stop exit2 和 controller exit2 原件保留。独立守护清理随后只向身份匹配的原 supervisor 发 TERM，观察到其与 API server 原 PID/start-token 均已消失、30119 端口关闭，最终 state 于 **13:46:57.300601 UTC** 记录 `stopped / stop requested`；export 前后快照与该 state 完全一致。补救关停与原 controller 的失败分别报告。

正式检索精度、正式性能、官方分数解释及提交就绪状态均为 `false`。HTTP 实跑后另加透明外部 evaluator 执行桥，目前仅通过 CPU 合同测试，详见第 6、7 节；不能把这项后加工程实现写入旧 GPU producer 的执行 seal。赛事最后提交由用户手动完成。

| 证据层 | 当前状态 | 判据 |
|---|---|---|
| 001 失败原件 | 本地独立核验完成 | 33 位 run ID 被真实 supervisor 在启动模型前拒绝 |
| retry002 HTTP | 本地独立核验通过 | 五条真实响应、usage、2048 维有限 L2 向量及开发参考比较 |
| retry002 原停止入口 | 本地核验失败原件，保留失败 | 逻辑与解析后配置路径不同，安全身份检查拒绝发信号 |
| 独立守护清理 | 本地独立核验通过 | 原 PID 启动时间、配置身份、源码 SHA、端口关闭 |
| 官方精度与性能 | 尚未完成 | 固定检索材料、GPU baseline、正式 evaluator 与负载契约 |

## 2. 验证环境

模型为 `openbmb/Trident-Qwen3VL-2B`，固定 revision `a5aedf8733f2107ab75b42214c78496d85759077`。模型路径为 `/root/private_data/guoxin-candidates/models/trident`，候选 Python 为 `/root/private_data/guoxin-candidates/envs/trident-001/bin/python`，服务端口为 `127.0.0.1:30119`。模型锁与发布方源码锁沿用 [model-lock](../tasks/trident-qwen3vl2b-bw-transformers/model-lock.json) 和 [upstream-lock](../tasks/trident-qwen3vl2b-bw-transformers/upstream-lock.json)。

session005 的 [native-before](../evidence/session005/remote-trident-http-final-001/project/evidence/session005/trident-http-retry002/native-before.json) 与 [native-after](../evidence/session005/remote-trident-http-final-001/project/evidence/session005/trident-http-retry002/native-after.json) 全对象相等，七项包的版本、安装根路径和 METADATA SHA 均保持一致，本轮没有安装 native 包。该检查证明这些 metadata 身份稳定，不是完整二进制目录审计。

| 组件 | session005 原件记录 |
|---|---|
| Torch | `2.10.0+das.opt1.dtk2604.20260325.g6b060a` |
| torchvision | `0.25.0+das.opt1.dtk2604.torch2100.20260325.g860505` |
| vLLM | `0.18.1+das.fa71803.dtk2604` |
| vllm-hcu | `0.18.1+das.dtk2604` |
| Transformers / Pillow | `5.5.0` / `12.1.1`，复用 native |
| qwen-vl-utils | `0.0.14`，位于既有候选 venv |
| 配置 | BF16、TP1、8K、eager、4 seq、memory0.45 |

固定源码在实际 launch 前强制执行完整 checkpoint hash gate，成功 launch 已越过该 gate；本轮没有另导出逐文件 checkpoint hash 结果 JSON，不能伪造该原件。[session004 报告](report-004-trident-qwen3vl2b-bw-transformers.md)保留此前完整 checkpoint 核验背景。实际 Jupyter 镜像完整 registry/repo:tag、当前实例开关机与卡时平台记录由本轮平台交付另行闭合；公开 base 镜像名称不能充当当前镜像完整引用。同实例期间另有 Kimi checkpoint 下载，本轮 smoke 不报告隔离正式性能。

| 源码包 | 项目与运行 ID | 冻结身份 |
|---|---|---|
| 001 | `guoxin-hackathon-20261023-session005`；33 位 ID `40f92c092d064ce8839ea9987bc4c704b` | ZIP SHA `9f7de950a133ebede7455f301300485c2594daa90c42be126aef3b2eeb984ad6` |
| retry002 | `guoxin-hackathon-20261023-session005-trident-retry002`；32 位 ID `1b62c9a782bf40ec923a16edc132e8ad` | ZIP SHA `f1104acb8dd5b5c4d35bc111c20144def9567f9ad7765b15f514d6e893689536` |
| 独立清理工具 001 | 固定绑定 retry002，不改旧 state/config/源码 | SHA `8c2dffd12dc089a3b418e6aaef2d67abf1f2df9a50506901ea2b70e9132b58a1` |

retry002 的 source manifest SHA 为 `6c888030447e5f1f81c13308628ac45d52fdc59a0b5db00e837641855aeb69a7`。新 UUID 使用 `uuid4().hex` 固定一次，并通过真实 canonical supervisor 的 `checked_run_id` 及 launch dry-run；001 原件保持不变。

## 3. 适配优化方案

采用 native vLLM 的 `pooling` runner、`convert=embed`、LAST pooling、FP32 head 后 L2 归一化。配置为 BF16、TP1、8K、max_num_seqs4、memory0.45、eager。HTTP 入口使用 vLLM 官方 `/v1/embeddings` messages 扩展，query/document 指令、图像先于文本、末尾空 assistant 与 `continue_final_message` 保持固定。[官方接口示例](https://github.com/vllm-project/vllm/blob/v0.18.1/examples/pooling/embed/vision_embedding_online.py)。

图像使用固定 `qwen-vl-utils==0.0.14` 按发布方规则预处理，再以 `do_resize=false` 送入 vLLM。代码不修改模型权重、tokenizer、RoPE、指令或池化位置。兼容性只覆盖本轮实际输入，任意消息格式、多图和视频边界需另外验证。

## 4. 服务启动脚本

```bash
/usr/bin/python3.10 /root/session005-trident-http-002.install.py \
  /root/session005-trident-http-002.zip

/root/private_data/guoxin-candidates/envs/trident-001/bin/python \
  /root/private_data/guoxin-hackathon-20261023-session005-trident-retry002/scripts/session005_trident_http_retry002.py preflight

timeout -s TERM -k 65s 2300s \
  /root/private_data/guoxin-candidates/envs/trident-001/bin/python \
  /root/private_data/guoxin-hackathon-20261023-session005-trident-retry002/scripts/session005_trident_http_retry002.py run
```

controller 给包含权重 hash 的启动步骤 750 秒、smoke 600 秒、主阶段预算 2100 秒，另留清理预算；外层 OS `timeout` 提供 2300 秒及最终 kill 宽限。canonical run 自身同步 hash 不在其服务计时内，不能把 canonical service timeout 当作从 hash 起整条执行的总时限。

[实际 run-final](../evidence/session005/remote-trident-http-final-001/project/evidence/session005/trident-http-retry002/run-final.json) 为 13:28:25.228793 至 13:31:06.236911 UTC，controller 耗时 161.281498 秒，其中 launch 126.894445 秒、smoke 子进程 32.612301 秒；两步均 exit0、未超时、子进程已回收。随后原 stop exit2 使 controller 保持 exit2。服务在补救清理前仍存活，完整生命周期以最终 state 的 13:46:57 UTC 结束为准。

原停止失败来自路径身份比较：`launch` 的 `/proc` argv 使用 `task/runtime/config-UUID.json`，`supervise(args.config.resolve())` 的 state 保存真实 evidence 路径。原 stop 逐字比较路径而拒绝发信号。独立工具仅在固定项目、运行 ID、supervisor PID/start token、固定 runtime SHA、完整 argv、同一解析后 config、server PPID/token/command 均相符后向原 supervisor 发送 TERM，由原 finally 清理服务。没有重写历史 state/config 来绕过校验。

## 5. Smoke 测试

固定输入为一个文本 query，以及文本正例、文本负例、图像、图文四个 document；图片为 session004 同一合成红矩形。客户端保存五份原始 HTTP embedding 响应，逐份核对模型 ID、usage、向量维度、有限数值和 L2 范数。

五条响应的完整 SHA、usage 和重新计算的范数见 [独立 audit JSON](../evidence/session005/remote-trident-http-final-001-audit.json)；原始数值保存在 [smoke.jsonl](../evidence/session005/remote-trident-http-final-001/project/evidence/session005/trident-http-retry002/smoke.jsonl)。

| case | 实际 L2 范数 | prompt/total tokens | completion tokens | 与 query 余弦点积 |
|---|---:|---:|---:|---:|
| query_text | 1.000000018904 | 24 / 24 | 0 | — |
| doc_text | 1.000000021152 | 23 / 23 | 0 | 0.8738650785 |
| doc_negative | 1.000000051552 | 23 / 23 | 0 | 0.7526463320 |
| doc_image | 1.000000028419 | 278 / 278 | 0 | 0.7866230580 |
| doc_fused | 1.000000020030 | 281 / 281 | 0 | 0.7910106371 |

独立重新计算排序为 `[0,3,2,1]`，对五个离线开发向量逐元素比较的最大绝对差为 **0.0**。本地不同 Python 版本的累加算法使点积与远端记录最多相差 `9.9921e-16`，小于显式 `1e-12` 重算容差；排序和向量元素相等分别验证，不靠四舍五入推断。

请求原件确认发布方 query/doc 指令、图像先于文本、末尾空 assistant、`continue_final_message=true` 与零生成 usage；图像和图文请求中的 512×512 PNG 字节 SHA 完全相同，原始图像 SHA 与固定 fixture 相符。HTTP 服务没有导出每条实际 prompt token IDs，不能把 usage token 数相同表述为另一次逐 token 审计；该项 session004 离线 probe 曾独立检查。

参考是 session004 真实 native vLLM 离线 probe 的五条开发向量，原 worker-result SHA 为 `7de46591cf782cbb50ac6d670f822868f766097b44854d568a2b13f83eec3044`。它不是发布方 Transformers 数值参考，也不是官方 baseline；合成输入没有官方 qrels，排序一致不能换算为 Recall/nDCG 或精度达标。

## 6. 精度测评

还缺官方 embedding evaluator/selfcheck、固定检索 subset 与 qrels、预处理和指令约定、指标实现、GPU baseline。现在已实现独立 [透明外部执行桥](../scripts/trident_official_bridge_005.py) 与后加 [verify005.sh](../tasks/trident-qwen3vl2b-bw-transformers/verify005.sh)：显式指定 selfcheck/evaluator、原样 argv、数据文件清单、qrels、baseline、输出文件契约与超时后，可运行外部脚本，保留 stdout/stderr、退出状态及输入前后 SHA；缺材料返回 2，`--dry-run` 只输出计划。完整环境变量合同见 [OFFICIAL-BRIDGE-005](../tasks/trident-qwen3vl2b-bw-transformers/OFFICIAL-BRIDGE-005.md)。

新增桥成功状态只叫 `executed_uninterpreted`，不猜测官方 JSON schema、不读取分数作判断、不产生验收 PASS；`official_schema_interpretation_implemented=false`。已执行 CPU mock 合同回归：Windows 12 通过、1 项 POSIX 软链接测试跳过；WSL 13/13 通过，覆盖真实直属子进程自检、bench 参数原样、缺材料、非零退出、超时回收、路径、输出缺失、输入变动和旧证据拒绝覆盖。另验证材料环境路径在进入新输出 cwd 前冻结为绝对路径，额外 argv 保持逐字不改。这些临时材料不是官方样本或模型精度结果。

实际 HTTP GPU producer 的 runtime、smoke、model lock 与原 `verify.sh`/`bench.sh` 均未改。005 根标准入口可通过显式组装 override 指向 `verify005.sh`/`bench005.sh`，保留旧任务入口并记录其为后加 CPU overlay；它与本轮已核 GPU 源码 seal 分开。官方材料取得后仍须审核脚本与来源、实现正式 schema/阈值解释，再真实完成精度验证。

## 7. 性能参考

还缺正式 embedding 性能负载与阈值。后加 [bench005.sh](../tasks/trident-qwen3vl2b-bw-transformers/bench005.sh) 只执行显式 embedding 负载脚本及其输出文件合同，不继承生成式 T02 的 4K/16K→1024-token 假设，也不注入 `BENCH_MODE=image`。外部脚本 exit0 与输出文件存在只证明执行完成，性能含义、阈值和与 GPU baseline 的比较仍待官方合同解释。

五条顺序 smoke 只能证明接口与开发输入兼容，不构成预热后重复、并发、吞吐或尾延迟测试。embedding 任务不填生成 1024 token、TTFT/TPOT 的生成式负载成绩。未观测的峰值显存、正式精度分数及加速比保持未知。

## 8. 注意事项与归档

已独立核验的 intermediate001 ZIP SHA 为 `7d071b6bda8863d60461d3f4de5cf23bdea3df041bfa41cae471e4bba85c7bfb`。27 个 entry、26 个 payload、31 个源码前后 hash seal 均通过路径、大小、SHA、CRC 和安全解包回读；[独立审计记录](../evidence/session005/remote-intermediate-001-audit.json)确认 001 在模型启动前因 33 位 ID 退出 2，不能称 GPU 内核失败。

最终 [本地完整性记录](../evidence/session005/remote-trident-http-final-001.local-verification.json) 与 [独立结果审计](../evidence/session005/remote-trident-http-final-001-audit.json) 已完成。ZIP 为 121608 bytes，SHA256 `1865af152496f46dc7ff9d26004b6cde16fdd53a070d3d2dcb355e5923fea912`；实际 manifest SHA256 为 `d1d63542eb0bc5489763bb0c7a34b57532a927bd630bbb8afba42fd43bbf1179`。41 个 entry、40 个 payload、33 个前后 source seals 全部通过，13 份本项目实际脚本/入口 seal 与冻结 retry002 源码包逐一吻合。外部 cleanup helper SHA 与本地审阅版本及 helper 自报 SHA 三方一致。

独立清理的 [执行记录](../evidence/session005/remote-trident-http-final-001/project/evidence/session005/trident-guarded-cleanup-executed-001.json) 绑定 supervisor PID281/start-token278950610 和 server PID282/start-token278950641，观察两个原进程均已消失、30119 已关闭；最终 state 与 export 前后快照完全相等。[原 stop 失败](../evidence/session005/remote-trident-http-final-001/project/evidence/session005/trident-http-retry002/cleanup.json)继续保留。

exporter 保存源码前后 hash 身份，未包含的源码 payload 不重构；实际 argv、日志、原始响应及 run/state 共同支持执行结论，hash-only 记录单独不构成执行证明。最终平台关机、余额与后续代码仓库交付由本轮平台终态另行记录。所有 session005 新报告与证据独立保存，不覆盖已冻结 session004 报告、仓库或 stage。

| 状态字段 | 当前值 |
|---|---|
| `HTTP_GPU_raw_evidence_locally_audited` | `true`，五条原始响应与 server POST200 对应 |
| `original_retry002_controller_cleanup_verified` | `false`，保留原始失败 |
| `independent_guarded_cleanup_locally_audited` | `true`，两个原进程与端口均已清理 |
| `official_accuracy_verified` | `false` |
| `official_performance_verified` | `false` |
| `official_execution_bridge_implemented` | `true`，后加 005 透明执行桥，仅 CPU 合同测试 |
| `official_schema_interpretation_implemented` | `false` |
| `submission_ready` | `false` |
| `contest_submission_performed` | `false` |
