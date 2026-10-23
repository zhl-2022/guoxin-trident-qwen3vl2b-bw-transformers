# Session003：RoboBrain 当前兼容版本实测记录

2026-10-20（UTC+8），当前 `083dc53...` 兼容源码在第二个 BW Notebook 完成 24 项 GPU 数值对照及七个合成图功能请求。默认入口成功启动，七个回答均完整结束，随后同一 run ID 的服务正常停止并确认清理。官方精度、性能、语义验收及 `adaptation_ready` 仍为 false；合成图定位结果不是正式 accuracy。

## 1. 证据来源与身份

本记录只审计已下载原始文件，没有重新运行 GPU。原始证据保留在 `evidence/session003/robo-asr/` 与 `evidence/session003/second-instance/`；关键证据另作字节完全一致的平面副本，便于仓库导出。原始日志、错误和警告均保留，未修改运行时文件。

| 项目 | 实际记录 |
|---|---|
| Notebook | `2112237945233408001`，BW 单卡、原生 vLLM/DTK 环境 |
| 服务 run ID | `c43dadd550fb44dabd84f52df49711d5` |
| 服务地址 / 模型身份 | `127.0.0.1:30001` / `robobrain2-5-8b-nv` |
| 兼容源码 SHA256 | `083dc53f3e4f8171b774469bffa724c2269a9d289769f56a9d7a5624cd81659f` |
| 数值 validator SHA256 | `ba6a94283fa63418948c466eac7fc8949a56cf244eb6f9c3be8dcd78ef673dd1` |
| 新图脚本 SHA256 | `e0dba59f274a1629b53b0cf3c9c4fd3adf72220b41164c34f03e8a72778cbeb7` |
| 编排器 SHA256 | `3fdd8ee4ca4daa47993b5589367ac95a551f0fa7ebc26f8d23d8e516c4004238` |
| 共享监督器 SHA256 | `04b0c91fd0596360f142021800f1487af5a83264cb6de5971b954b45dc80b422` |
| 运行时 task.json SHA256 | `ec35ac53f1b74d75f1fc4524f7076de956d3d3e1b60e9aceecec872624e7e09b` |
| 下载归档 SHA256 | `61dae05b974e3b0151e90885818493bfe27f461c58125f529eae1c79cbcba9fd` |

归档为 1,842,310 字节。根代理已验证 121 个 payload 及 manifest 后解压；本次审计独立复核归档 SHA256。16 个相关源码/任务入口文件的运行前哈希、原始快照字节及本地当前文件在结果登记前全部一致。首次结果登记仅更新 README、结果文档和 task 元数据，当时运行脚本没有变动，因此运行时 task 哈希与更新后的元数据文件不同是预期行为。后续控制逻辑的 CPU 补丁另见第6节，不能将新补丁写成已在003执行。逐文件校验、原始路径和平面副本 SHA256 见[源码及证据审计](../evidence/robobrain-source-audit-003.json)。

历史 `cc6591...` 的 18 项数值和 Math004 图片证据继续保留，只用于那个历史版本。此次结果独立绑定当前 `083dc53...`，没有继承历史 GPU 通过状态。

## 2. GPU 数值与广播预算

[本轮数值原文](../evidence/robobrain-numerical-003.json)的 24 个 case 全部通过；[第二实例先行数值原文](../evidence/robobrain-numerical-initial-003.json)同样为 24 个 case 全部通过。这是同一套检查的两次执行，不能累加为 48 个不同 case。Torch 模块报告 `2.10.0`、HIP `6.3.26093`、实际设备 `BW`；包发行版本与模块版本分别保留在[环境证据](../evidence/robobrain-env-003.json)中。

| 检查 | 本轮结果 |
|---|---|
| 输出有限值 / 与完整 Math 对照 | 24/24 通过，最大绝对误差 `0.0009765625` |
| float32 容差 | `rtol=2e-5`、`atol=2e-5` |
| BF16 容差 | `rtol=0.01`、`atol=0.01` |
| 新增广播检查 | float32/BF16 × 三组 Q/K/V head 几何，共六项 |
| 广播几何 | heads `1/4/4`、`1/1/4`、`4/1/1`；batch `1/2/1`；V 特征维 40 |
| 分块预算 | 每项六块，Q 长度 `7+7+7+7+7+2`；每块 score 元素 ≤ 3752 |
| validator 计数 | seen 28、eligible/completed 26、unchanged 2、chunks 146 |

六个广播 case 直接记录实际广播后的 batch/head、每块 Q/K 长度和 score 元素数，修复的 `Hq=1 / Hkv>1` 预算路径已在 BW 上复验。数值脚本记录的 `peak_allocated_bytes=630342144` 仅属于这些测试张量，不能作为完整模型服务峰值显存或性能成绩。

## 3. 七个图片请求与实际 Math 路径

图片为本项目生成的红方块和蓝圆，尺寸分别为 640×448、512×512。图片哈希及合成坐标见[原始 manifest 副本](../evidence/robobrain-heldout-manifest-003.json)；[红图](../evidence/robobrain-heldout-red-003.png)与[蓝图](../evidence/robobrain-heldout-blue-003.png)保留原始字节。七次请求均非空、`finish_reason=stop`、未达到 token 上限、原文未截断，见[完整请求结果](../evidence/robobrain-heldout-003.jsonl)。

| 请求 | 原始输出 / 合成图诊断 |
|---|---|
| 红图 general | 识别右侧红方块和白色背景 |
| 红图 grounding | `[610, 240, 880, 600]`；合成框 IoU `0.8839659402511187` |
| 红图 pointing | `[(746, 425)]`，在合成框内 |
| 蓝图 general | 识别左下蓝圆、白色背景及图中文字 |
| 蓝图 grounding | `[118, 539, 418, 825]`；合成框 IoU `0.9024416623854203` |
| 蓝图 pointing | `[(275, 675)]`，在合成框内 |
| 双图 general | 识别第一图右上红方块与第二图左下蓝圆 |

本轮未执行轨迹或深度正确性评测。manifest 的 `processor_grid_actually_verified=false` 保持原样；服务日志独立观察到实际 vision 长度 1120、1024，并不等于直接检查了 processor grid。

[完整服务日志](../evidence/robobrain-server-full-003.log)记录实际 `TORCH_SDPA` 路径和 Math 后端，初始化形状为 `[1,4096,16,72]`，真实图片长度为 1120/1024。进程退出计数为 seen/eligible/completed 各 81、unchanged 0、chunks 1323，最大单块 score 为 8,388,608 元素。保留每图完整 K/V 与分图边界；服务日志没有未完成的 eligible Math 调用。GELU 回退和退出时 process-group 警告保留在原始日志，具体 fused kernel 故障根因仍未证明。

## 4. 服务生命周期与平台状态

本轮使用 BF16、TP=1、上下文 32768、显存比例 0.85、eager、最大并行序列 4，每请求最多两张图、零视频。默认兼容入口和 1024×1024 初始化 dummy profiling 已实际执行；这不是更大图像的显存保证，也没有改变真实输入的图像处理规则。实际配置见[运行配置副本](../evidence/robobrain-run-config-003.json)。

| 时间（UTC+8，2026-10-20） | 事件 |
|---|---|
| 01:51:21–01:51:41 | env_check 完成，exit 0 |
| 01:51:41–01:52:13 | 24 项数值检查完成，exit 0 |
| 01:52:23.996595 | 监督器创建当前 run |
| 01:56:24.539129 | 服务 ready |
| 01:56:24–01:56:39 | 七次图片请求，exit 0 |
| 01:56:49.631522 | 服务 stopped，reason 为 `stop requested` |

[ready 状态](../evidence/robobrain-ready-state-003.json)与[stopped 状态](../evidence/robobrain-stopped-state-003.json)的 run ID、监督器/服务 PID 及启动身份一致。[监督器日志](../evidence/robobrain-supervisor-full-003.log)、[停止命令输出](../evidence/robobrain-stop-003.log)及[编排器完整报告](../evidence/robobrain-session-report-003.json)给出 `cleanup_verified=true`。完整报告同时包含后续 ASR 阶段，其约 547.6 秒总时长不能归属于 RoboBrain 吞吐或推理延迟；创建到 ready 约 240.5 秒也只表示本次初始化就绪间隔。

本轮实际加载上限 600 秒、单模型总时限 1800 秒。根代理在平台 UI 实际观察第二个 Notebook 已保存环境并关机；此项为根代理转述，本文没有独立浏览器观察。服务 stopped 与平台停止计费分别记录，不能用前者推断后者。历史预算字段不是本轮后的实时余额。

## 5. 仍未齐备的正式验收条件

- 本模型原版赛事评测脚本、固定数据子集、GPU baseline、任务专用指标/解析器和固定解码参数尚未取得。
- 官方适用图像尺寸/负载、完整镜像 registry 地址与 tag 尚未确认；digest 可增强复现，但不能自加为正式强制条件。
- 没有正式 accuracy、吞吐、TTFT、TPOT、服务峰值显存、提升比例及完整六档官方性能结果。
- 1024×1024 profiling 之外的边界、更大实际图片、干净镜像完整复跑和具体 fused kernel 根因仍需后续验证。

当前产物属于已完成实际开发验证的兼容候选。保持 `official_accuracy=false`、`official_performance=false`、`official_acceptance=false`，最终赛事提交继续等待用户通知。

## 6. 后续控制补丁：仅 CPU 验证

在原始结果审计后，修复了编排器失败路径的两项问题：并发启动时误认其他新 run 的身份，以及 supervisor 消失但 server 仍在加载时错误声明清理完成。编排器现在先生成唯一 run ID，通过可选 `run.sh --run-id` / `stop.sh --run-id` 透传；启动仅认领精确匹配的状态，停止在任务控制锁内核对同一身份，再执行原有 PID/token/config 校验。配置文件独占创建，旧记录不覆盖。清理同时检查 supervisor、server、端口和同 run 的终态。默认未传 run ID 的启动仍自动生成 UUID。

此补丁没有重新执行 GPU。003 原始源码快照、日志、数值和七次图片请求仍绑定第1节旧执行哈希；它们作为该实际执行路径的开发证据继续有效。当前新增控制逻辑只能称 CPU 验证，不能借用003的 GPU结果。

| 当前文件 | CPU 补丁 SHA256 |
|---|---|
| `scripts/session003_robo_asr_checks.py` | `20f358fa199f4e18db08072aa6ccf1a21019fff197233c335ab2555d50491345` |
| `scripts/recon2reason_runtime.py` | `66e9a97c63469698bfc09fbc1158a28fbd7a48f52877abb21156785a33ab1221` |
| `tests/test_session003_robo_asr_checks.py` | `46e34fb5378d88d06c70a1745bfa87e73bb4b2daf968ddd9444f0cc27fd9be2c` |

[CPU 补丁验证记录](../evidence/rules/session003-run-identity-cpu-004.json)保存具体范围：编排器回归 Windows 15项通过/1项 Linux 锁测试跳过，WSL 16项通过；共享 official bridge 回归 Windows 17项通过/1项 Linux 进程组测试跳过，WSL 18项通过；Recon Bash 入口4项通过，RoboBrain/ASR 的 `--run-id --dry-run` 透传2项通过。没有重新开卡、改动003原始证据或作正式提交。
