# Session004 验收要求与材料缺口对照

本文件以 2026-10-23 本轮取得的赛事说明、技术报告模板及本地证据为依据。它用于确定接下来做什么、报告怎样填写；不把正在进行的 session004 算力工作提前写为成功，不改写 session003 的原始失败或验收状态。结构化清单见 [requirements-and-gaps-20261023.json](../evidence/session004/requirements-and-gaps-20261023.json)。

## 1. 直接结论与依据

1. **Recon 的 11/12 格开发覆盖，不自动等于整题不可提交。** 模板第 8 节允许主动说明未支持负载、排队超时和没有数据的档位。当前证据只能说两次执行都受总时间预算截断，不能把“显存不足、天然不支持 16K/c64”写成已证原因；该格也不能冒称完成。
2. **尚未选择全仓开源许可证，不是模板规定的强制障碍。** 用户已经选择保留版权范围说明与上游原许可。内部 `license_ready=false` 不应被提升为赛事模板硬门槛；仍须遵守各上游代码、权重、数据的实际许可。
3. **真实验收缺口是明确材料和可复现产物。** 当前已核材料中，没有与三题准确绑定的完整 evaluator/selfcheck、固定评测数据及参考、GPU baseline 和解码/解析契约；实际 Jupyter 镜像的完整可获取引用也未闭合。不能用自建 smoke、开发 WER/CER 或发布方分数替代这些材料。
4. **Transformers-only 候选尚未证明完成赛事计件要求。** 最新说明将计件写为模型 × 卡型 × 框架（SGLang/vLLM），参赛流程还要求框架版本。直接 Transformers 推理可作为候选开发证据；要计为新增完成件，仍需验证对应框架路径，或取得该分题明确允许其他执行方式的说明。
5. **最后验收表单由用户手动提交。** 本轮任务是准备代码、完整报告和可核原始产物，不代替用户执行最后提交。

直接来源与读取边界：

| 来源 | 本轮核对内容 | 本地依据与限制 |
|---|---|---|
| [赛事官网](https://www.scnet.cn/home/subject/modular/index890.html) | 主代理已浏览参赛说明原图：四赛道、任务清单与 Feishu 一致；详情/模板入口指向下列 Feishu，公开 landing 没有 evaluator 下载入口 | 此项来自主代理传回的 UI 核对上下文，本文件作者没有重新亲读官网图片；不外推为所有渠道都没有发布材料 |
| [最新 Feishu 赛事详情](https://bvjoh3z2qoz.feishu.cn/wiki/OkLBws6uWiG1Q0kpHLKcxXLmnif) | T02/T04 范围、指标、计件粒度、SCNet 算力要求与提交流程 | [tracks-current.ax.txt](../evidence/session004/rules-current-20261023/tracks-current.ax.txt)、[participation-current.ax.txt](../evidence/session004/rules-current-20261023/participation-current.ax.txt)；仅证明快照包含的可见文本 |
| [用户给出的技术报告模板](../国芯国模黑客松-技术报告模板.md) | 八章节、标准脚本、精度原件、SGLang 压测客户端、第 8 节披露规则 | 与[已归档模板](../evidence/official-materials/20261019/report-template.md)逐字节一致，SHA256 `b151143f6fa38c87e301cfc9335a94d17fcd49c7113a53fe05f89c04b45f349f` |

## 2. 任务、框架与验收口径

| 项目 | 最新说明中的要求 | 三题当前应采用的处理 |
|---|---|---|
| 算力与卡型 | 正式参赛使用 SCNet；硬件为 BW 或 AI-64GB，自带算力不计入官方验收 | 本项目已有 BW 单卡的实际平台证据。Windows/WSL CPU 测试证明工程行为，不能代替 SCNet 实测 |
| 计件粒度 | 模型 × 卡型 × SGLang/vLLM；记录框架版本；量化单独计件，同规格部分变体不重复计 | 三题当前实际服务框架是 native DAS vLLM/BF16；SGLang **客户端**测 vLLM **服务**，仍是 vLLM 这件，不自动再计 SGLang 服务一件 |
| T02 通用 VL/VQA | 完整图像预处理、vision encoder、LLM 链路，支持多图；OCRBench/MMMU 或对应 VQA 集与 GPU baseline 比较 | OCRBench 千分制劣化上限 20 分，MMMU 上限 2 pt；实际分题是否采用这些集须由分题材料绑定，不能替 Recon/Robo 猜判分集 |
| T02 具身/VLA | 本赛季只做 serving；公示图像含深度图，解析后的动作/坐标与 GPU 参考输出一致率至少 98%，单图/多图正常 | Robo 与涉及空间/动作输出的 Recon 路径应取得准确输入集、参考、解析器及容差；合成方块坐标、IoU 和数值 kernel 对照不等于该 98% 指标；机器人环境任务不属于本季必做项 |
| T02 embedding | 图文输入到向量输出，使用公示检索子集和检索指标 | Trident 等候选不能套生成式 chat 压测，需真正 embedding/检索路径 |
| T04 ASR | 音频到文本，含流式；公示子集的 WER/CER，较官方 GPU baseline 劣化不超过 2% | 需固定 reference、规范化、分词、语言和解码设置；“2%”的具体差值算法不能从文本模型 2 pt 样例自行推导；整文件上传后的 SSE 输出与实时音频输入分开说明 |
| T04 强制对齐 | 音频加文本到词/字时间戳，AAS（ms）与 GPU baseline 比较 | Transformers-only 时间戳输出可先做功能预检，但不先宣称已满足框架粒度或 AAS 验收 |
| T03 生成类 | 公示 seed/prompt/条件、全 pipeline 产物和对应 GPU 参考指标 | 不把单子模块成功或缺少参考的生成样例当完整新题验收 |

提交优先与可领取状态不能从下拉题目列表直接推导。用户说明按提交先后竞速；可见赛事文本同时写了首次验收通过计件。需要保留提交与验收的实际回执，不能把“还可选择”写成“已确认没有其他人提交”。

## 3. 模板八节与本次需要准备的材料

| 模板章节 | 可以填写的已知事实 | 仍需补齐或明确写未完成 |
|---|---|---|
| 1 简介 | 精确模型、BW、BF16、实际 vLLM 版本、三份代码仓库；上游来源 | 评审可获取的最终镜像完整地址与本次仓库版本 |
| 2 验证环境 | SCNet 区域、实例配置、镜像 ID/version/name/tag、原始 env.log | `registry/repo:tag`；不能将公开 base 地址直接当作真实 Jupyter 镜像引用，也不能以 UI 标签代替实测版本 |
| 3 适配方案 | Robo guarded vision math、ASR TF5.5 兼容插件、隔离 CPU 依赖及相应执行源码 | 新版本是否实际 GPU 运行，由新的原始 snapshot 单独证明；不能沿用旧源码的通过结论 |
| 4 启动 | env/run/stop 的实际 argv 与版本，服务健康和所有权控制 | 最终镜像从干净状态一键复跑；不要把 `--dry-run` 写成服务已启动 |
| 5 Smoke | 已保存的单/多图、多轮、流式、ASR 音频返回 | 对应新版本的完整原始 smoke.log；固定功能 fixture 不是正式精度数据 |
| 6 精度 | 如实写“分题材料未齐，正式评测未执行”；可另列开发检查 | 原版 evaluator/selfcheck、数据/reference、baseline、解析/规范化/解码与正式 verify_result.json；不能把模拟 selfcheck 的测试通过当官方 selfcheck |
| 7 性能 | 独立 SGLang client、实际 vLLM 服务；T02 六档两轮、第二轮取数；原始 native 文件另存 | 本次真实运行结果及输入/图像政策；T04 另按音频方法，不强套图文 608 请求；新 client 不能继承旧 native vLLM random-mm 成绩 |
| 8 注意事项 | 实际超时、未完成档、加载时间、支持范围、已知警告 | 对没有原件的格明确写“无数据/未完成”，不能删去超时日志，也不自动宣判整题失败 |

标准入口 `verify.py --selfcheck`应执行真实官方脚本的 selfcheck。当前执行桥支持仅 selfcheck、归档原文，缺脚本时返回 2；成功只表示 selfcheck 执行完成。未知 schema 的完整外部评测仍为 `executed_uninterpreted`，不得自动升为精度 PASS。

私有代码仓库已经保存不等于评审已可访问。最终应明确仓库链接、准确 commit、评审获取方式，并独立核验归档；不擅自更改仓库公开性。许可证范围说明按用户现有决定保留。

## 4. 新旧结果与源码的对应关系

| 证据范围 | 可证明什么 | 不继承到哪里 |
|---|---|---|
| session003 Recon 原矩阵与独立补跑 | 11/12 格开发覆盖、480 个完整请求、491520 个输出 token；末格超时；[最终审计](../evidence/recon-matrix-final-audit-003.json) | 不是完整官方矩阵；vLLM `random-mm` 的命令和结果不能改名为 SGLang client 结果 |
| session003 Robo/ASR | 对应当时源码的 BW 数值/图片功能、发布方派生 WAV 转写和整文件 SSE；[总报告](session003-delivery.md) | 后续控制器、SSE、入口修复只有各自新测试/实跑才能取得新结论 |
| session003 最终交付 closure | [三仓库与 ZIP 文件一致性](../evidence/final-delivery-closure-003.json)，正式标志仍 false | 不证明 session004 新代码已上传、已 GPU 实跑、已获评审访问或已提交 |
| session004 通用官方执行桥 | [CPU/临时脚本与 Bash 入口验证](../evidence/session004/official-bridge-validation-20261023/validation.json)；runtime SHA `0edc33da22c07a804c93047f918a4a39a58af90997390c49695d8b39a891d85b` | 模拟脚本不是官方 evaluator，本地测试不产生官方准确率 |
| session004 SGLang 004b | driver SHA `7318269cd4ae006c564c18b433cf3b6de852787acad9bde31947f9dfe67abead`，bootstrap SHA `3a18d7f08a90affcbd57474b72aba8288df6143cae3c8100f8a63eb2b01c431b`；[CPU 与真实 venv 路径回归](../evidence/session004/sglang-robo-validation-004b-20261023/validation.json) | 官方 client 的 import、SCNet 实测与结果仍要对应远端原件；旧 004 helper 有解释器 symlink 问题，不部署 |
| 本轮 A/B 算力执行 | 主代理单独记录正在执行的服务、代码 SHA、run ID、原始输出、清理与平台终态 | 本对照文档不提前评估尚未读到的计算结果，也不以旧“所有实例已关机”快照描述本轮实时状态 |

SGLang 004b 要求服务名与本地模型路径一致，避免上游 image processor 把短别名当下载标识；默认计划 608 个测量请求加 12 个额外 warmup。即使 detailed `output_lens`均为 1024，也需说明该字段可能来自 SSE usage 或请求长度 fallback，不能冒称已独立认证精确输出 token 数。详见[客户端准备说明](../evidence/session004/sglang-client-preparation-20261023.md)。

## 5. 具体缺项与关闭条件

| 缺项 | 当前状态 | 关闭条件 |
|---|---|---|
| 官方 evaluator 与 selfcheck | 已核来源尚未取得与三题绑定的完整包 | 保存原版文件与 SHA、入口参数，真实执行 selfcheck 并归档原文 |
| 固定测试数据与参考 | 功能 fixture/发布方音频已有，不能替代 | 模型分题固定 manifest、数据版本/ID、独立答案或 GPU 参考，以及数据许可 |
| GPU baseline | 尚未取得准确对应三题配置的官方文件 | 模型/精度、任务集、框架或适用比较口径、分数及判定算法可核 |
| 指标与调用契约 | 通用指标已写明，分题解析/预处理仍未闭合 | 例如动作/坐标解析与容差、OCR/VQA 数据映射、ASR 规范化/语言/解码/2%算法明确 |
| T02 本次性能原件 | 004b 工程就绪，实际结果由本轮运行产生 | 保留每档两轮命令和 native 原件；未完成档允许在第 8 节如实披露，无需为了凑齐无限重复开卡 |
| T04 性能方法 | 已有显式外部脚本/输出 contract 的音频桥，尚缺方法 | 获得音频时长、batch/并发、预热、吞吐/RTF/延迟/流式定义后实测；不伪用 T02 |
| 完整实际镜像引用 | 现有镜像 ID/tag 可定位，完整评审引用未闭合 | 保存真实可获取的 `registry/repo:tag` 与获取方式，并在干净环境复跑；digest 是可复现补充，不虚增为模板强制项 |
| 新增题框架符合性 | TF-only 或独立 pipeline 候选仍是开发阶段 | 证明对应模型×卡型×框架路径，或取得该分题明确允许的例外；不得先累计完成件数 |
| 最终提交材料 | 报告、代码、原始文件及访问方式需要同一版本对应 | 生成完整八节报告并列明未完成项，由用户审阅并手动提交[验收表](https://bvjoh3z2qoz.feishu.cn/share/base/form/shrcnLIas6FqdBnaJ9sDLUUNXnc) |

当前没有新的官方验收原件支持改变 `formal_ready=false`、`official_accuracy=false`、`official_performance=false`、`submitted=false`。这些标志与“已经有实际适配结果、仓库已保存、正在继续开发”并不冲突。正式状态只能在对应材料、实跑和判定原件到位后更新；最终提交仍由用户完成。
