# 三题正式评测材料第二轮定向核查

核查日期：2026-10-20（Asia/Shanghai）。本轮读取公开网页、匿名 GitLab API、已有 `zhl-2022` 身份的 GitHub 只读 API，并下载小型文本、示例 manifest 和源码。没有操作 SCNet、运行模型或下载权重，没有新建远端项目、发布成果、提交比赛或联系主办方。

## 1. 本轮结论

**补完上一轮的深层目录盲区后，仍未从本轮检查的公开入口取得 Recon2Reason、RoboBrain2.5-8B-NV、Qwen3-ASR-1.7B 对应的赛事固定数据、GPU baseline 和正式判分器完整组合。**这不是“主办方没有材料”的证明，也不能推导材料必须私发。当前可继续做工程封装、开发评测、兼容复验、性能工具与提交材料整理，但需要把开发结果和正式验收结果分别标注。

本轮发现可复用的通用材料：OpenDAS EvalScope 中英文 ASR 规范化与评分器、FLEURS/LibriSpeech 适配器、OCRBench/MMMU 适配器；它们没有声明本次赛事固定子集或 GPU 成绩。HYGON 的 `quality-gate` 是 PR 工程检查，SGLang 的 nightly `baseline` 是隐藏状态回归参考，均不能充当本次赛事 baseline。

完整查询、每页目录、固定提交、URL、HTTP 状态、响应及下载哈希保存在 [research-round2-20261020.json](../evidence/research-round2-20261020.json)。下载的小文件位于 `evidence/official-materials/20261020-round2/`，均未执行。

## 2. 深层目录与公开入口覆盖

| 本轮入口 | 实际覆盖 | 结果与边界 |
|---|---|---|
| [OpenDAS/evalscope_dataset](https://developer.sourcefind.cn/codes/OpenDAS/evalscope_dataset) | 固定提交 `b36f21f3c5cc7c13461c6047377d4e6c58780db1`；递归树每页 100 项，共 **21 页、2026 项**，最后一页 26 项且 `X-Next-Page` 为空 | 补齐上一轮仅第一页的盲区；路径中没有精确三模型的赛事 `verify.py`、baseline 或专属目录 |
| 同仓库 `custom_eval/` | 阅读全部 18 个小型 JSONL/TSV/CSV 文本示例，没有下载五张图片 | 示例是 dog、museum、Tokyo 等通用 VQA，以及通用文本/检索样例；不是三模型的赛事固定集 |
| 同仓库评测代码 | 下载 FLEURS、LibriSpeech、MMMU、OCRBench 适配器及 WER 规范化源码和对应文档 | 可作开发评测参考；缺比赛映射、baseline 与固定 subset |
| OpenDAS 公开项目目录 | 每页 100，上游返回 **98 个项目**，下一页为空 | 没有新发现独立赛事项目名称；没有逐个下载 98 个仓库内容 |
| 光合公开项目定向检索 | `国芯国模`、`黑客松`、`guoxin`、`hackathon`、`Recon2Reason`、`RoboBrain2.5`、`baseline`、`evalscope` | 前七项项目元数据检索无命中；`evalscope` 4 项。没有重做 626 项 ModelZoo 列表 |
| [HYGON-AI](https://github.com/HYGON-AI) 公开仓库目录 | GitHub 分页 API，完整取得 **48 个公开仓库** | 仓库名没有单独赛事包；这不等于每个文件都已读取 |
| [quality-gate](https://github.com/HYGON-AI/quality-gate/tree/5b03a2bb34e471fb3f84d1c22a34175d94eb1761) | 提交 `5b03a2bb34e471fb3f84d1c22a34175d94eb1761`，完整 84 项树，`truncated=false` | 增量代码合规、质量、安全门禁，不是模型精度评测 |
| [inference-cookbook-das](https://github.com/HYGON-AI/inference-cookbook-das/tree/69f207ba8c466b77ace1a56115cc4b9ecd63fdc5) | 提交 `69f207ba8c466b77ace1a56115cc4b9ecd63fdc5`，完整 135 项树，`truncated=false` | 存在 Qwen3-ASR/ForcedAligner 等部署配方；未见本赛事三题固定评测包 |
| [SkillHub](https://github.com/HYGON-AI/SkillHub/tree/130aa48c2c55ce3366e01daf54e8494b80d78138) | 提交 `130aa48c2c55ce3366e01daf54e8494b80d78138`，完整 88 项树，`truncated=false` | 通用 agent skill 发布/贡献流程，没有赛事评测入口 |
| GitHub 精确代码检索 | 三模型名配 `verify.py`/baseline、赛事名、HYGON 范围内 `verify.py`/baselines/scnet/黑客松 | 结果逐项见下一节；所有本轮查询结果均在单页范围内完整取得，赛事名另用 `per_page=100` 补齐全部 31 项 |
| 搜索引擎定向发现 | 16 条查询，含官方域名过滤和精确模型名 | Recon 模型卡可定位；没有新发现赛事独立附件下载地址。新闻与第三方模型监控不作为技术结论依据 |

## 3. 精确搜索命中如何解释

| 查询 | 命中数 | 核查结论 |
|---|---:|---|
| `Recon2Reason filename:verify.py` | 0 | 未定位精确判分入口 |
| `Recon2Reason baseline` | 3 | 第三方 release watch、模型注册表及归档状态，不是精度参考输出或评分基线 |
| `RoboBrain2.5-8B-NV filename:verify.py` | 0 | 未定位精确判分入口 |
| `RoboBrain2.5-8B-NV baseline` | 21 | 论文摘录、第三方空间评测/归档等；没有来自赛事/HYGON 的固定验收包 |
| `Qwen3-ASR-1.7B filename:verify.py` | 1 | 第三方歌词校验项目 `backend/lyric_verify.py`；文件名包含 verify 不代表赛事评分器 |
| `"国芯国模"` | 31 | 新闻、投资/财报、资讯归档；没有对应黑客松评测包 |
| `org:HYGON-AI filename:verify.py` | 15 | 框架内核、推测解码、ONNX 等校验代码；没有三模型赛事判分器 |
| `org:HYGON-AI baselines` | 55 | 框架测试/性能回归设施，不能据关键词认定官方比赛 baseline |
| `org:HYGON-AI Recon2Reason` / `hackathon` / `"黑客松"` / `scnet` / `Qwen3-ASR baseline` | 均 0 | 限定当前 GitHub 索引默认分支，不含私有仓库及所有历史分支 |

另对 GitLab 15891 进行了四次 `scope=blobs` 内容检索，查询为 Recon2Reason、国芯国模、RoboBrain2.5、Qwen3-ASR，均返回 **401**。没有尝试新的账号登录、绕过权限或声称已完成全站源码搜索；完整递归树及所选小型文件确实已经读取。

## 4. Recon 精确发布方资料再次确认

[BAAI 精确模型元数据](https://huggingface.co/api/models/BAAI/Recon2Reason-Reasoning-4B) 当前 revision 仍为 `f8faf41a1d5fe392f2d1ee31884f79398da59c99`。完整 19 个 sibling 文件名包括模型文件、tokenizer、配置、`SHA256SUMS` 和 `verify_release.py`，没有赛事数据/判分目录。再次下载的 [README](https://huggingface.co/BAAI/Recon2Reason-Reasoning-4B/blob/f8faf41a1d5fe392f2d1ee31884f79398da59c99/README.md)、[verify_release.py](https://huggingface.co/BAAI/Recon2Reason-Reasoning-4B/blob/f8faf41a1d5fe392f2d1ee31884f79398da59c99/verify_release.py) 和 `generation_config.json` 均保存哈希。

README 的 Evaluation 只给 BF16、PyTorch 2.8.0、Transformers 4.57.1、SDPA、greedy 和 GPU 型号，没有列出具体 benchmark 样本、得分或判分器。`verify_release.py` 仍是参数/token 完整性及可选白图单 token 检查，不能写成正式精度验证。公开 HF 数据集名称检索 `Recon2Reason` 也没有找到同名发布数据集；这不能排除使用其他名称发布。

## 5. 可以复用但不能改称赛事材料的新增代码

[FLEURS 适配器](https://developer.sourcefind.cn/codes/OpenDAS/evalscope_dataset/-/blob/b36f21f3c5cc7c13461c6047377d4e6c58780db1/evalscope/benchmarks/fleurs/fleurs_adapter.py) 使用 `lmms-lab/fleurs`，并非此前列出的原始 `google/fleurs` 下载入口；使用时必须固定实际数据来源、版本和 split，不能把二者当成同一份已校验样本。

[WER 源码](https://developer.sourcefind.cn/codes/OpenDAS/evalscope_dataset/-/blob/b36f21f3c5cc7c13461c6047377d4e6c58780db1/evalscope/metrics/text_normalizer/wer.py) 具有英语及中文规范化，对 `cmn_hans`/`yue_hant` 使用字符序列编辑距离，对其他语言使用词序列，并以总编辑距离除以参考总长度聚合。这为 ASR 开发评分提供了现成参考；是否就是本赛事口径仍未知，不能擅自替代。

[HYGON SGLang nightly precision](https://github.com/HYGON-AI/sglang-das/blob/f89209aa84a4fd0619d0662f6adf6980740efc78/docs/docs/references/nightly_precision_regression.mdx) 文档实际描述 8×H200 的跨轮 hidden-state 回归、两 token 固定 prompt 和 rolling baseline；不是 ASR WER、图像问答精度或动作/坐标参考。下载记录保存了实际读取的固定 commit。没有执行其上传 baseline 或联网写操作。

## 6. 仍需取得的精确材料

| 模型 | 缺少的赛事绑定材料 |
|---|---|
| **Recon2Reason-Reasoning-4B** | 适用的题型/指标；固定图片与样本 manifest、标注；对应 GPU 数值 baseline 或逐样本参考输出；原版评分器及答案抽取；固定解码和图像预处理参数；性能所用图片/分辨率口径 |
| **RoboBrain2.5-8B-NV** | 该 2.5 NV 版本的固定图像/深度输入；对应 GPU 坐标/动作输出；官方解析、聚合、单位及适用门槛；固定生成/预处理；性能负载。旧 RoboBrain2.0 的工具不能直接替代 |
| **Qwen3-ASR-1.7B** | 音频 manifest、版本/split、参考转写；WER/CER 规范化与分词；对应 GPU baseline；流式与长音频的验收范围；正式音频性能方法、持续时间及 RTF/延迟聚合 |

三题还需由工程实际产生：完整可取镜像地址与 tag、准确的本题运行配置、实际精度与性能原始记录、干净环境复跑及报告。上述工程工作不能全部归因为缺材料。镜像 digest 可以辅助复现，未将其增加为模板未写的强制门槛。当前没有独立认领回执的开发前置条件。

## 7. 光合私有代码项目、Draft 与身份规范

光合当前自身的 [Projects API 帮助](https://developer.sourcefind.cn/codes/help/api/projects.md) 和[项目帮助](https://developer.sourcefind.cn/codes/help/user/project/index.md) 均匿名读取 HTTP 200，确实说明 `private`、`internal`、`public` 三类 visibility；`private` 必须逐用户明确授权访问。这比只引用泛用 GitLab 资料更接近当前托管实例的证据。帮助 HTML 原文、纯文本和哈希均已保存。**本轮没有核验当前用户在光合的登录身份、namespace、创建权限或管理员可选项，不能从 SCNet/GitHub 身份推导这些权限。**

[GitLab 的 Draft 文档](https://docs.gitlab.com/user/project/merge_requests/drafts/) 说明 Draft 是合并请求状态。没有发现“Draft 仓库”是单独隐私类型的说明；项目名或 README 写 Draft 不会改变项目访问级别。准备阶段如确有需要创建非公开项目，应核验实际 `visibility=private`，并核实最终比赛评审如何访问；不自动推断私有链接可作为最终公开提交。

光合 [ModelZooStd.md](https://developer.sourcefind.cn/codes/modelzoo/template/-/blob/88a5074ab8b2c88bbf24de306ffcab97ee9e87a4/ModelZooStd.md) 固定提交 `88a5074ab8b2c88bbf24de306ffcab97ee9e87a4` 的要求是：

- `modelCode` 使用创建后真实 GitLab Project ID，不编造、不拿模型 clone ID 替代。
- `modelName` 使用官方模型系列名称；`modelDescription` 尽量在 50 字内。
- `processType`、`appCategory`、`frameType`、`accelerateType` 如实填写，算法类别与 `icon.png` 配套；实际设备只识别为 BW 时不擅自补造 BW1000 后缀。
- 根目录包含 `model.properties`、`LICENSE`、`icon.png`、依赖与 README，提供迷你数据；保留上游 README 为 `README_origin.md`。除权重外尽量不超过 50 MB。

这些是社区仓库元数据规范，不是额外的赛事认领或正式评分门槛。创建代码仓库、上传代码和发送最终比赛表单是不同操作；本轮只读核查没有做其中任何写操作。

## 8. RoboBrain 不开卡仍可继续完成的工作

以下是静态工程清单，不需要为寻找不到新材料而继续无限检索：

1. **扩充新 SHA 的 GPU 复验脚本。**当前 CPU 版修复头广播预算，GPU validator 还应加入相同 `Hq=1/Hkv>1` 与 batch 广播几何，准备好逐块预算、FP32/BF16 误差及 guard 元数据，之后用实际 BW 执行。
2. **准备大图和不同长宽比的开发 manifest。**本地生成已知框/点真值，记录图片哈希、预期视觉 patch 长度、请求参数、严格输出解析与异常/截断分类；不能把合成图直接命名为官方精度集。
3. **固定发行与依赖输入。**整理只使用当前受支持基础栈的安装/启动说明、运行文件清单、源码 hash guard、失败退出规则和依赖约束；未知的完整镜像地址留缺项，不能猜测。
4. **完成端到端评测和性能接口。**用本地 CPU 假服务验证请求计时、流式错误、并发结果聚合、日志不可覆盖及 manifest 身份；官方负载/判分器到位后接入，禁止制造替代正式分数。
5. **补齐报告和仓库交付布局。**把历史 baseline 错误、固定形状诊断、历史通用 GPU 成功和当前 CPU 修改版分开；准备真实模型来源、第三方许可、迷你功能样本及 `model.properties` 待填写字段。

已完成的默认兼容入口统一、头广播预算修复和 CPU 测试不重复列为未完成项。实际 GPU 数值复验、真实大图负载、完整精度/性能和干净镜像运行必须取得相应运行证据后才能标为完成。
