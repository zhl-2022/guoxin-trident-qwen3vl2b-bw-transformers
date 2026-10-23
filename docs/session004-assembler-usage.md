# Session004 六仓库本地装配使用说明

新入口为 [assemble_session004_repositories.py](../scripts/assemble_session004_repositories.py)，与旧三题装配器独立。它只生成全新本地源码目录，不调用 Git、创建远端仓库、压缩最终 ZIP、启动模型或提交赛事表单。实际 GPU 原件和正式结论须由主流程核对后写入报告；装配成功只证明文件快照和入口约束。

## 1. 已实现的边界

- 六个 slug、模型 ID、赛道和 `report-004-<slug>.md` 映射写在显式 registry；不从目录猜测目标题。
- 原三题、MiMo 及 Trident 共五题生成透明根七个 Shell 入口与 `verify.py`，参数直接传给 canonical task。ForcedAligner 是唯一仍使用根 `entry.py` 的目标，不生成尚未实现的标准服务入口。
- Trident 的标准入口代码齐全与 GPU HTTP 验证分开：`framework=vllm`、`framework_execution_mode_reported_by_task=native_vllm_pooling_conversion` 描述已归档的离线 `LLM.embed` 路径；`offline_pooling_GPU_verified_reported_by_task=true` 明确来自 task/profile 记录。assembler 不运行或独立认证 GPU。根 `ROOT_TASK.json` 继续固定 `HTTP_service_GPU_verified=false`、`official_embedding_evaluator_bridge_implemented=false`、`standard_submission_ready=false`，不能从 `service=True` 推导它们通过。
- 每份根 README 必须正好有模板的八节。报告普通链接和引用式链接按 `docs/` 原路径解析后重定位到仓库根，代码围栏不改动，百分号编码和 fragment/query 保留有效语义。
- 收录六题源码、小配置、锁、fixture、必要 CPU tests、报告明确链接的证据和少量固定回归输入。三个旧装配器集成测试不随新仓库收录，它们绑定旧三题报告/collector；排除名单会写在 dry-run 输出中。
- 固定中文模板校验已知 SHA。两个 upstream ZIP 同时校验固定整包 SHA、revision、CRC、成员集合、逐文件大小/SHA 和路径类型。额外 wheel 只允许固定的 `packaging 25.0` 验证 fixture，不安装它。
- 默认排除实时 runtime/logs、权重、venv、缓存、Git 和凭据文件；文本再检查明显凭据赋值与私钥。扫描错误直接抛出，不能用静默漏文件获得“通过”。
- `--extra-evidence` 只接受 `evidence/session004/` 下明确归档的原件；此时可以保存归档包中的 `runtime/server.log` 等记录，仍不允许实时 `tasks/*/runtime`、权重或凭据。该开关不认证官方评分，也不自动证明导出包来源。
- 归档 ASR 环境仅有六个 prepare 元数据文件的窄例外：`validation.stdout`、`validation.stderr`、`ready.json`、`pip-install.log`、`native-before.json`、`missing-wheels.lock.txt`，路径须精确处于 `evidence/session004/<archive>/project/tasks/qwen3-asr-bw-vllm/runtime/audio-venv/asr-prepare-evidence/`。其他 venv 文件、嵌套文件和实时环境继续拒绝，不能据此收录整个 audio-venv。
- 输出必须是全新目录，已有空目录也拒绝。禁止输出到旧 `repositories-stage003` 或 canonical 源码/证据目录。正式 manifest 排除自身 hash，其他 payload 写入后逐文件回读，并记录字节数与 SHA。

## 2. 当前检查

实际 CPU 检查记录保存于 [validation.json](../evidence/session004/assembler-validation-20261023/validation.json)。Windows 和 WSL 都只在系统临时目录构建六份仓库并回读；Windows 跳过的实际 Bash 检查在 WSL 执行。检查还覆盖缺官方 selfcheck 返回 2、argv 透传、拒绝符号链接/路径穿越/异常 ZIP/权重/凭据、文件系统扫描错误和写入前校验。

ASR 归档窄例外与负面策略测试的后续验证见 [validation003](../evidence/session004/assembler-validation-003-20261023/final-check.json)。Trident 纳入五题标准入口的本轮目标记录保存于 `evidence/session004/assembler-validation-004-20261023/`，最终状态见该目录的 `final-check.json`；首轮验证器的缺失测试导入和 skip 计数序列化错误保留原始日志，修正后使用独立重试记录。它只验证代码与状态隔离，不执行 GPU、推送或实际交付 stage。Trident `verify/bench` 目前是明确阻塞入口，缺 embedding 官方调用桥和评测契约，不能把根 `verify.py` 存在说成官方桥已经实现。

运行目标检查命令：

```powershell
python -B -m unittest discover -s tests -p test_assemble_session004_repositories.py -v
wsl --distribution Ubuntu --exec python3 -B -m unittest discover -s /mnt/e/hg20261019/tests -p test_assemble_session004_repositories.py -v
```

## 3. 接入本轮 GPU 原件后 finalize

1. 从 SCNet 导出本轮真实原件，核对 exporter seal、文件列表、大小和 SHA，保留实际失败与不支持记录。把核验后的解包目录放入新的 `evidence/session004/<export-name>/`，不覆盖旧 003 或本轮先前记录。
2. 按 [精确字段表](stage004-report-fields.json) 更新六份报告与必要索引。正式材料未取得时保留待补；不要从 CPU 通过、功能回答或客户端吞吐推导正式精度通过。
3. 先 dry-run，核对输出任务身份、体积、排除项和归档原件列表。下例显式纳入已解包的中间与最终两份归档；执行前仍需核对它们的 seal。当前变更未创建实际交付 stage。

```powershell
python -B scripts/assemble_session004_repositories.py --output E:/hg20261019/dist/repositories-stage004 --extra-evidence evidence/session004/remote-intermediate-001 --extra-evidence evidence/session004/remote-final-001
```

4. 确认报告与实际原件一致后，才向同一个全新目录执行 `--build`。如果已有输出，选择新的版本路径；不自动合并、删除或覆盖。

```powershell
python -B scripts/assemble_session004_repositories.py --output E:/hg20261019/dist/repositories-stage004 --extra-evidence evidence/session004/remote-intermediate-001 --extra-evidence evidence/session004/remote-final-001 --build
```

5. 逐题检查 `ROOT_TASK.json`、根 README、manifest 和缺材料行为，再单独安排代码版本冻结、访问方式和最终压缩材料。本脚本不创建提交 SHA、公开链接或赛事回执；最后提交由用户手动完成。

只装配某一目标可追加 `--task <registered-slug>`；共享测试依赖仍保留六题源码，但 `ROOT_TASK.json` 只声明本仓库唯一目标题。
