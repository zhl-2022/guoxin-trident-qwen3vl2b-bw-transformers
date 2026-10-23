# Trident session005 外部评测执行契约

## 1. 已实现范围

`verify005.sh` 和 `bench005.sh` 调用新增 `scripts/trident_official_bridge_005.py`，透明执行调用者显式提供的 embedding evaluator、selfcheck 或负载脚本，保存原始 stdout/stderr、退出码、超时、输入前后 SHA 和声明输出的原始文件 SHA。桥不修改模型、tokenizer、指令或采样，不自动添加生成 1024 token、`BENCH_MODE=image` 或任何自选数据集。

成功状态只叫 `executed_uninterpreted`，含义是外部进程 exit0、声明输入未变、声明输出文件存在且可哈希。桥不解析官方分数字段、不判断容差、不认证脚本发布方，也不产生 official PASS。所有正式精度、性能和 `submission_ready` 保持 false。

这套桥在 HTTP GPU 实测后添加，只通过 CPU mock 执行合同。实际 HTTP GPU producer 仍是原 session004 标准源码，不把新增桥写入旧 GPU 源码 seal。旧任务 `verify.sh`、`bench.sh` 和 runtime 不修改；session005 独立组装可显式把根 verify/bench 指向这两个 005 wrapper，并记录 overlay 来源。

## 2. 必填材料

调用者须先核实官方材料的发布地址、revision、下载原件和授权范围。本桥只记录实际选中文件的身份，不能因为文件名含 official 就认定来源官方。下列路径均为本机明确路径，普通文件及其祖先不得是软链接或 Windows reparse point。

| 环境变量 | 契约 |
|---|---|
| `OFFICIAL_TRIDENT_DATA_DIR` | 已取得的数据目录 |
| `OFFICIAL_TRIDENT_DATA_FILES_JSON` | 非空 JSON 字符串数组，逐一列出相对数据目录的实际输入文件；不自动遍历未知目录 |
| `OFFICIAL_TRIDENT_QRELS` | 明确的 qrels 文件 |
| `OFFICIAL_TRIDENT_BASELINE` | 明确的 GPU baseline 文件；桥保持原文，不解释其 schema |
| `OFFICIAL_TRIDENT_OUTPUT_DIR` | 尚不存在的新输出目录，与数据树独立；已有目录永不覆盖 |
| `OFFICIAL_TRIDENT_TIMEOUT_SECONDS` | 外部进程执行上限，正数且不超过 7200 秒 |
| `OFFICIAL_TRIDENT_HASH_TIMEOUT_SECONDS` | 可选，每轮输入/输出哈希预算，默认 30 秒，最大 600 秒 |
| `OFFICIAL_TRIDENT_PYTHON_BIN` | 可选，外部 Python evaluator 解释器，默认使用当前桥解释器 |

针对具体阶段，还须显式设置下列三项，不提供默认脚本、参数或输出 schema：

| 命令 | 脚本 | 原样 argv 数组 | 输出文件契约 |
|---|---|---|---|
| `verify005.sh` | `OFFICIAL_TRIDENT_VERIFY_SCRIPT` | `OFFICIAL_TRIDENT_VERIFY_ARGS_JSON` | `OFFICIAL_TRIDENT_VERIFY_OUTPUTS_JSON` |
| `verify005.sh --selfcheck` | `OFFICIAL_TRIDENT_SELFCHECK_SCRIPT` | `OFFICIAL_TRIDENT_SELFCHECK_ARGS_JSON` | `OFFICIAL_TRIDENT_SELFCHECK_OUTPUTS_JSON` |
| `bench005.sh` | `OFFICIAL_TRIDENT_BENCH_SCRIPT` | `OFFICIAL_TRIDENT_BENCH_ARGS_JSON` | `OFFICIAL_TRIDENT_BENCH_OUTPUTS_JSON` |

参数数组即使为空，也必须显式写 `[]`。输出契约是非空相对文件路径数组，由实际 evaluator/负载脚本决定；桥不假定 `score`、`PASS`、Recall 或 nDCG 字段。外部脚本通过显式 argv 或这些环境变量读取数据、qrels、baseline 和输出位置；桥不猜测官方 CLI 旗标。已验证的脚本、data/qrels/baseline/output 环境路径在进入新输出工作目录前冻结为绝对路径，原样记录 child 环境绑定。额外 argv 字符串不改写，参数中的文件路径应由调用者明确提供绝对路径。脚本须在前台等待其工作结束，进程及仍在同一自有 session 中的后代会在退出/超时后清理。

`--selfcheck` 仍要求完整显式材料合同，可以让官方 selfcheck 自己决定如何校验材料。材料尚未提供时，实际命令返回 2 并打印具体缺项；不能把未运行写成通过。`--dry-run` 只检查合同并输出计划，不创建输出目录或启动外部进程。

## 3. 文件与执行边界

输入前后指纹仅覆盖 evaluator 脚本、任务身份文件、qrels、baseline 和显式声明的数据文件；未列出的数据目录内容未被证明保持不变。每个声明文件上限 256 MiB、每次总哈希上限 20 GiB、数据列表上限 4096 项。超出时需按正式数据合同另行设计分片，不跳过校验。

输入哈希、子进程运行、清理和输出哈希分阶段计时，`TIMEOUT_SECONDS` 只对应子进程执行阶段；不能把它当整条命令从首个输入哈希到最终落盘的唯一时限。stdout/stderr 合并保存为 `bridge-stdout.log`，轮询 4 MiB 日志上限，超过后停止并标记失败，不把截断或超限输出解释为有效结果。POSIX 使用自有进程 session；Windows CPU 测试只证明直属子进程清理。

桥只写新的输出目录，不恢复、覆盖或“修复”输入。外部程序若改动声明输入，记录前后身份差异并返回 2。该机制检测改变，不是运行任意第三方程序的安全沙箱。

原始文件包括 `bridge-start.json`、`bridge-inputs-before.json`、`bridge-stdout.log`、`bridge-final.json` 和 evaluator 按契约自行生成的输出。既有证据不覆盖，错误、非零退出、超时、输出缺失或输入改变都保留失败状态。得到官方 schema 与验收阈值后，仍须另实现并审核 interpretation，再真实运行正式验证。

## 4. 复现命令

先填入已经取得的真实材料环境变量，再在任务目录执行：

```bash
bash verify005.sh --dry-run
bash verify005.sh --selfcheck
bash verify005.sh
bash bench005.sh
```

每次实际执行使用不同的 `OFFICIAL_TRIDENT_OUTPUT_DIR`。CPU tests 中的临时数据和 opaque 输出仅用于桥合同回归，禁止作为模型精度、官方 baseline 或性能结果提交。
