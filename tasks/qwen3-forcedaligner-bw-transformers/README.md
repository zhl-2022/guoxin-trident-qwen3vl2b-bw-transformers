# Qwen3-ForcedAligner：session004 GPU时间戳工程参照

真实 GPU runner 已输出13个汉字边界，开发契约通过；**TF-only计件资格、原生框架标准服务、正式AAS/性能尚未完成，submission_ready=false。** 完整八节报告见 [技术报告](../../docs/report-004-qwen3-forcedaligner-bw-transformers.md)。

## 实测结果

- 模型 `Qwen/Qwen3-ForcedAligner-0.6B`，固定revision `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`；完整远端checkpoint SHA通过。
- 固定官方源码 QwenLM/Qwen3-ASR `7c6daf77a2421100f5fb066495372c00129d39ff`，BF16/SDPA，隔离Transformers4.57.6，复用native Torch2.10/DTK26.04。
- 输入4.2039375秒中文音频及“甚至出现交易几乎停滞的情况。”，保留13个有效文本单元，时间范围0.400–3.680秒。
- align调用12.206611秒、worker监督阶段72.990906秒（不含此前checkpoint hash），completed/exit0；没有独立timestamp reference，不能换算成AAS或正式吞吐。
- [完整原始结果](../../evidence/session004/remote-intermediate-001/candidates/results/forcedaligner-pilot-001/worker-result.json)、[运行终态](../../evidence/session004/remote-intermediate-001/candidates/results/forcedaligner-pilot-001/run-final.json)、[环境](../../evidence/session004/remote-intermediate-001/candidates/results/forcedaligner-pilot-001/environment.json)。原结果SHA为 `c91746c4e7b826749418b407aaf2e46458af930ed382c70e416229ce8b394184`。

## 复跑入口

以下命令应从工程根在原SCNet候选环境执行，结果目录须新建；这里只提供复跑方式，本次本地文档更新没有开卡。

```bash
/root/private_data/guoxin-candidates/envs/forcedaligner-001/bin/python -B tasks/qwen3-forcedaligner-bw-transformers/entry.py run --model-dir /root/private_data/guoxin-candidates/models/forcedaligner --upstream-root /root/private_data/guoxin-candidates/source/qwen3-asr --requests tasks/qwen3-forcedaligner-bw-transformers/requests.example.json --output /root/private_data/guoxin-candidates/results/forcedaligner-pilot-NEW --max-seconds 360
```

本pilot保守音频上限180秒，已有中文短样本不证明其他语言/长音频覆盖。原生vLLM ASR架构的生成头不能直接代替本题5000类timestamp head，见 [架构与框架核查](../../docs/forcedaligner-native-route-20261023.md)。

## 交付边界

CPU依赖partial修复只补audioread、复用42项，原record和native metadata不变；GPU context discovery如实记录，不声称CPU-only。初次超时原件保留。

还缺独立对齐reference、正式AAS定义与GPU baseline/evaluator、原生框架资格或明确例外、标准服务和性能方法、可取完整镜像引用与干净复跑。最终源码与仓库由总装配补封；旧manifest尚不能当作本次修改后的最终seal。上游Apache-2.0与新代码版权说明保留，最后表单由用户手动提交。
