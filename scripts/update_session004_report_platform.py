#!/usr/bin/env python3
"""Incorporate verified platform observations into the six editable reports."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUGS = ('recon2reason-bw-vllm', 'robobrain-bw-vllm', 'qwen3-asr-bw-vllm',
         'mimo-v2.6-distill-qwen9b-bw-vllm', 'trident-qwen3vl2b-bw-transformers',
         'qwen3-forcedaligner-bw-transformers')


def main():
    platform = json.loads((ROOT / 'evidence/platform-checkpoint-20261023-004.json').read_bytes())
    assert platform['all_instances_stopped'] is True
    for slug in SLUGS:
        path = ROOT / f'docs/report-004-{slug}.md'
        text = path.read_text(encoding='utf-8')
        marker = '<!-- session004-platform-closure -->'
        if marker in text:
            raise ValueError('Already incorporated: ' + slug)
        paragraph = (marker + '\n\n'
            '本轮已独立核对[SCNet 平台关机与余额记录](../evidence/platform-checkpoint-20261023-004.json)：'
            'A/B 两个本轮实例及旧参考实例均为“已关机”；A 手动关机并选择保存环境，B 按预定时限自动关机。'
            '可见余额为 **488.0238888888 卡时**，与开机前可见余额差 4.6686111111 卡时；'
            '余额差可能包含延迟结算，不当作最终账单。资源为华中一区 A 区 113 组、BW64GB×1、CPU15核、内存59GB。'
            '账户镜像 ID `2071293281204580354` / version `2082674417745739777`，'
            'Notebook 可选择标签 `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`；'
            '完整实际 registry/repo:tag 和评审获取方式尚缺，digest 仅为可复现补充项。'
            '平台配置不替代任务原始环境/驱动实测。\n\n'
            f'本轮代码仓库目标：[zhl-2022/guoxin-{slug} 的 codex/session004 分支]'
            f'(https://github.com/zhl-2022/guoxin-{slug}/tree/codex/session004)。'
            '仓库保持私有；实际推送 commit 和逐文件回读以独立交付回执为准，评审权限仍须落实。'
            '光合既有三仓库的 session003 与本轮 GitHub 分支分别记录，不假称光合已同步新版。\n\n')
        heading = '## 2. 验证环境\n\n'
        if heading not in text:
            raise ValueError('Section two missing: ' + slug)
        text = text.replace(heading, heading + paragraph, 1)
        text = text.replace('当前最终 stage004 尚未创建；', '最终文件身份以 stage004 的 repository-manifest004.json 与独立构建回执为准；')
        text = text.replace('stage004 本地目录及本轮远端版本尚待单独装配核验。',
                            '本轮 stage004 和远端分支按独立构建、推送及回读回执核对。')
        if slug == 'recon2reason-bw-vllm':
            text = text.replace('| 实例 ID、CPU/内存/存储、实际设备 | 待 session004 平台与 env 原件 |',
                '| 实例 ID、CPU/内存/存储、实际设备 | A `2112186448839770114`；平台 BW64GB×1/15核/59GB；存储配额与独立设备型号清单未新增认证 |')
            text = text.replace('| 服务映射及截图 | 待本轮事实；未使用映射时写无 |',
                '| 服务映射及截图 | 服务仅 127.0.0.1:30000，无自定义外网服务映射；关机截图在本轮平台记录中 |')
        with path.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        print(slug)


if __name__ == '__main__':
    main()
