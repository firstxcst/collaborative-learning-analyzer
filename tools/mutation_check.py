#!/usr/bin/env python3
"""变异测试：向源码注入确定性缺陷，确认测试套件真的会失败。

为什么需要它
------------
旧版本的测试断言是同义反复（例如对已被 clamp 的值断言 ``0 <= x <= 100``），
向计分逻辑注入 3 处严重缺陷后仍然 **6/6 全绿** —— 也就是零回归防护。
本脚本用「改坏了必须变红」来验证测试的有效性。

每个变异：备份原文件 → 应用改动 → 跑测试 → 还原。
只要有一个变异是「测试仍全绿」，脚本就以非零码退出。

用法::

    python tools/mutation_check.py            # 全部变异
    python tools/mutation_check.py --list     # 只列出变异
    python tools/mutation_check.py --only evenness_silent
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"


@dataclass
class Mutation:
    """一处故意注入的缺陷。"""

    name: str
    path: str
    old: str
    new: str
    description: str


MUTATIONS: list[Mutation] = [
    Mutation(
        name="evenness_silent_returns_one",
        path="fusion_engine.py",
        old="    total = sum(shares)\n    if total <= 0 or not shares:\n        return 0.0",
        new="    total = sum(shares)\n    if total <= 0 or not shares:\n        return 1.0",
        description="还原旧缺陷：全员沉默时均衡度返回满分 1.0（旧版因此给沉默组 90 分）",
    ),
    Mutation(
        name="evenness_uses_absolute_score",
        path="fusion_engine.py",
        old="    probs = [max(0.0, s) / total for s in shares]\n    hhi = sum(p * p for p in probs)",
        new=(
            "    probs = [max(0.0, s) for s in shares]\n"
            "    mean = sum(probs) / max(1, len(probs))\n"
            "    hhi = sum(((p - mean) / (mean or 1.0)) ** 2 for p in probs) / max(1, len(probs))\n"
            "    return max(0.0, min(1.0, 1.0 - hhi * 2))"
        ),
        description="把均衡度改回「基于绝对分的标准差」思路（对水平高低敏感）",
    ),
    Mutation(
        name="health_drops_activity_gate",
        path="fusion_engine.py",
        old="        raw = 100.0 * _clamp01(activity) * _clamp01(quality)",
        new="        raw = 100.0 * _clamp01(quality)",
        description="去掉乘性活跃度门控，沉默组重新变得可以拿高分",
    ),
    Mutation(
        name="depth_computed_when_semantic_missing",
        path="fusion_engine.py",
        old=(
            "        depth = (\n"
            "            self._interaction_depth(semantic_result, audio_result.total_duration)\n"
            "            if semantic_usable\n"
            "            else None\n"
            "        )"
        ),
        new="        depth = self._interaction_depth(semantic_result, audio_result.total_duration)",
        description="语义不可用时仍计算交互深度，等于把编造的 0 分混进质量分",
    ),
    Mutation(
        name="weights_not_renormalised",
        path="fusion_engine.py",
        old=(
            "        total = sum(self.config.contribution_weights[d] for d in used)\n"
            "        return {d: self.config.contribution_weights[d] / total for d in used}"
        ),
        new="        return {d: self.config.contribution_weights[d] for d in used}",
        description="缺失维度不重新归一化权重，破坏权重和为 1 的契约",
    ),
    Mutation(
        name="silent_parse_defaults_to_neutral",
        path="semantic_agent.py",
        old=(
            '        usable = [\n'
            '            result["topic_relevance"],\n'
            '            result["opinion_collisions"],\n'
            '            result["argument_depth_score"],\n'
            '        ]\n'
            '        if all(v is None for v in usable):\n'
            '            raise SemanticParseError("关键字段全部缺失或类型非法")\n'
            "        return result"
        ),
        new=(
            '        for key, fallback in (("topic_relevance", 0.5),\n'
            '                              ("argument_depth_score", 0.5),\n'
            '                              ("consensus_quality", 0.5)):\n'
            '            if result[key] is None:\n'
            '                result[key] = fallback\n'
            "        return result"
        ),
        description="还原旧缺陷：解析失败静默降级为 0.5 中性分",
    ),
    Mutation(
        name="load_drops_contributions",
        path="data_models.py",
        old=(
            "            individual_contributions=[\n"
            "                IndividualContribution.from_dict(c)\n"
            "                for c in data.get(\"individual_contributions\", [])\n"
            "            ],"
        ),
        new="            individual_contributions=[],",
        description="还原旧缺陷：load() 丢掉个体贡献度",
    ),
    Mutation(
        name="whisper_device_accepts_auto",
        path="audio_agent.py",
        old='    if cleaned in {"", "auto"}:\n        return None',
        new='    if cleaned == "":\n        return None',
        description='还原旧缺陷：把 "auto" 原样传给 whisper，torch.device("auto") 会抛错',
    ),
    Mutation(
        name="merge_assigns_all_overlaps",
        path="audio_agent.py",
        old=(
            "            best_idx, best_overlap = -1, 0.0\n"
            "            for idx, dia in enumerate(dia_segments):\n"
            "                ov = self._overlap(asr, dia)\n"
            "                if ov > best_overlap:\n"
            "                    best_idx, best_overlap = idx, ov\n"
            "            if best_idx >= 0 and best_overlap >= 0.5 * max(asr.duration, 1e-6):\n"
            "                buckets[best_idx].append(asr.text)"
        ),
        new=(
            "            for idx, dia in enumerate(dia_segments):\n"
            "                if self._overlap(asr, dia) > 0:\n"
            "                    buckets[idx].append(asr.text)"
        ),
        description="还原旧缺陷：把同一句话重复分配给所有重叠的说话人轮次",
    ),
    Mutation(
        name="cohesion_hardcoded_default",
        path="video_agent.py",
        old="        if not result.person_attention_stats or analyzed_duration <= 0:\n            return None",
        new="        if not result.person_attention_stats or analyzed_duration <= 0:\n            return 0.5",
        description="还原旧缺陷：没有视觉数据时返回硬编码 0.5，凭空编造中等值",
    ),
    Mutation(
        name="vision_imports_cv2_eagerly",
        path="video_agent.py",
        old="from __future__ import annotations\n\nimport math",
        new="from __future__ import annotations\n\nimport cv2  # noqa: F401\nimport math",
        description="还原旧缺陷：视觉依赖在模块顶层导入，纯语音路径也被迫安装 opencv",
    ),
]


def apply_mutation(mutation: Mutation, target: Path) -> None:
    text = target.read_text(encoding="utf-8")
    if mutation.old not in text:
        raise RuntimeError(
            f"变异 {mutation.name} 的锚点文本未找到：{target.name}\n"
            f"期望片段：\n{mutation.old}"
        )
    target.write_text(text.replace(mutation.old, mutation.new, 1), encoding="utf-8")


def run_tests(extra_args: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", *extra_args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def main() -> int:
    parser = argparse.ArgumentParser(description="变异测试：验证测试套件能否检出缺陷")
    parser.add_argument("--list", action="store_true", help="列出全部变异")
    parser.add_argument("--only", help="只运行指定名称的变异")
    parser.add_argument("--pytest-args", default="", help="额外传给 pytest 的参数")
    args = parser.parse_args()

    if args.list:
        for mutation in MUTATIONS:
            print(f"{mutation.name:38} {mutation.path:20} {mutation.description}")
        return 0

    selected = [m for m in MUTATIONS if not args.only or m.name == args.only]
    if not selected:
        print(f"未找到名为 {args.only} 的变异", file=sys.stderr)
        return 2

    extra = args.pytest_args.split() if args.pytest_args else []

    print("=" * 78)
    print("基线：未注入缺陷")
    print("=" * 78)
    baseline_code, baseline_out = run_tests(extra)
    baseline_line = [l for l in baseline_out.splitlines() if "passed" in l or "failed" in l]
    print(f"  退出码 {baseline_code}  {baseline_line[-1] if baseline_line else ''}")
    if baseline_code != 0:
        print("基线测试未通过，无法进行变异测试。请先修好测试。", file=sys.stderr)
        print(baseline_out[-3000:])
        return 2

    print()
    print("=" * 78)
    print(f"注入 {len(selected)} 处缺陷，逐一确认测试会变红")
    print("=" * 78)

    survivors: list[Mutation] = []
    with tempfile.TemporaryDirectory(prefix="cla_mutation_") as tmp:
        backup_dir = Path(tmp)
        for mutation in selected:
            target = SRC / mutation.path
            backup = backup_dir / mutation.path
            shutil.copy2(target, backup)
            try:
                apply_mutation(mutation, target)
                code, output = run_tests(extra)
                killed = code != 0
                summary = [
                    l for l in output.splitlines()
                    if "passed" in l or "failed" in l or "error" in l.lower()
                ]
                print(
                    f"  [{'检出' if killed else '漏检'}] {mutation.name:38} "
                    f"-> {summary[-1] if summary else ''}"
                )
                if not killed:
                    survivors.append(mutation)
            finally:
                shutil.copy2(backup, target)

    print()
    print("=" * 78)
    if survivors:
        print(f"结果：{len(survivors)}/{len(selected)} 处缺陷未被检出 —— 测试仍有盲区")
        for mutation in survivors:
            print(f"  - {mutation.name}: {mutation.description}")
        return 1
    print(f"结果：{len(selected)}/{len(selected)} 处缺陷全部被检出。测试具备回归防护能力。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
