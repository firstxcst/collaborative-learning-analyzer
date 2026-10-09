#!/usr/bin/env python3
"""示例一：分析一次小组讨论（真实媒体夹具）。

用法::

    # 1) 先生成真实媒体夹具（Windows）
    python tools/make_fixtures.py --out fixtures

    # 2) 走已有转录（无需 ASR 推理，离线可用）
    python examples/analyze_discussion.py --transcript fixtures/discussion.transcript.json \
        --topic "为什么天空是蓝色的" --provider heuristic

    # 3) 走真实音频（需要安装 audio extra 与 whisper 权重）
    python examples/analyze_discussion.py --audio fixtures/discussion.wav \
        --topic "为什么天空是蓝色的" --provider heuristic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collaborative_learning_analyzer.config import (  # noqa: E402
    SemanticAgentConfig,
    get_paths,
)
from collaborative_learning_analyzer.pipeline import analyze  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="分析一次小组讨论")
    parser.add_argument("--audio", help="音频路径")
    parser.add_argument("--video", help="视频路径")
    parser.add_argument("--transcript", help="已有转录 JSON 路径")
    parser.add_argument("--topic", help="讨论主题")
    parser.add_argument("--members", type=int, default=4, help="成员数")
    parser.add_argument(
        "--provider",
        default="heuristic",
        choices=["heuristic", "openai", "dashscope", "vllm"],
        help="语义分析提供者。heuristic 为离线规则基线，不需要 API 密钥。",
    )
    parser.add_argument("--output", default="report.json")
    args = parser.parse_args()

    if not args.audio and not args.transcript:
        parser.error("必须提供 --audio 或 --transcript 至少一项")

    report = analyze(
        audio_path=args.audio,
        video_path=args.video,
        transcript_path=args.transcript,
        topic=args.topic,
        num_members=args.members,
        semantic_config=SemanticAgentConfig(provider=args.provider),
    )

    output = get_paths(create=True).output / args.output
    report.save(str(output))

    print(f"协作健康分：{report.overall_health_score:.1f}（{report.health_level.value}）")
    print(f"活跃度 {report.participation_activity:.2f} / 均衡度 {report.evenness:.2f}")
    for contribution in report.individual_contributions:
        print(
            f"  {contribution.person_id}: "
            f"发言 {contribution.speaking_seconds:.1f}s "
            f"({contribution.speaking_share:.0%}) 贡献度 {contribution.total_score:.3f}"
        )
    if report.warnings:
        print("\n过程告警：")
        for warning in report.warnings:
            print(f"  - {warning}")
    print(f"\n报告已保存：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
