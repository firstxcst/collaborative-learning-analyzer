"""命令行入口。

修复的审计问题 #36：旧仓库没有任何 ``console_scripts`` 入口，``src/__main__.py`` 白写，
装完之后没有命令可用。现在 ``cla`` 命令可用，并且 ``python -m`` 也能跑。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .config import get_paths
from .pipeline import analyze


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cla",
        description="协作学习分析助手：基于多模态分析生成小组协作过程性评价报告",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("analyze", help="分析课堂小组讨论音视频")
    run.add_argument("--audio", help="音频文件路径")
    run.add_argument("--video", help="视频文件路径（可选）")
    run.add_argument(
        "--transcript",
        help="已有转录文件（JSON）。分会场麦克风 / 教师字幕 / 外部 ASR 结果，可跳过本地 ASR",
    )
    run.add_argument("--members", type=int, default=4, help="小组成员数量（默认 4）")
    run.add_argument("--group-id", default="group_1", help="小组标识")
    run.add_argument("--topic", help="讨论主题，用于判断主题相关度")
    run.add_argument("--output", default="report.json", help="报告输出文件名")
    run.add_argument(
        "--skip-diarization",
        action="store_true",
        help="跳过说话人分离（会丧失个体区分能力，报告中会标注）",
    )
    run.add_argument("--skip-video", action="store_true", help="跳过视觉分析")
    run.add_argument("--skip-semantic", action="store_true", help="跳过语义分析")
    run.add_argument(
        "--align-profiles",
        action="store_true",
        help="用已注册声纹把说话人标签对齐到学生身份",
    )
    run.add_argument("--annotated-video", help="输出标注视频的路径")

    sub.add_parser("paths", help="显示运行时使用的目录")
    return parser


def _cmd_analyze(args: argparse.Namespace) -> int:
    if not args.audio and not args.transcript:
        print("错误：必须提供 --audio 或 --transcript 至少一项", file=sys.stderr)
        return 2

    output_path = get_paths(create=True).output / args.output

    report = analyze(
        audio_path=args.audio,
        video_path=args.video,
        num_members=args.members,
        group_id=args.group_id,
        topic=args.topic,
        transcript_path=args.transcript,
        skip_diarization=args.skip_diarization,
        skip_video=args.skip_video,
        skip_semantic=args.skip_semantic,
        align_profiles=args.align_profiles,
        save_annotated_video=args.annotated_video,
    )
    report.save(str(output_path))

    completeness = report.data_completeness
    print("=" * 62)
    print(f"小组 {report.group_id} · 协作健康分 {report.overall_health_score:.1f}"
          f"（{report.health_level.value}）")
    print("=" * 62)
    print(f"  时长        : {report.total_duration:.1f} 秒")
    print(f"  成员        : {', '.join(report.member_ids) or '（未识别）'}")
    print(f"  活跃度      : {report.participation_activity:.2f}")
    print(f"  均衡度      : {report.evenness:.2f}")
    print(f"  主题相关度  : " + ("不可用" if report.topic_relevance is None else f"{report.topic_relevance:.2f}"))
    print(f"  交互深度    : " + ("不可用" if report.interaction_depth is None else f"{report.interaction_depth:.2f}"))
    print(f"  模态完整性  : " + ", ".join(
        f"{k}={'有' if v else '无'}" for k, v in completeness.items()
    ))
    if report.diagnoses:
        print("\n诊断：")
        for item in report.diagnoses:
            print(f"  - {item}")
    if report.suggestions:
        print("\n建议：")
        for item in report.suggestions:
            print(f"  - {item}")
    if report.warnings:
        print("\n⚠️ 过程告警（数据完整性问题，请务必阅读）：")
        for item in report.warnings:
            print(f"  - {item}")
    print(f"\n报告已保存：{output_path}")
    return 0


def _cmd_paths(_args: argparse.Namespace) -> int:
    paths = get_paths()
    print(json.dumps(
        {
            "project_root": str(paths.project_root),
            "data": str(paths.data),
            "output": str(paths.output),
            "models": str(paths.models),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "paths":
        return _cmd_paths(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
