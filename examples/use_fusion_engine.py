#!/usr/bin/env python3
"""示例二：直接用融合引擎（研究者用法）。

适用场景：你已经用别的工具拿到了各模态结果（例如学校的多麦克风阵列、
或自己训练的模型输出），只想要计分与诊断部分。

注意：这里传入的必须是**你自己的真实测量数据**。
本示例用极简数值仅为了说明接口形状，请勿把示例数值当作分析结论。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collaborative_learning_analyzer.config import FusionEngineConfig  # noqa: E402
from collaborative_learning_analyzer.data_models import (  # noqa: E402
    AudioAnalysisResult,
    IndividualContribution,
    SemanticAnalysisResult,
    VideoAnalysisResult,
)
from collaborative_learning_analyzer.fusion_engine import FusionEngine, hhi_evenness  # noqa: E402


def main() -> int:
    # ---- 你自己的语音测量：每个成员的实际发言时长（秒）与发言轮次 ----
    speaking = {"stu_A": 95.0, "stu_B": 70.0, "stu_C": 55.0, "stu_D": 18.0}
    turns = {"stu_A": 12, "stu_B": 9, "stu_C": 7, "stu_D": 2}

    audio = AudioAnalysisResult(
        speaker_stats={
            member: {
                "total_duration": seconds,
                "turns": turns[member],
                "longest_monologue": seconds / max(1, turns[member]) * 2,
                "turn_taking_score": 0.75,
            }
            for member, seconds in speaking.items()
        },
        total_duration=420.0,
        total_speaking_time=sum(speaking.values()),
        diarization_applied=True,
    )

    # ---- 你自己的语义分析结果（逐人） ----
    semantic = SemanticAnalysisResult(
        topic_relevance=0.82,
        opinion_collisions=9,
        argument_depth_score=0.71,
        consensus_quality=0.66,
        turn_taking_pattern="balanced",
        per_speaker={"stu_A": 0.88, "stu_B": 0.74, "stu_C": 0.69, "stu_D": 0.31},
        windows_analyzed=4,
        complete=True,
    )

    # ---- 视觉结果：没做就传空的；融合层会把它排除在计分之外并给出告警 ----
    video = VideoAnalysisResult()

    engine = FusionEngine(FusionEngineConfig())
    report = engine.fuse(
        audio,
        video,
        semantic,
        group_id="demo_group",
        member_ids=list(speaking),
    )

    print(f"健康分 {report.overall_health_score:.1f}（{report.health_level.value}）")
    print(f"活跃度 {report.participation_activity:.3f} 均衡度 {report.evenness:.3f}")
    print(f"质量分 {report.quality_score:.3f} 主题相关 {report.topic_relevance}")
    print(f"交互深度 {report.interaction_depth}")
    for contribution in report.individual_contributions:
        print(
            f"  {contribution.person_id}: 份额 {contribution.speaking_share:.0%} "
            f"语义 {contribution.semantic_score:.2f} 贡献度 {contribution.total_score:.3f}"
        )
    print("\n诊断：")
    for item in report.diagnoses:
        print(f"  - {item}")

    print("\n均衡度函数可直接单独使用（份额完全均等 = 1，一人独占 = 0）：")
    print(f"  hhi_evenness([1,1,1,1]) = {hhi_evenness([1, 1, 1, 1]):.3f}")
    print(f"  hhi_evenness([1,0,0,0]) = {hhi_evenness([1, 0, 0, 0]):.3f}")
    print(f"  hhi_evenness([0,0,0,0]) = {hhi_evenness([0, 0, 0, 0]):.3f}  # 全员沉默")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
