"""计分模型的回归测试 —— 本文件是整个测试套件的核心。

这些断言全部是**可失败**的边界断言：它们针对旧实现的具体缺陷设定了期望行为，
任何一处逻辑退化都会立刻变成红灯（已用变异测试验证，见 docs/AUDIT_REMEDIATION.md）。

覆盖的审计问题：#1 #2 #3 #4 #5 #6 #7 #9 #28 #29
"""

from __future__ import annotations

import pytest

from collaborative_learning_analyzer.config import FusionEngineConfig, get_fusion_config
from collaborative_learning_analyzer.data_models import (
    AudioAnalysisResult,
    CollaborationLevel,
    SemanticAnalysisResult,
    VideoAnalysisResult,
)
from collaborative_learning_analyzer.fusion_engine import FusionEngine, hhi_evenness


def build_audio(
    speaking_seconds: list[float],
    duration: float = 300.0,
    turns_per_speaker: int | None = None,
) -> AudioAnalysisResult:
    """构造语音结果。这里的数值是**输入参数**，不是被伪造的测量结果。"""
    stats = {}
    for index, seconds in enumerate(speaking_seconds):
        turns = turns_per_speaker if turns_per_speaker is not None else max(
            1, int(seconds // 10) or 1
        )
        stats[f"s{index + 1}"] = {
            "total_duration": float(seconds),
            "turns": turns,
            "longest_monologue": float(seconds),
            "turn_taking_score": 0.8,
        }
    return AudioAnalysisResult(
        speaker_stats=stats,
        total_duration=duration,
        total_speaking_time=sum(speaking_seconds),
        diarization_applied=True,
    )


def build_semantic(
    topic: float | None = 0.8,
    collisions: int = 6,
    depth: float | None = 0.7,
    consensus: float | None = 0.6,
    per_speaker: dict[str, float] | None = None,
    complete: bool = True,
) -> SemanticAnalysisResult:
    return SemanticAnalysisResult(
        topic_relevance=topic,
        opinion_collisions=collisions,
        argument_depth_score=depth,
        consensus_quality=consensus,
        turn_taking_pattern="balanced",
        per_speaker=per_speaker or {},
        windows_analyzed=3 if complete else 0,
        windows_failed=0 if complete else 3,
        complete=complete,
    )


def score(
    engine: FusionEngine,
    speaking: list[float],
    topic: float | None = 0.8,
    collisions: int = 6,
    depth: float | None = 0.7,
    consensus: float | None = 0.6,
    duration: float = 300.0,
    video: VideoAnalysisResult | None = None,
):
    audio = build_audio(speaking, duration=duration)
    members = list(audio.speaker_stats)
    return engine.fuse(
        audio,
        video or VideoAnalysisResult(),
        build_semantic(topic, collisions, depth, consensus),
        group_id="g",
        member_ids=members,
    )


@pytest.fixture(scope="module")
def engine() -> FusionEngine:
    return FusionEngine()


# --------------------------------------------------------------------- 均衡度

@pytest.mark.parametrize(
    "shares, expected",
    [
        ([1, 1, 1, 1], 1.0),      # 完全均等
        ([1, 0, 0, 0], 0.0),      # 一人独占
        ([1, 1, 0, 0], 1.0 / 3.0),  # 两人对半（可解析验证）
    ],
)
def test_evenness_matches_closed_form(shares, expected):
    assert hhi_evenness(shares) == pytest.approx(expected, abs=1e-9)


def test_evenness_of_silent_group_is_zero_not_one():
    """关键回归：全员沉默的均衡度必须是 0，旧实现返回 1.0（满分）。"""
    assert hhi_evenness([0, 0, 0, 0]) == 0.0


def test_evenness_is_scale_invariant():
    """均衡度只关心份额分布，与总体发言量无关。"""
    assert hhi_evenness([10, 10, 10]) == pytest.approx(hhi_evenness([1, 1, 1]))
    assert hhi_evenness([90, 30, 30]) == pytest.approx(hhi_evenness([9, 3, 3]))


# --------------------------------------------------------------------- 健康分

def test_silent_group_gets_critical_and_zero(engine):
    """最致命反例：全员一言不发必须落在 critical，而不是 90 分。"""
    report = score(engine, [0, 0, 0, 0], topic=0.5, collisions=0, depth=0.5, consensus=0.5)
    assert report.overall_health_score == 0.0
    assert report.health_level == CollaborationLevel.CRITICAL
    assert report.participation_activity == 0.0
    assert report.evenness == 0.0


def test_silent_group_scores_lower_than_balanced_group(engine):
    """单调性：沉默组必须低于正常均衡组。旧实现这两者是反的。"""
    silent = score(engine, [0, 0, 0, 0], topic=0.5, collisions=0, depth=0.5, consensus=0.5)
    balanced = score(engine, [70, 70, 70, 70])
    assert silent.overall_health_score < balanced.overall_health_score


def test_monopolizing_scores_lower_than_balanced(engine):
    """一人独占必须低于同等参与量的均衡组。"""
    monopolizing = score(engine, [280, 0, 0, 0])
    balanced = score(engine, [70, 70, 70, 70])
    assert monopolizing.overall_health_score < balanced.overall_health_score
    assert monopolizing.evenness == 0.0


def test_off_topic_scores_lower_than_on_topic(engine):
    on_topic = score(engine, [70, 70, 70, 70], topic=0.9)
    off_topic = score(engine, [70, 70, 70, 70], topic=0.1)
    assert off_topic.overall_health_score < on_topic.overall_health_score


def test_low_activity_is_penalized(engine):
    """活跃度不足要被扣分：只说 30 秒的 5 分钟讨论不能得高分。"""
    report = score(engine, [8, 8, 8, 6], duration=300.0)
    assert report.participation_activity < 0.25
    assert report.overall_health_score < 40
    assert report.health_level in {CollaborationLevel.POOR, CollaborationLevel.CRITICAL}


def test_all_five_levels_are_reachable(engine):
    """修复 #3：五个等级都必须可达（旧实现 critical 是死档，下界约 32 分）。

    每个场景都直接断言**具体等级**，因此任何一处公式退化都会立刻暴露。
    """
    scenarios = {
        # (发言秒数, 主题相关度, 观点碰撞次数, 论证深度, 共识质量)
        CollaborationLevel.CRITICAL: ([0, 0, 0, 0], 0.0, 0, 0.0, 0.0),
        CollaborationLevel.POOR: ([30, 30, 30, 30], 0.3, 0, 0.3, 0.3),
        CollaborationLevel.FAIR: ([50, 50, 50, 50], 0.45, 1, 0.45, 0.45),
        CollaborationLevel.GOOD: ([60, 60, 60, 60], 0.6, 3, 0.6, 0.5),
        CollaborationLevel.EXCELLENT: ([75, 75, 75, 75], 1.0, 12, 1.0, 1.0),
    }
    reports = {}
    for level, (speaking, topic, collisions, depth, consensus) in scenarios.items():
        report = score(
            engine,
            speaking,
            topic=topic,
            collisions=collisions,
            depth=depth,
            consensus=consensus,
        )
        reports[level] = report
        assert report.health_level == level, (
            f"期望 {level.value}，实际 {report.health_level.value}"
            f"（分数 {report.overall_health_score:.1f}）"
        )

    ordered = [
        reports[level].overall_health_score
        for level in (
            CollaborationLevel.CRITICAL,
            CollaborationLevel.POOR,
            CollaborationLevel.FAIR,
            CollaborationLevel.GOOD,
            CollaborationLevel.EXCELLENT,
        )
    ]
    assert ordered == sorted(ordered), f"分数应随质量单调上升，实际为 {ordered}"
    assert ordered[0] == 0.0 and ordered[-1] > 85.0


def test_health_score_stays_in_range(engine):
    for speaking in ([0, 0], [1000, 0], [50, 50], [1, 2, 3]):
        report = score(engine, speaking)
        assert 0.0 <= report.overall_health_score <= 100.0


def test_score_range_is_fully_reachable(engine):
    """修复 #3 的量化断言：分数必须能低到 30 以下，也要能高到 85 以上。

    旧实现的可达下界约 32.2 分（贡献度上限 0.3 → 标准差最大 0.15 → 均衡度最小 0.7），
    因此 critical(0-29) 永远不可能出现。
    """
    lowest = score(
        engine, [0, 0, 0, 0], topic=0.0, collisions=0, depth=0.0, consensus=0.0
    )
    highest = score(
        engine,
        [75, 75, 75, 75],
        topic=1.0,
        collisions=20,
        depth=1.0,
        consensus=1.0,
        duration=240.0,
    )
    assert lowest.overall_health_score < 30.0, "健康分必须能进入 critical 区间"
    assert highest.overall_health_score > 85.0, "健康分必须能进入 excellent 区间"


# ------------------------------------------------------------- 缺失模态不编造

def test_missing_semantic_is_excluded_not_defaulted(engine):
    """修复 #22：语义不可用时该维度应被排除，而不是当作 0.5 参与计分。"""
    audio = build_audio([70, 70, 70, 70])
    report = engine.fuse(
        audio,
        VideoAnalysisResult(),
        SemanticAnalysisResult(),  # 全新的空结果 = 语义不可用
        group_id="g",
        member_ids=list(audio.speaker_stats),
    )
    assert report.data_completeness["semantic"] is False
    assert report.topic_relevance is None
    assert report.interaction_depth is None
    # 权重在可用维度（evenness）上重新归一化，因此质量分等于均衡度
    assert report.quality_score == pytest.approx(report.evenness)
    assert any("语义" in w for w in report.warnings)


def test_partial_semantic_coverage_is_flagged(engine):
    audio = build_audio([70, 70, 70, 70])
    semantic = build_semantic()
    semantic.complete = False
    semantic.windows_analyzed = 2
    semantic.windows_failed = 1
    report = engine.fuse(
        audio, VideoAnalysisResult(), semantic, group_id="g", member_ids=["s1", "s2", "s3", "s4"]
    )
    assert report.data_completeness["semantic"] is False
    assert any("覆盖率不完整" in w for w in report.warnings)


def test_optional_semantic_metrics_are_respected(engine):
    """论证深度缺省时，交互深度应由剩余子项聚合，而不是变成 0。"""
    audio = build_audio([70, 70, 70, 70])
    semantic = build_semantic(depth=None, consensus=None, collisions=10)
    report = engine.fuse(
        audio, VideoAnalysisResult(), semantic, group_id="g",
        member_ids=["s1", "s2", "s3", "s4"],
    )
    assert report.interaction_depth is not None
    assert report.interaction_depth > 0


# --------------------------------------------------------------- 个体贡献度

def test_individual_semantic_scores_are_per_person(engine):
    """修复 #5：个体语义分必须逐人，不能把全局分数复制给每个人。"""
    audio = build_audio([60, 60, 60, 60])
    semantic = build_semantic(
        per_speaker={"s1": 1.0, "s2": 0.5, "s3": 0.2, "s4": 0.0}
    )
    report = engine.fuse(
        audio, VideoAnalysisResult(), semantic, group_id="g",
        member_ids=["s1", "s2", "s3", "s4"],
    )
    scores = {c.person_id: c.semantic_score for c in report.individual_contributions}
    assert scores == pytest.approx({"s1": 1.0, "s2": 0.5, "s3": 0.2, "s4": 0.0})
    assert len(set(scores.values())) > 1, "不同成员不应得到同一个语义分"


def test_individual_scores_differ_when_participation_differs(engine):
    audio = build_audio([120, 60, 30, 10])
    semantic = build_semantic(per_speaker={"s1": 0.9, "s2": 0.7, "s3": 0.4, "s4": 0.2})
    report = engine.fuse(
        audio, VideoAnalysisResult(), semantic, group_id="g",
        member_ids=["s1", "s2", "s3", "s4"],
    )
    totals = [c.total_score for c in report.individual_contributions]
    assert len(set(totals)) == len(totals)
    assert totals[0] > totals[-1]


def test_silent_member_gets_zero_speaking_score(engine):
    audio = build_audio([100, 100, 100, 0])
    report = engine.fuse(
        audio, VideoAnalysisResult(), build_semantic(), group_id="g",
        member_ids=["s1", "s2", "s3", "s4"],
    )
    by_id = {c.person_id: c for c in report.individual_contributions}
    assert by_id["s4"].speaking_score == 0.0
    assert by_id["s4"].speaking_share == 0.0


def test_no_nonverbal_attribution_without_mapping(engine):
    """修复 #37：没有视觉↔成员对齐表时，不得把视觉数据张冠李戴到成员头上。"""
    audio = build_audio([70, 70, 70, 70])
    video = VideoAnalysisResult(
        analyzed=True,
        person_attention_stats={
            "person_1": {"attention_to_others": 60.0, "pointing_frequency": 5}
        },
        cohesion_score=0.6,
        fps=25.0,
        frames_processed=100,
        analyzed_duration=300.0,
    )
    report = engine.fuse(
        audio, video, build_semantic(), group_id="g",
        member_ids=["s1", "s2", "s3", "s4"],
    )
    assert all(c.nonverbal_score == 0.0 for c in report.individual_contributions)
    assert all(
        "nonverbal" not in c.diagnosis["details"]["scored_dimensions"]
        for c in report.individual_contributions
    )


def test_nonverbal_attribution_with_mapping(engine):
    """提供对齐表后，非语言维度才参与该成员的计分。"""
    audio = build_audio([70, 70, 70, 70])
    video = VideoAnalysisResult(
        analyzed=True,
        person_attention_stats={
            "person_1": {"attention_to_others": 90.0, "pointing_frequency": 12}
        },
        cohesion_score=0.6,
        fps=25.0,
        frames_processed=100,
        analyzed_duration=300.0,
    )
    report = engine.fuse(
        audio, video, build_semantic(), group_id="g",
        member_ids=["s1", "s2", "s3", "s4"],
        member_mapping={"person_1": "s1"},
    )
    by_id = {c.person_id: c for c in report.individual_contributions}
    assert by_id["s1"].nonverbal_score > 0
    assert by_id["s2"].nonverbal_score == 0.0


def test_video_ids_never_become_members(engine):
    """修复 #37 的核心回归：视觉 ID 不得被当作成员，造出“幽灵成员”。"""
    audio = AudioAnalysisResult(
        speaker_stats={"unknown": {"total_duration": 245.0, "turns": 16, "longest_monologue": 40.0}},
        total_duration=300.0,
        total_speaking_time=245.0,
    )
    video = VideoAnalysisResult(
        analyzed=True,
        person_attention_stats={
            "person_1": {"attention_to_others": 20.0, "pointing_frequency": 3},
            "person_2": {"attention_to_others": 18.0, "pointing_frequency": 2},
        },
    )
    report = engine.fuse(audio, video, build_semantic(), group_id="g")
    assert report.member_ids == ["unknown"]
    assert {c.person_id for c in report.individual_contributions} == {"unknown"}
    assert not any(m.startswith("person_") for m in report.member_ids)
    assert any("unknown" in w for w in report.warnings)


# ------------------------------------------------------------------- 配置校验

def test_problematic_group_produces_actionable_suggestions(engine):
    """有明确问题时必须给出可执行建议（否则诊断就是摆设）。"""
    report = score(engine, [280, 0, 0, 0], topic=0.2, collisions=0, depth=0.2, consensus=0.2)
    assert report.suggestions, "低质量小组必须给出建议"
    assert any("轮流" in s or "沉默" in s for s in report.suggestions)


def test_healthy_group_needs_no_suggestions(engine):
    """反面：表现优秀的组不该硬凑建议出来。"""
    report = score(
        engine, [75, 75, 75, 75], topic=1.0, collisions=12, depth=1.0, consensus=1.0,
        duration=240.0,
    )
    assert report.health_level == CollaborationLevel.EXCELLENT
    assert report.suggestions == []


def test_config_validation_rejects_bad_weights():
    config = FusionEngineConfig()
    config.quality_weights = {"evenness": 0.5, "topic_relevance": 0.2, "interaction_depth": 0.2}
    with pytest.raises(ValueError):
        config.validate()


def test_config_validation_rejects_bad_depth_weights():
    config = FusionEngineConfig()
    config.depth_weights = {"collision": 0.5, "argument": 0.5, "consensus": 0.5}
    with pytest.raises(ValueError):
        config.validate()


def test_default_config_is_valid():
    get_fusion_config().validate()


def test_contribution_weights_reject_unknown_dimension(engine):
    from collaborative_learning_analyzer.data_models import IndividualContribution

    contribution = IndividualContribution(person_id="x")
    with pytest.raises(ValueError):
        contribution.calculate_total({"speaking": 0.5, "typo_dimension": 0.5})
    with pytest.raises(ValueError):
        contribution.calculate_total({"speaking": 0.3, "turn_taking": 0.3})
