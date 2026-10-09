"""数据模型测试：往返无损、可选指标语义、权重校验。

覆盖的审计问题：#22 #38 #5
"""

from __future__ import annotations

import json

import pytest

from collaborative_learning_analyzer.data_models import (
    AudioAnalysisResult,
    CollaborationLevel,
    GroupCollaborationReport,
    IndividualContribution,
    PointingEvent,
    SemanticAnalysisResult,
    SpeakingSegment,
    VideoAnalysisResult,
)


def make_report() -> GroupCollaborationReport:
    audio = AudioAnalysisResult(
        segments=[SpeakingSegment("s1", 0.0, 5.0, "文本")],
        speaker_stats={"s1": {"total_duration": 5.0, "turns": 1, "longest_monologue": 5.0}},
        total_duration=10.0,
        total_speaking_time=5.0,
        diarization_applied=True,
        warnings=["示例告警"],
    )
    video = VideoAnalysisResult(
        analyzed=True,
        pointing_events=[PointingEvent("s1", "material", 1.0, 2.0, 0.8)],
        cohesion_score=0.66,
        fps=25.0,
        frames_processed=10,
        analyzed_duration=10.0,
    )
    semantic = SemanticAnalysisResult(
        topic_relevance=0.8,
        opinion_collisions=3,
        argument_depth_score=0.7,
        consensus_quality=0.6,
        per_speaker={"s1": 0.9},
        windows_analyzed=1,
        complete=True,
    )
    contributions = [
        IndividualContribution(
            person_id="s1",
            speaking_seconds=5.0,
            speaking_share=1.0,
            turns=1,
            speaking_score=1.0,
            turn_taking_score=0.8,
            semantic_score=0.9,
            nonverbal_score=0.5,
            total_score=0.85,
        )
    ]
    return GroupCollaborationReport(
        group_id="g1",
        total_duration=10.0,
        member_ids=["s1"],
        audio_result=audio,
        video_result=video,
        semantic_result=semantic,
        individual_contributions=contributions,
        participation_activity=0.9,
        evenness=1.0,
        topic_relevance=0.8,
        interaction_depth=0.7,
        turn_taking_pattern="balanced",
        consensus_quality=0.6,
        quality_score=0.8,
        overall_health_score=80.0,
        health_level=CollaborationLevel.GOOD,
        diagnoses=["d"],
        suggestions=["s"],
        data_completeness={"audio": True, "video": True, "semantic": True},
        warnings=["w"],
    )


def test_save_load_roundtrip_is_lossless(tmp_path):
    """修复 #38：旧实现的 load() 会丢掉 individual_contributions 与三个子结果。"""
    report = make_report()
    path = tmp_path / "report.json"
    report.save(str(path))
    loaded = GroupCollaborationReport.load(str(path))

    assert loaded.group_id == report.group_id
    assert loaded.member_ids == report.member_ids
    assert len(loaded.individual_contributions) == 1
    assert loaded.individual_contributions[0].person_id == "s1"
    assert loaded.individual_contributions[0].total_score == pytest.approx(0.85)
    assert loaded.individual_contributions[0].diagnosis == {}
    assert loaded.audio_result is not None
    assert loaded.video_result is not None
    assert loaded.semantic_result is not None
    assert loaded.audio_result.warnings == ["示例告警"]
    assert loaded.video_result.pointing_events[0].target_type == "material"
    assert loaded.semantic_result.per_speaker == {"s1": 0.9}
    assert loaded.data_completeness == report.data_completeness
    assert loaded.health_level == CollaborationLevel.GOOD
    assert loaded.turn_taking_pattern == "balanced"


def test_roundtrip_preserves_none_optional_metrics(tmp_path):
    """不可用指标必须在往返后仍然是 None，不能被写成 0 或 0.5。"""
    report = make_report()
    report.topic_relevance = None
    report.interaction_depth = None
    report.consensus_quality = None
    report.semantic_result = SemanticAnalysisResult()
    path = tmp_path / "r.json"
    report.save(str(path))

    loaded = GroupCollaborationReport.load(str(path))
    assert loaded.topic_relevance is None
    assert loaded.interaction_depth is None
    assert loaded.consensus_quality is None
    assert loaded.semantic_result.topic_relevance is None
    assert loaded.semantic_result.argument_depth_score is None


def test_to_dict_keeps_optional_none_not_zero():
    report = make_report()
    report.topic_relevance = None
    data = report.to_dict()
    assert data["topic_relevance"] is None
    assert json.dumps(data, ensure_ascii=False)  # 可序列化


def test_speaking_segment_duration_never_negative():
    assert SpeakingSegment("a", 5.0, 2.0, "").duration == 0.0


def test_level_from_score_boundaries():
    assert CollaborationLevel.from_score(85.0) == CollaborationLevel.EXCELLENT
    assert CollaborationLevel.from_score(84.99) == CollaborationLevel.GOOD
    assert CollaborationLevel.from_score(70.0) == CollaborationLevel.GOOD
    assert CollaborationLevel.from_score(69.99) == CollaborationLevel.FAIR
    assert CollaborationLevel.from_score(50.0) == CollaborationLevel.FAIR
    assert CollaborationLevel.from_score(49.99) == CollaborationLevel.POOR
    assert CollaborationLevel.from_score(30.0) == CollaborationLevel.POOR
    assert CollaborationLevel.from_score(29.99) == CollaborationLevel.CRITICAL
    assert CollaborationLevel.from_score(0.0) == CollaborationLevel.CRITICAL


def test_calculate_total_rejects_mismatched_weight_sum():
    contribution = IndividualContribution(person_id="x")
    with pytest.raises(ValueError, match="之和必须为 1.0"):
        contribution.calculate_total({"speaking": 0.5, "turn_taking": 0.4})


def test_calculate_total_accepts_subset_after_renormalisation():
    contribution = IndividualContribution(person_id="x", speaking_score=1.0, semantic_score=0.5)
    total = contribution.calculate_total({"speaking": 0.6, "semantic": 0.4})
    assert total == pytest.approx(0.6 * 1.0 + 0.4 * 0.5)


def test_video_result_defaults_are_not_fabricated():
    """修复 #12：未执行视觉分析时 cohesion 必须是 None，而不是硬编码 0.5。"""
    result = VideoAnalysisResult()
    assert result.cohesion_score is None
    assert result.analyzed is False
    assert result.person_attention_stats == {}
