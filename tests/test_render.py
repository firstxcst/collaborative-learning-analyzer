"""HTML 报告渲染测试。

重点验证两件事：
1. **完全自包含** —— 不引用任何外部资源，否则离线/内网环境会渲染成白页；
2. **不可用 ≠ 0** —— ``None`` 指标必须显示为「不可用」，不能画成 0 或省略。
"""

from __future__ import annotations

import re

import pytest

from collaborative_learning_analyzer.data_models import (
    AudioAnalysisResult,
    CollaborationLevel,
    GroupCollaborationReport,
    IndividualContribution,
    SemanticAnalysisResult,
    VideoAnalysisResult,
)
from collaborative_learning_analyzer.render import render_html, render_report


def make_report(**overrides) -> GroupCollaborationReport:
    audio = AudioAnalysisResult(
        speaker_stats={"s1": {"total_duration": 30.0, "turns": 3, "longest_monologue": 12.0}},
        total_duration=60.0,
        total_speaking_time=30.0,
        diarization_applied=True,
    )
    contribution = IndividualContribution(
        person_id="s1",
        speaking_seconds=30.0,
        speaking_share=1.0,
        turns=3,
        speaking_score=1.0,
        turn_taking_score=0.9,
        semantic_score=0.8,
        nonverbal_score=0.0,
        total_score=0.85,
        diagnosis={
            "strengths": ["参与讨论（3 次发言）"],
            "weaknesses": ["非语言参与度低"],
            "details": {"scored_dimensions": ["speaking", "turn_taking", "semantic"]},
        },
    )
    report = GroupCollaborationReport(
        group_id="g1",
        total_duration=60.0,
        member_ids=["s1"],
        audio_result=audio,
        video_result=VideoAnalysisResult(),
        semantic_result=SemanticAnalysisResult(
            topic_relevance=0.8, windows_analyzed=1, complete=True
        ),
        individual_contributions=[contribution],
        participation_activity=0.9,
        evenness=1.0,
        topic_relevance=0.8,
        interaction_depth=0.7,
        quality_score=0.85,
        overall_health_score=85.0,
        health_level=CollaborationLevel.EXCELLENT,
        diagnoses=["诊断一"],
        suggestions=["建议一"],
        data_completeness={"audio": True, "video": False, "semantic": True},
        warnings=["视觉分析未产出数据"],
    )
    for key, value in overrides.items():
        setattr(report, key, value)
    return report


def test_html_is_fully_self_contained():
    """不得出现任何外部资源引用（离线/内网环境必须能正常渲染）。"""
    html = render_html(make_report())
    assert "<!DOCTYPE html>" in html

    banned = [
        r'<script[^>]*src=',
        r'<link[^>]*href="https?://',
        r'@import\s+url\(',
        r'url\(\s*["\']?https?://',
    ]
    for pattern in banned:
        assert not re.search(pattern, html), f"HTML 引用了外部资源：{pattern}"


def test_unavailable_metric_is_marked_not_zero():
    """``None`` 必须显示为「不可用」，不能画成 0%。"""
    report = make_report(topic_relevance=None, interaction_depth=None)
    html = render_html(report)
    assert "不可用" in html
    # 不应把 None 渲染成 0%
    assert "主题相关度</span><span class=\"value\">0%" not in html


def test_missing_modality_is_declared_prominently():
    report = make_report()
    html = render_html(report)
    assert "未参与计分" in html
    assert "video" in html
    assert "chip off" in html  # 未使用的模态要有明确标识


def test_nonverbal_unavailable_is_named_in_row():
    report = make_report()
    html = render_html(report)
    assert "非语言维度未参与计分" in html


def test_score_and_level_are_rendered():
    report = make_report()
    html = render_html(report)
    assert "85.0" in html
    assert "协作优秀" in html


def test_contributions_table_has_real_numbers():
    report = make_report()
    html = render_html(report)
    assert "s1" in html
    assert "30.0s" in html
    assert "0.850" in html


def test_html_escapes_group_id():
    report = make_report(group_id="<script>alert(1)</script>")
    html = render_html(report)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_report_from_json_file(tmp_path):
    report = make_report()
    json_path = tmp_path / "report.json"
    report.save(str(json_path))

    out = render_report(str(json_path), tmp_path / "nested" / "report.html")
    assert out.is_file()
    assert "协作优秀" in out.read_text(encoding="utf-8")


def test_render_report_accepts_object(tmp_path):
    out = render_report(make_report(), tmp_path / "r.html")
    assert out.is_file()


def test_limitations_footer_is_present():
    """报告页脚必须带使用边界声明，避免被当成评价依据。"""
    html = render_html(make_report())
    assert "不得作为评价学生的依据" in html
