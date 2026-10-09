"""语义模块测试：解析健壮性、分窗覆盖、离线规则基线。

覆盖的审计问题：#21 #22 #23 #24 #25 #26 #27 #28 #29 #5
本文件全部使用**离线路径**（heuristic provider 与直接调用解析器），不需要 API 密钥或网络。
"""

from __future__ import annotations

import pytest

from collaborative_learning_analyzer.data_models import SpeakingSegment
from collaborative_learning_analyzer.semantic_agent import (
    DEFAULT_PROMPT_TEMPLATE,
    SemanticAgent,
    SemanticParseError,
)
from collaborative_learning_analyzer.config import SemanticAgentConfig


@pytest.fixture
def agent() -> SemanticAgent:
    return SemanticAgent(SemanticAgentConfig(provider="heuristic"))


def make_segments(count: int = 40, speaking_seconds: float = 6.0) -> list[SpeakingSegment]:
    speakers = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"]
    return [
        SpeakingSegment(
            speaker_id=speakers[i % 4],
            start_time=i * speaking_seconds,
            end_time=(i + 1) * speaking_seconds - 0.5,
            text="我觉得这个思路有道理，但是我们需要考虑实际操作中的限制条件。",
        )
        for i in range(count)
    ]


# ------------------------------------------------------------------ 解析健壮性

def test_parse_response_handles_fenced_json(agent):
    raw = '```json\n{"topic_relevance": 0.9, "opinion_collisions": 4}\n```'
    data = agent.parse_response(raw)
    assert data["topic_relevance"] == pytest.approx(0.9)
    assert data["opinion_collisions"] == 4


def test_parse_response_extracts_json_from_surrounding_prose(agent):
    """修复 #22 的一半：JSON 外面有解释文字时应当**正确解析**，而不是退化成默认值。"""
    raw = '好的，以下是分析结果：\n{"topic_relevance": 0.77, "opinion_collisions": 2}\n希望有帮助。'
    data = agent.parse_response(raw)
    assert data["topic_relevance"] == pytest.approx(0.77)
    assert data["opinion_collisions"] == 2


def test_parse_response_returns_none_for_null_fields(agent):
    """修复 #21：null / 非法类型必须变成 None，而不是抛未捕获异常或编造 0.5。"""
    raw = (
        '{"topic_relevance": null, "opinion_collisions": 4,'
        ' "argument_depth_score": "高", "consensus_quality": null}'
    )
    data = agent.parse_response(raw)
    assert data["topic_relevance"] is None
    assert data["argument_depth_score"] is None
    assert data["consensus_quality"] is None
    assert data["opinion_collisions"] == 4  # 有效字段照常保留


def test_parse_response_with_only_nulls_counts_as_failure(agent):
    """全部关键字段都不可用时，该窗口应判为失败（以便如实统计覆盖率），
    而不是返回一个“看起来正常”的全 None 结果。"""
    raw = '{"topic_relevance": null, "opinion_collisions": null, "argument_depth_score": null}'
    with pytest.raises(SemanticParseError):
        agent.parse_response(raw)


def test_parse_response_accepts_numeric_strings(agent):
    raw = '{"topic_relevance": "0.8", "opinion_collisions": "3", "argument_depth_score": 0.5}'
    data = agent.parse_response(raw)
    assert data["topic_relevance"] == pytest.approx(0.8)
    assert data["opinion_collisions"] == 3


def test_parse_response_clamps_out_of_range_values(agent):
    raw = '{"topic_relevance": 5.0, "opinion_collisions": -3, "argument_depth_score": -1}'
    data = agent.parse_response(raw)
    assert data["topic_relevance"] == 1.0
    assert data["opinion_collisions"] == 0
    assert data["argument_depth_score"] == 0.0


def test_parse_response_raises_when_no_json_present(agent):
    """这是修复 #22 的核心：解析不出来必须报错，不能静默返回 0.5。"""
    with pytest.raises(SemanticParseError):
        agent.parse_response("抱歉，我无法分析这段对话。")


def test_parse_response_raises_when_all_key_fields_missing(agent):
    with pytest.raises(SemanticParseError):
        agent.parse_response('{"note": "无可奉告"}')


def test_parse_response_collects_per_speaker(agent):
    raw = '{"topic_relevance": 0.5, "per_speaker": {"SPEAKER_00": 0.9, "SPEAKER_01": "0.4"}}'
    data = agent.parse_response(raw)
    assert data["per_speaker"] == {"SPEAKER_00": 0.9, "SPEAKER_01": 0.4}


# ------------------------------------------------------------------ 分窗覆盖

def test_window_planning_covers_entire_discussion(agent):
    """修复 #23/#24：长对话要分窗完整覆盖，而不是截成头尾两段。"""
    agent.config.window_seconds = 120.0
    segments = make_segments(count=120, speaking_seconds=4.0)  # 480 秒 ≈ 8 分钟
    windows = agent.plan_windows(segments)
    assert len(windows) >= 4

    # 覆盖性：每个发言片段都被某个窗口包含
    for seg in segments:
        assert any(lo <= seg.start_time < hi or lo < seg.end_time <= hi for lo, hi in windows)

    # 连续性：窗口首尾相接、无空洞
    assert windows[0][0] == pytest.approx(segments[0].start_time)
    assert windows[-1][1] == pytest.approx(segments[-1].end_time)
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert next_start == pytest.approx(prev_end)


def test_short_discussion_uses_single_window(agent):
    agent.config.window_seconds = 300.0
    segments = make_segments(count=5, speaking_seconds=4.0)
    assert len(agent.plan_windows(segments)) == 1


def test_window_segments_include_context_margin(agent):
    agent.config.window_overlap_seconds = 15.0
    segments = make_segments(count=40, speaking_seconds=10.0)
    window = agent._window_segments(segments, (100.0, 150.0))
    # 包含核心区间 + 前后各 15 秒的上下文，而不是被裁剪掉
    assert min(s.start_time for s in window) < 100.0
    assert max(s.end_time for s in window) > 150.0


def test_no_head_tail_truncation_api_remains(agent):
    """旧实现里把长转录“取头 2000 + 尾 2000”的方法应当已被彻底移除。"""
    assert not hasattr(agent, "_compress_transcript")


# ------------------------------------------------------------------ 离线基线

def test_heuristic_provider_produces_real_signals(agent):
    segments = make_segments(count=20, speaking_seconds=5.0)
    result = agent.analyze(segments, context="光的散射")
    assert result.windows_analyzed >= 1
    assert result.complete is True
    assert result.opinion_collisions > 0          # 文本里含“但是”
    assert result.argument_depth_score is not None
    assert set(result.per_speaker) == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"}
    assert all(0.0 <= v <= 1.0 for v in result.per_speaker.values())
    assert result.topic_relevance is not None


def test_heuristic_without_topic_marks_relevance_unavailable(agent):
    """没有主题就无法判断相关度 —— 此时必须返回 None，不能编造 0.5。"""
    result = agent.analyze(make_segments(count=10), context=None)
    assert result.topic_relevance is None
    assert result.complete is True  # 其余指标仍然可用


def test_empty_transcript_is_skipped_not_defaulted(agent):
    result = agent.analyze([])
    assert result.windows_analyzed == 0
    assert result.complete is False
    assert result.topic_relevance is None
    assert result.argument_depth_score is None
    assert result.warnings


def test_blank_text_is_skipped(agent):
    segments = [SpeakingSegment("s1", 0.0, 1.0, "   ")]
    result = agent.analyze(segments)
    assert result.windows_analyzed == 0
    assert result.warnings


def test_window_failure_is_recorded_not_hidden(agent, monkeypatch):
    """修复 #22 的行为级验证：窗口分析失败必须计入 windows_failed 并给出告警。"""

    def boom(self, segments, core, context=None):
        raise SemanticParseError("模拟解析失败")

    monkeypatch.setattr(SemanticAgent, "analyze_window", boom)
    result = agent.analyze(make_segments(count=100, speaking_seconds=5.0), context="主题")
    assert result.windows_analyzed == 0
    assert result.windows_failed >= 1
    assert any("失败" in w for w in result.warnings)
    assert result.topic_relevance is None


def test_partial_failure_aggregates_only_successful_windows(agent, monkeypatch):
    calls = {"n": 0}
    original = SemanticAgent.analyze_window

    def flaky(self, segments, core, context=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise SemanticParseError("第二个窗口失败")
        return original(self, segments, core, context)

    monkeypatch.setattr(SemanticAgent, "analyze_window", flaky)
    result = agent.analyze(make_segments(count=100, speaking_seconds=6.0), context="主题")
    assert result.windows_analyzed >= 1
    assert result.windows_failed == 1
    assert result.complete is False
    assert result.topic_relevance is not None


# ------------------------------------------------------------------ 提供者配置

def test_dashscope_model_resolution_does_not_crash():
    """修复 #26：旧实现在 dashscope 分支读取不存在的 qwen_model 字段，必然 AttributeError。"""
    from collaborative_learning_analyzer.config import APIConfig

    agent = SemanticAgent(
        SemanticAgentConfig(provider="dashscope"), api_config=APIConfig()
    )
    assert agent._resolve_model() == "qwen2.5-72b-instruct"


def test_explicit_model_overrides_default():
    from collaborative_learning_analyzer.config import APIConfig

    agent = SemanticAgent(
        SemanticAgentConfig(provider="openai", model="gpt-4o-mini"), api_config=APIConfig()
    )
    assert agent._resolve_model() == "gpt-4o-mini"


def test_prompt_template_keeps_placeholders():
    for token in ("__TOPIC__", "__START__", "__END__", "__TRANSCRIPT__"):
        assert token in DEFAULT_PROMPT_TEMPLATE
