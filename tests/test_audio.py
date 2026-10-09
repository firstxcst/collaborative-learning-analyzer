"""语音模块测试：转录导入、文本归属、统计指标、设备参数。

覆盖的审计问题：#30 #31 #43 及旧实现的重复文本分配缺陷。
"""

from __future__ import annotations

import json

import pytest

from collaborative_learning_analyzer.audio_agent import (
    UNKNOWN_SPEAKER,
    AudioAgent,
    _normalize_device,
)
from collaborative_learning_analyzer.data_models import SpeakingSegment


@pytest.fixture
def agent() -> AudioAgent:
    return AudioAgent()


# ---------------------------------------------------------------- 设备参数

@pytest.mark.parametrize(
    "given, expected",
    [
        ("auto", None),
        ("AUTO", None),
        ("", None),
        ("  ", None),
        (None, None),
        ("cpu", "cpu"),
        ("cuda", "cuda"),
    ],
)
def test_whisper_device_normalisation(given, expected):
    """修复 #30：旧默认值 "auto" 会让 whisper 执行 torch 的 model.to("auto")，
    必然抛 RuntimeError，默认配置开箱即崩。"""
    assert _normalize_device(given) == expected


def test_default_config_device_is_none_not_auto():
    from collaborative_learning_analyzer.config import get_audio_config

    assert get_audio_config().whisper_device is None


# ---------------------------------------------------------------- 转录导入

def test_load_transcript_object_form(agent, tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "source": "会场麦克风",
                "segments": [
                    {"speaker_id": "stu_A", "start": 0.0, "end": 2.5, "text": "第一句"},
                    {"speaker_id": "stu_B", "start": 2.5, "end": 5.0, "text": "第二句"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    segments, warnings = agent.load_transcript(str(path))
    assert [s.speaker_id for s in segments] == ["stu_A", "stu_B"]
    assert segments[0].duration == pytest.approx(2.5)
    assert any("会场麦克风" in w for w in warnings)


def test_load_transcript_array_form_and_sorting(agent, tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            [
                {"speaker_id": "b", "start": 5.0, "end": 6.0, "text": "后"},
                {"speaker_id": "a", "start": 1.0, "end": 2.0, "text": "先"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    segments, _ = agent.load_transcript(str(path))
    assert [s.text for s in segments] == ["先", "后"]


def test_load_transcript_requires_speaker_id(agent, tmp_path):
    """缺少 speaker_id 的条目应判为无效：默默塞成 unknown 会把数据问题藏起来。"""
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {"speaker_id": "a", "start": 0.0, "end": 1.0, "text": "ok"},
                    {"start": 3.0, "end": 4.0, "text": "没有说话人"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    segments, warnings = agent.load_transcript(str(path))
    assert [s.speaker_id for s in segments] == ["a"]
    assert any("跳过" in w for w in warnings)


def test_load_transcript_skips_invalid_entries(agent, tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {"speaker_id": "a", "start": 0.0, "end": 1.0, "text": "ok"},
                    {"speaker_id": "b", "start": "非法", "end": 2.0, "text": "坏"},
                    {"speaker_id": "c", "end": 4.0},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    segments, warnings = agent.load_transcript(str(path))
    assert len(segments) == 1
    assert any("跳过" in w for w in warnings)


def test_load_transcript_swaps_reversed_times(agent, tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps([{"speaker_id": "a", "start": 5.0, "end": 2.0, "text": "倒过来"}]),
        encoding="utf-8",
    )
    segments, _ = agent.load_transcript(str(path))
    assert segments[0].start_time == 2.0 and segments[0].end_time == 5.0


def test_load_transcript_empty_is_flagged(agent, tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"segments": []}), encoding="utf-8")
    segments, warnings = agent.load_transcript(str(path))
    assert segments == []
    assert any("未包含任何有效发言" in w for w in warnings)


def test_analyze_with_transcript_does_not_touch_audio(agent, tmp_path):
    """提供了转录就应当跳过本地 ASR：无需音频、无需 whisper。"""
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {"speaker_id": "s1", "start": 0.0, "end": 10.0, "text": "甲"},
                    {"speaker_id": "s2", "start": 10.0, "end": 18.0, "text": "乙"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = agent.analyze(audio_path=None, transcript_path=str(path))
    assert result.total_duration == pytest.approx(18.0)
    assert result.total_speaking_time == pytest.approx(18.0)
    assert set(result.speaker_stats) == {"s1", "s2"}
    assert result.diarization_applied is True


def test_analyze_requires_one_input(agent):
    with pytest.raises(ValueError):
        agent.analyze(audio_path=None, transcript_path=None)


# ---------------------------------------------------------------- 文本归属

def test_merge_assigns_each_utterance_to_exactly_one_turn(agent):
    """旧实现把与某说话人轮次“有任何重叠”的 ASR 片段全部拼接，同一句话会被重复分配。"""
    dia = [
        SpeakingSegment("SPEAKER_00", 0.0, 10.0, ""),
        SpeakingSegment("SPEAKER_01", 10.0, 20.0, ""),
    ]
    asr = [
        SpeakingSegment(UNKNOWN_SPEAKER, 0.0, 10.0, "第一句"),
        SpeakingSegment(UNKNOWN_SPEAKER, 10.0, 20.0, "第二句"),
        SpeakingSegment(UNKNOWN_SPEAKER, 9.0, 11.0, "跨界句"),
    ]
    merged = agent._merge_segments(dia, asr)
    joined = " ".join(s.text for s in merged)
    assert joined.count("第一句") == 1
    assert joined.count("第二句") == 1
    assert joined.count("跨界句") == 1, "跨越边界的一句话只能归属一个说话人"


def test_merge_keeps_speaker_labels(agent):
    dia = [
        SpeakingSegment("SPEAKER_00", 0.0, 10.0, ""),
        SpeakingSegment("SPEAKER_01", 10.0, 20.0, ""),
    ]
    asr = [SpeakingSegment(UNKNOWN_SPEAKER, 1.0, 9.0, "甲说的话")]
    merged = agent._merge_segments(dia, asr)
    by_speaker = {s.speaker_id: s.text for s in merged}
    assert by_speaker["SPEAKER_00"] == "甲说的话"
    assert by_speaker["SPEAKER_01"] == ""


def test_merge_discards_utterance_outside_all_turns(agent):
    dia = [
        SpeakingSegment("SPEAKER_00", 0.0, 10.0, ""),
        SpeakingSegment("SPEAKER_01", 20.0, 30.0, ""),
    ]
    asr = [SpeakingSegment(UNKNOWN_SPEAKER, 12.0, 18.0, "无人区")]
    merged = agent._merge_segments(dia, asr)
    assert all("无人区" not in s.text for s in merged)


def test_merge_without_diarization_returns_asr(agent):
    asr = [SpeakingSegment(UNKNOWN_SPEAKER, 0.0, 1.0, "x")]
    assert agent._merge_segments([], asr) == asr


# ---------------------------------------------------------------- 统计指标

def test_speaker_stats_fields(agent):
    segments = [
        SpeakingSegment("s1", 0.0, 10.0, "一二三四五"),
        SpeakingSegment("s1", 10.5, 12.0, "六"),
        SpeakingSegment("s2", 13.0, 20.0, "七"),
    ]
    stats = agent._compute_speaker_stats(segments)
    assert stats["s1"]["turns"] == 2
    assert stats["s1"]["total_duration"] == pytest.approx(11.5)
    assert stats["s1"]["longest_monologue"] == pytest.approx(10.0)
    assert stats["s1"]["text_chars"] == 6
    assert stats["s2"]["turns"] == 1


def test_turn_taking_score_rewards_smooth_transitions(agent):
    smooth = [SpeakingSegment("s", 0.0, 5.0, ""), SpeakingSegment("s", 5.0, 10.0, "")]
    choppy = [SpeakingSegment("s", 0.0, 5.0, ""), SpeakingSegment("s", 9.0, 14.0, "")]
    assert agent._compute_turn_taking_score(smooth) > agent._compute_turn_taking_score(choppy)
    assert agent._compute_turn_taking_score(smooth) == pytest.approx(1.0)


def test_turn_taking_score_single_segment(agent):
    assert agent._compute_turn_taking_score([SpeakingSegment("s", 0, 1, "")]) == 1.0


# ---------------------------------------------------------------- 模块卫生

def test_no_global_warning_filter_is_installed():
    """修复 #31：库不得在导入时调用 warnings.filterwarnings("ignore") 污染宿主进程。

    用 AST 检查真正的**调用语句**，而不是字符串匹配（文档里提到这个反模式是合理的）。
    """
    import ast
    import inspect

    from collaborative_learning_analyzer import (
        audio_agent,
        fusion_engine,
        semantic_agent,
        video_agent,
    )

    for module in (audio_agent, fusion_engine, semantic_agent, video_agent):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "filterwarnings"
                and isinstance(func.value, ast.Name)
                and func.value.id == "warnings"
            ):
                raise AssertionError(
                    f"{module.__name__} 调用了 warnings.filterwarnings，会污染宿主进程"
                )
