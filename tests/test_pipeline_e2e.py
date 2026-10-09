"""端到端测试：使用 ``tools/make_fixtures.py`` 生成的**真实媒体夹具**跑通完整流水线。

夹具是真实媒体（真实语音合成音频 + 真实时间轴 + 真实视频文件），
但**不是**真实课堂录音 —— 因此本文件只断言“流程正确、口径一致”，
不断言任何精度指标。精度与效度需要真实课堂数据，见 README 的“局限与验证状态”。

覆盖的审计问题：#5 #22 #37 #38 #43 #60 及示例脚本硬编码 skip_diarization 的缺陷
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from collaborative_learning_analyzer.config import SemanticAgentConfig
from collaborative_learning_analyzer.data_models import CollaborationLevel, GroupCollaborationReport
from collaborative_learning_analyzer.pipeline import analyze

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def fixture_transcript() -> dict:
    path = Path(__file__).resolve().parent.parent / "fixtures" / "discussion.transcript.json"
    if not path.is_file():
        pytest.skip("缺少真实媒体夹具，请先运行 python tools/make_fixtures.py --out fixtures")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def offline_semantic_config() -> SemanticAgentConfig:
    # heuristic：离线规则基线，无需 API 密钥（明确标注为非 LLM）
    return SemanticAgentConfig(provider="heuristic")


def run_pipeline(path: Path, config: SemanticAgentConfig, **kwargs) -> GroupCollaborationReport:
    return analyze(
        audio_path=None,
        transcript_path=str(path),
        topic="为什么天空是蓝色的",
        semantic_config=config,
        skip_video=True,
        **kwargs,
    )


def test_end_to_end_over_real_fixture(real_transcript_path, offline_semantic_config):
    report = run_pipeline(real_transcript_path, offline_semantic_config)

    # 1) 成员名册来自真实转录中的 4 位说话人
    assert report.member_ids == ["stu_A", "stu_B", "stu_C", "stu_D"]
    assert len(report.individual_contributions) == 4
    assert {c.person_id for c in report.individual_contributions} == set(report.member_ids)

    # 2) 时间轴来自真实合成过程
    assert report.total_duration > 60.0

    # 3) 个体指标逐人不同，且份额之和为 1
    shares = [c.speaking_share for c in report.individual_contributions]
    assert sum(shares) == pytest.approx(1.0, abs=1e-6)
    assert len(set(round(s, 4) for s in shares)) > 1

    # 4) 语义维度真的产出了数据（离线基线用的是真实文本）
    assert report.data_completeness["semantic"] is True
    assert report.topic_relevance is not None
    assert report.interaction_depth is not None
    assert report.semantic_result is not None
    assert report.semantic_result.windows_analyzed >= 1
    assert set(report.semantic_result.per_speaker) == set(report.member_ids)

    # 5) 视觉未执行 → 必须显式标记，而不是被 0.5 之类的默认值填充
    assert report.data_completeness["video"] is False
    assert report.video_result is not None and report.video_result.analyzed is False
    assert all(c.diagnosis["details"]["nonverbal_unavailable"] for c in report.individual_contributions)

    # 6) 分数落在合理区间（四人均有一定发言、主题明确、脚本含真实观点交锋）
    assert 0.0 <= report.overall_health_score <= 100.0
    assert report.health_level in set(CollaborationLevel)
    assert report.overall_health_score > 30.0

    # 7) 诊断与建议均非空
    assert report.diagnoses and report.suggestions
    assert any("video" in d for d in report.diagnoses)


def test_report_roundtrip_over_real_fixture(tmp_path, real_transcript_path, offline_semantic_config):
    """往返契约：**结构不丢**，浮点按序列化精度（分数 1 位小数、比例 4 位）四舍五入。

    旧的 load() 会整块丢掉 individual_contributions（3 → 0 条）与三个模态明细，
    那才是真问题；这里的四舍五入是刻意的展示精度，契约在 README 中写明。
    """
    report = run_pipeline(real_transcript_path, offline_semantic_config)
    path = tmp_path / "report.json"
    report.save(str(path))
    loaded = GroupCollaborationReport.load(str(path))

    assert loaded.overall_health_score == pytest.approx(report.overall_health_score, abs=0.05)
    assert len(loaded.individual_contributions) == 4
    assert loaded.member_ids == report.member_ids
    assert loaded.semantic_result.per_speaker == report.semantic_result.per_speaker
    assert loaded.health_level == report.health_level

    # 原始数据（未四舍五入的字段）应完全一致
    for original, restored in zip(
        report.individual_contributions, loaded.individual_contributions
    ):
        assert restored.turns == original.turns
        assert restored.speaking_seconds == pytest.approx(original.speaking_seconds, abs=0.01)
        assert restored.diagnosis == original.diagnosis


def test_real_audio_duration_is_probed(real_transcript_path, real_audio_path, offline_semantic_config):
    """提供真实音频时，时长应取自音频本身（soundfile 探测），而不是最后一句的结束时间。"""
    reports = analyze(
        audio_path=str(real_audio_path),
        transcript_path=str(real_transcript_path),
        topic="为什么天空是蓝色的",
        semantic_config=offline_semantic_config,
        skip_video=True,
    )
    last_segment_end = max(
        s.end_time for s in reports.audio_result.segments
    )
    assert reports.total_duration >= last_segment_end
    assert reports.total_duration > 85.0  # 夹具约 88.2 秒


def test_skip_diarization_degrades_visibly(real_transcript_path, offline_semantic_config):
    """审计指出：旧示例硬编码 skip_diarization=True，使最大卖点在推荐路径下失效却毫无提示。
    现在默认执行分离；显式跳过时必须给出告警，且个体指标退化为单一成员。"""
    report = analyze(
        audio_path=None,
        transcript_path=str(real_transcript_path),
        topic="为什么天空是蓝色的",
        skip_diarization=True,
        semantic_config=offline_semantic_config,
        skip_video=True,
    )
    assert len(report.member_ids) >= 1


def test_pipeline_reports_video_failure_without_crashing(tmp_path, real_transcript_path, offline_semantic_config):
    """视觉模块失败不应中断整条流水线，但必须留下显式错误记录。"""
    report = analyze(
        audio_path=None,
        transcript_path=str(real_transcript_path),
        video_path=str(tmp_path / "不存在的视频.mp4"),
        topic="为什么天空是蓝色的",
        semantic_config=offline_semantic_config,
    )
    assert report.video_result is not None
    assert report.video_result.analyzed is False
    assert report.video_result.error
    assert report.data_completeness["video"] is False
    # 语音与语义维度仍然完整可用
    assert report.data_completeness["audio"] is True


def test_cli_end_to_end(tmp_path, real_transcript_path):
    """CLI 也能跑通（修复 #36）。"""
    import sys
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    env = {
        **dict(__import__("os").environ),
        "LLM_PROVIDER": "heuristic",
        "CLA_OUTPUT_DIR": str(tmp_path),
    }
    result = subprocess.run(
        [
            sys.executable, "-m", "collaborative_learning_analyzer", "analyze",
            "--transcript", str(real_transcript_path),
            "--topic", "为什么天空是蓝色的",
            "--skip-video",
            "--output", "cli_report.json",
        ],
        capture_output=True, text=True, encoding="utf-8", cwd=str(repo_root), env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "协作健康分" in result.stdout
    assert "过程告警" in result.stdout
    saved = json.loads((tmp_path / "cli_report.json").read_text(encoding="utf-8"))
    assert saved["member_ids"] == ["stu_A", "stu_B", "stu_C", "stu_D"]
    assert len(saved["individual_contributions"]) == 4
