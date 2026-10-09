"""端到端流水线：语音 → 视觉 → 语义 → 融合。

修复的审计问题：
* #43/#37 身份对齐接入流水线：说话人标签可用声纹对齐到学生身份；视觉 ID 与成员的
  对应关系必须显式提供，**不再把视觉 ID 直接当作成员**（旧实现会造出“全程未发言”幽灵成员）。
* #22 各阶段的降级与失败提示全部汇总到报告的 ``warnings``，不做静默处理。
* 旧示例硬编码 ``skip_diarization=True``，导致最大卖点（个体参与度）在推荐路径下失效；
  现在默认为**执行**说话人分离，仅在缺少 token 或用户显式要求时才降级，并给出提示。
"""

from __future__ import annotations

from typing import Dict, Optional

try:  # 作为包导入
    from .audio_agent import AudioAgent
    from .config import (
        AudioAgentConfig,
        FusionEngineConfig,
        SemanticAgentConfig,
        VideoAgentConfig,
    )
    from .data_models import (
        AudioAnalysisResult,
        GroupCollaborationReport,
        VideoAnalysisResult,
    )
    from .fusion_engine import FusionEngine
    from .semantic_agent import SemanticAgent
    from .video_agent import VideoAgent
except ImportError:  # 独立运行
    from audio_agent import AudioAgent
    from config import (
        AudioAgentConfig,
        FusionEngineConfig,
        SemanticAgentConfig,
        VideoAgentConfig,
    )
    from data_models import AudioAnalysisResult, GroupCollaborationReport, VideoAnalysisResult
    from fusion_engine import FusionEngine
    from semantic_agent import SemanticAgent
    from video_agent import VideoAgent

__all__ = ["analyze"]


def analyze(
    audio_path: str,
    video_path: Optional[str] = None,
    num_members: int = 4,
    group_id: str = "default",
    topic: Optional[str] = None,
    transcript_path: Optional[str] = None,
    skip_diarization: bool = False,
    skip_video: bool = False,
    skip_semantic: bool = False,
    align_profiles: bool = False,
    member_mapping: Optional[Dict[str, str]] = None,
    audio_config: Optional[AudioAgentConfig] = None,
    video_config: Optional[VideoAgentConfig] = None,
    semantic_config: Optional[SemanticAgentConfig] = None,
    fusion_config: Optional[FusionEngineConfig] = None,
    save_annotated_video: Optional[str] = None,
) -> GroupCollaborationReport:
    """一键分析小组协作音视频。

    Args:
        audio_path: 音频文件路径（必需，除非提供 transcript_path）
        video_path: 视频文件路径（可选）
        num_members: 小组成员数量
        group_id: 小组标识
        topic: 讨论主题，用于判断主题相关度
        transcript_path: 已有转录文件路径（分会场麦克风 / 教师字幕 / 外部 ASR 结果）。
            提供后将跳过本地 ASR 推理。
        skip_diarization: 跳过说话人分离。**会丧失个体区分能力**，结果中会有明确告警。
        skip_video: 跳过视觉分析
        skip_semantic: 跳过语义分析
        align_profiles: 用已注册声纹把说话人标签对齐到学生身份
        member_mapping: 视觉 ID → 成员 ID 的手工对齐表。
            未提供时，非语言维度不会归因到具体成员（避免张冠李戴）。
        save_annotated_video: 标注视频输出路径（需要视觉分析）

    Returns:
        GroupCollaborationReport
    """
    audio_agent = AudioAgent(audio_config)
    fusion_engine = FusionEngine(fusion_config)

    # ---------- 语音 ----------
    audio_result: AudioAnalysisResult = audio_agent.analyze(
        audio_path=audio_path or None,
        num_speakers=num_members,
        skip_diarization=skip_diarization,
        align_profiles=align_profiles,
        transcript_path=transcript_path,
    )

    # ---------- 视觉 ----------
    video_result = VideoAnalysisResult()
    if video_path and not skip_video:
        try:
            video_agent = VideoAgent(video_config)
            video_result = video_agent.analyze(
                video_path,
                num_members=num_members,
                save_annotated=bool(save_annotated_video),
                output_path=save_annotated_video,
            )
        except Exception as exc:  # 视觉是可选的，失败不应中断整条流水线
            video_result = VideoAnalysisResult(
                analyzed=False, error=f"{type(exc).__name__}: {exc}"
            )
            video_result.warnings = [f"视觉分析失败，该模态已排除：{exc}"]

    # ---------- 语义 ----------
    semantic_result = None
    if not skip_semantic:
        semantic_config = semantic_config or SemanticAgentConfig()
        semantic_agent = SemanticAgent(semantic_config)
        semantic_result = semantic_agent.analyze(audio_result.segments, context=topic)

    # ---------- 融合 ----------
    report = fusion_engine.fuse(
        audio_result,
        video_result,
        semantic_result,
        group_id=group_id,
        member_mapping=member_mapping,
    )

    # 合并声纹对齐信息，便于后续把视觉 person_* 与成员关联
    if audio_result.speaker_mapping:
        report.warnings.append(
            "声纹对齐结果：" + ", ".join(
                f"{label}→{sid}" for label, sid in audio_result.speaker_mapping.items()
            )
        )
    if video_result.analyzed and not member_mapping:
        report.warnings.append(
            "未提供视觉 ID → 成员 的对齐表，因此非语言参与度未归因到具体成员"
            "（只保留小组级凝聚度）。"
        )
    return report
