"""协作学习分析助手

Usage:
    from collaborative_learning_analyzer import AudioAgent, VideoAgent, SemanticAgent, FusionEngine
    
    # 初始化各智能体
    audio_agent = AudioAgent()
    video_agent = VideoAgent()
    semantic_agent = SemanticAgent()
    fusion_engine = FusionEngine()
    
    # 分析音视频
    audio_result = audio_agent.analyze("discussion.wav", num_speakers=4)
    video_result = video_agent.analyze("discussion.mp4", num_members=4)
    semantic_result = semantic_agent.analyze(audio_result.segments)
    
    # 融合生成报告
    report = fusion_engine.fuse(audio_result, video_result, semantic_result)
    print(report.to_dict())
"""

__version__ = "1.0.0"

from .src.audio_agent import AudioAgent
from .src.video_agent import VideoAgent
from .src.semantic_agent import SemanticAgent
from .src.fusion_engine import FusionEngine
from .src.data_models import (
    GroupCollaborationReport,
    AudioAnalysisResult,
    VideoAnalysisResult,
    SemanticAnalysisResult,
    IndividualContribution,
    CollaborationLevel,
)

__all__ = [
    "AudioAgent",
    "VideoAgent",
    "SemanticAgent",
    "FusionEngine",
    "GroupCollaborationReport",
    "AudioAnalysisResult",
    "VideoAnalysisResult",
    "SemanticAnalysisResult",
    "IndividualContribution",
    "CollaborationLevel",
]
