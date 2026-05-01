"""协作学习分析助手 - 主入口

Usage:
    # 作为包使用
    from collaborative_learning_analyzer import AudioAgent, VideoAgent, SemanticAgent, FusionEngine
    
    # 命令行使用
    python -m collaborative_learning_analyzer --audio discussion.wav --video discussion.mp4
"""

__version__ = "1.0.0"

# 从 src 目录导入核心组件
import sys
from pathlib import Path

# 确保 src 目录在路径中
_src_path = Path(__file__).parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from audio_agent import AudioAgent
from video_agent import VideoAgent
from semantic_agent import SemanticAgent
from fusion_engine import FusionEngine
from data_models import (
    GroupCollaborationReport,
    AudioAnalysisResult,
    VideoAnalysisResult,
    SemanticAnalysisResult,
    IndividualContribution,
    CollaborationLevel,
    SpeakingSegment,
)
from config import get_config, get_audio_config, get_video_config, get_semantic_config

__all__ = [
    # 智能体
    "AudioAgent",
    "VideoAgent", 
    "SemanticAgent",
    "FusionEngine",
    # 数据模型
    "GroupCollaborationReport",
    "AudioAnalysisResult",
    "VideoAnalysisResult",
    "SemanticAnalysisResult",
    "IndividualContribution",
    "CollaborationLevel",
    "SpeakingSegment",
    # 配置
    "get_config",
    "get_audio_config",
    "get_video_config", 
    "get_semantic_config",
]


def analyze(
    audio_path: str,
    video_path: str = None,
    num_members: int = 4,
    group_id: str = "default",
    skip_diarization: bool = True,
    skip_video: bool = False,
    skip_semantic: bool = False,
) -> GroupCollaborationReport:
    """
    便捷函数：一键分析协作学习音视频
    
    Args:
        audio_path: 音频文件路径
        video_path: 视频文件路径（可选）
        num_members: 小组成员数量
        group_id: 小组标识
        skip_diarization: 跳过说话人分离（需要 pyannote token）
        skip_video: 跳过视觉分析
        skip_semantic: 跳过语义分析
        
    Returns:
        GroupCollaborationReport: 协作报告
    """
    # 初始化智能体
    audio_agent = AudioAgent()
    semantic_agent = SemanticAgent()
    fusion_engine = FusionEngine()
    
    # 语音分析
    audio_result = audio_agent.analyze(
        audio_path,
        num_speakers=num_members,
        skip_diarization=skip_diarization
    )
    
    # 视觉分析
    video_result = None
    if video_path and not skip_video:
        video_agent = VideoAgent()
        video_result = video_agent.analyze(video_path, num_members=num_members)
    
    # 语义分析
    semantic_result = None
    if not skip_semantic:
        semantic_result = semantic_agent.analyze(audio_result.segments)
    
    # 融合
    report = fusion_engine.fuse(
        audio_result,
        video_result or VideoAnalysisResult(),
        semantic_result or SemanticAnalysisResult(),
        group_id=group_id
    )
    
    return report


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="协作学习分析助手")
    parser.add_argument("--audio", required=True, help="音频文件路径")
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--members", type=int, default=4, help="小组成员数量")
    parser.add_argument("--group-id", default="default", help="小组标识")
    parser.add_argument("--output", default="report.json", help="输出文件路径")
    parser.add_argument("--skip-diarization", action="store_true", help="跳过说话人分离")
    parser.add_argument("--skip-video", action="store_true", help="跳过视觉分析")
    parser.add_argument("--skip-semantic", action="store_true", help="跳过语义分析")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("协作学习分析助手 v" + __version__)
    print("=" * 60)
    
    report = analyze(
        audio_path=args.audio,
        video_path=args.video,
        num_members=args.members,
        group_id=args.group_id,
        skip_diarization=args.skip_diarization,
        skip_video=args.skip_video,
        skip_semantic=args.skip_semantic,
    )
    
    # 保存报告
    report.save(args.output)
    
    print(f"\n✅ 协作健康分: {report.overall_health_score:.1f}")
    print(f"   健康等级: {report.health_level.value}")
    print(f"\n📄 报告已保存: {args.output}")
