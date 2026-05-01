"""协作学习分析助手"""

__version__ = "1.0.0"
__author__ = "Collaborative Learning Team"

from .audio_agent import AudioAgent
from .video_agent import VideoAgent
from .semantic_agent import SemanticAgent
from .fusion_engine import FusionEngine
from .data_models import GroupCollaborationReport

__all__ = [
    "AudioAgent",
    "VideoAgent", 
    "SemanticAgent",
    "FusionEngine",
    "GroupCollaborationReport",
]
