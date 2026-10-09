"""协作学习分析助手。

修复的审计问题 #4/#33：旧 ``__init__`` 在包导入时立即 ``import audio_agent / video_agent``，
而 ``video_agent`` 顶层又 ``import cv2``，导致**只想用语音也必须装 opencv**，
缺 opencv 时连 ``import`` 都失败。

现改为 PEP 562 惰性属性：真正访问某个类时才导入对应模块。
因此 ``from collaborative_learning_analyzer import FusionEngine`` 不再需要视觉依赖。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "1.1.0"
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
    "SpeakingSegment",
    "analyze",
    "__version__",
]

#: 逻辑名 -> (模块名, 模块内属性名)
_LAZY: dict[str, tuple[str, str]] = {
    "AudioAgent": ("audio_agent", "AudioAgent"),
    "VideoAgent": ("video_agent", "VideoAgent"),
    "SemanticAgent": ("semantic_agent", "SemanticAgent"),
    "FusionEngine": ("fusion_engine", "FusionEngine"),
    "GroupCollaborationReport": ("data_models", "GroupCollaborationReport"),
    "AudioAnalysisResult": ("data_models", "AudioAnalysisResult"),
    "VideoAnalysisResult": ("data_models", "VideoAnalysisResult"),
    "SemanticAnalysisResult": ("data_models", "SemanticAnalysisResult"),
    "IndividualContribution": ("data_models", "IndividualContribution"),
    "CollaborationLevel": ("data_models", "CollaborationLevel"),
    "SpeakingSegment": ("data_models", "SpeakingSegment"),
    "analyze": ("pipeline", "analyze"),
}

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查器
    from .audio_agent import AudioAgent
    from .data_models import (
        AudioAnalysisResult,
        CollaborationLevel,
        GroupCollaborationReport,
        IndividualContribution,
        SemanticAnalysisResult,
        SpeakingSegment,
        VideoAnalysisResult,
    )
    from .fusion_engine import FusionEngine
    from .pipeline import analyze
    from .semantic_agent import SemanticAgent
    from .video_agent import VideoAgent


def __getattr__(name: str) -> Any:
    """惰性加载公开符号（PEP 562）。"""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attribute)
    globals()[name] = value  # 缓存，后续访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
