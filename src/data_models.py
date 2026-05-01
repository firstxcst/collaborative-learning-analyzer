"""协作学习分析助手 - 数据模型定义"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class CollaborationLevel(str, Enum):
    """协作健康等级"""
    EXCELLENT = "excellent"  # 85-100: 协作优秀
    GOOD = "good"           # 70-84: 协作良好
    FAIR = "fair"           # 50-69: 协作一般
    POOR = "poor"           # 30-49: 协作较差
    CRITICAL = "critical"   # 0-29: 协作严重不足


@dataclass
class SpeakingSegment:
    """发言片段：谁在什么时候说了什么"""
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    
    @property
    def duration(self) -> float:
        """发言时长（秒）"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "duration": self.duration
        }


@dataclass
class AudioAnalysisResult:
    """语音分析结果"""
    segments: List[SpeakingSegment] = field(default_factory=list)
    speaker_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "speaker_stats": self.speaker_stats,
            "total_duration": self.total_duration
        }


@dataclass
class AttentionTarget:
    """注意力目标"""
    target_type: str  # "person" / "material" / "unknown"
    target_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class PointingEvent:
    """指点事件"""
    speaker_id: str
    target_type: str  # "material" / "screen" / "person"
    start_time: float
    end_time: float
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class GazeEvent:
    """视线事件"""
    source_id: str
    target_id: str
    start_time: float
    end_time: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class VideoAnalysisResult:
    """视觉分析结果"""
    attention_map: Dict[str, List[AttentionTarget]] = field(default_factory=dict)
    pointing_events: List[PointingEvent] = field(default_factory=list)
    gaze_events: List[GazeEvent] = field(default_factory=list)
    cohesion_score: float = 0.5  # 小组凝聚度 (0-1)
    person_attention_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cohesion_score": self.cohesion_score,
            "person_attention_stats": self.person_attention_stats,
            "num_pointing_events": len(self.pointing_events),
            "num_gaze_events": len(self.gaze_events)
        }


@dataclass
class SemanticAnalysisResult:
    """语义分析结果"""
    topic_relevance: float = 0.5          # 主题相关度 (0-1)
    opinion_collisions: int = 0           # 观点碰撞次数
    turn_taking_pattern: str = "balanced" # balanced/monopolizing/chaotic
    argument_depth_score: float = 0.5     # 论证深度 (0-1)
    consensus_quality: float = 0.5        # 共识质量 (0-1)
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_relevance": self.topic_relevance,
            "opinion_collisions": self.opinion_collisions,
            "turn_taking_pattern": self.turn_taking_pattern,
            "argument_depth_score": self.argument_depth_score,
            "consensus_quality": self.consensus_quality,
            "evidence": self.evidence
        }


@dataclass
class IndividualContribution:
    """个体贡献度"""
    person_id: str
    speaking_contribution: float = 0.0     # 发言贡献 (归一化)
    semantic_contribution: float = 0.0     # 语义贡献
    nonverbal_contribution: float = 0.0    # 非语言参与度
    pointing_contribution: float = 0.0     # 指点贡献
    gaze_contribution: float = 0.0         # 视线跟随贡献
    total_score: float = 0.0               # 总分 (加权平均)
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_total(self, weights: Dict[str, float]) -> float:
        """计算加权总分"""
        self.total_score = (
            weights.get("speaking_time", 0.3) * self.speaking_contribution +
            weights.get("semantic_relevance", 0.3) * self.semantic_contribution +
            weights.get("nonverbal_engagement", 0.2) * self.nonverbal_contribution +
            weights.get("pointing_frequency", 0.1) * self.pointing_contribution +
            weights.get("gaze_attention", 0.1) * self.gaze_contribution
        )
        return self.total_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "speaking_contribution": round(self.speaking_contribution, 3),
            "semantic_contribution": round(self.semantic_contribution, 3),
            "nonverbal_contribution": round(self.nonverbal_contribution, 3),
            "pointing_contribution": round(self.pointing_contribution, 3),
            "gaze_contribution": round(self.gaze_contribution, 3),
            "total_score": round(self.total_score, 3),
            "diagnosis": self.diagnosis
        }


@dataclass
class GroupCollaborationReport:
    """小组协作报告"""
    group_id: str
    total_duration: float
    member_ids: List[str]
    
    # 分析结果
    audio_result: Optional[AudioAnalysisResult] = None
    video_result: Optional[VideoAnalysisResult] = None
    semantic_result: Optional[SemanticAnalysisResult] = None
    
    # 融合指标
    individual_contributions: List[IndividualContribution] = field(default_factory=list)
    balance_score: float = 0.0                    # 均衡度
    collaboration_mode_score: float = 0.0         # 协作模式质量
    overall_health_score: float = 0.0             # 协作健康分 (0-100)
    health_level: CollaborationLevel = CollaborationLevel.FAIR
    
    # 诊断
    diagnoses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # 元数据
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "group_id": self.group_id,
            "total_duration": round(self.total_duration, 2),
            "member_ids": self.member_ids,
            "overall_health_score": round(self.overall_health_score, 1),
            "health_level": self.health_level.value,
            "balance_score": round(self.balance_score, 3),
            "collaboration_mode_score": round(self.collaboration_mode_score, 3),
            "individual_contributions": [c.to_dict() for c in self.individual_contributions],
            "diagnoses": self.diagnoses,
            "suggestions": self.suggestions,
            "created_at": self.created_at,
            "audio_summary": self.audio_result.to_dict() if self.audio_result else None,
            "video_summary": self.video_result.to_dict() if self.video_result else None,
            "semantic_summary": self.semantic_result.to_dict() if self.semantic_result else None
        }
    
    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    def save(self, path: str) -> None:
        """保存到文件"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, path: str) -> "GroupCollaborationReport":
        """从文件加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        report = cls(
            group_id=data["group_id"],
            total_duration=data["total_duration"],
            member_ids=data["member_ids"]
        )
        
        report.overall_health_score = data.get("overall_health_score", 0.0)
        report.health_level = CollaborationLevel(data.get("health_level", "fair"))
        report.balance_score = data.get("balance_score", 0.0)
        report.collaboration_mode_score = data.get("collaboration_mode_score", 0.0)
        report.diagnoses = data.get("diagnoses", [])
        report.suggestions = data.get("suggestions", [])
        report.created_at = data.get("created_at", "")
        
        return report
