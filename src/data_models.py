"""协作学习分析助手 - 数据模型定义"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class CollaborationLevel(Enum):
    """协作健康等级"""
    EXCELLENT = "excellent"      # 85-100
    GOOD = "good"                # 70-84
    FAIR = "fair"                # 50-69
    POOR = "poor"                # 30-49
    CRITICAL = "critical"       # 0-29


@dataclass
class SpeakingSegment:
    """单次发言片段"""
    speaker_id: str
    start_time: float  # 秒
    end_time: float    # 秒
    text: str = ""
    duration: float = 0.0  # 自动计算
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time


@dataclass
class AudioAnalysisResult:
    """语音智能体输出结果"""
    segments: List[SpeakingSegment] = field(default_factory=list)
    
    # 聚合指标
    speaker_stats: Dict[str, dict] = field(default_factory=dict)
    # {
    #   "speaker_1": {
    #     "total_duration": 45.2,
    #     "turns": 8,
    #     "longest_monologue": 12.3,
    #     "silence_duration": 5.0,
    #     "turn_taking_score": 0.85,  # 0-1，越高越有序
    #   }
    # }
    
    total_duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AttentionTarget:
    """注意力目标（谁或什么被关注）"""
    target_id: str  # person_1, book_2, paper_1 等
    target_type: str  # person / object
    start_time: float
    end_time: float
    duration: float = 0.0
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time


@dataclass
class PointingEvent:
    """指点事件"""
    speaker_id: str
    target_object: str  # book, paper, etc.
    start_time: float
    end_time: float
    duration: float = 0.0
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time


@dataclass
class GazeEvent:
    """视线关注事件"""
    person_id: str
    target_id: str  # 被关注的人的 ID
    start_time: float
    end_time: float
    duration: float = 0.0
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time


@dataclass
class VideoAnalysisResult:
    """视觉智能体输出结果"""
    # 每个人在不同时间段的注意力目标
    attention_map: Dict[str, List[AttentionTarget]] = field(default_factory=dict)
    
    # 指点事件列表
    pointing_events: List[PointingEvent] = field(default_factory=list)
    
    # 视线交互事件列表
    gaze_events: List[GazeEvent] = field(default_factory=list)
    
    # 聚合指标
    cohesion_score: float = 0.0  # 小组凝聚度 0-1
    person_attention_stats: Dict[str, dict] = field(default_factory=dict)
    # {
    #   "person_1": {
    #     "attention_to_others": 0.6,  # 关注他人的时间比例
    #     "attention_to_materials": 0.3,
    #     "pointing_frequency": 5,
    #   }
    # }
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SemanticAnalysisResult:
    """语义智能体输出结果"""
    # 结构化分析结果
    topic_relevance: float = 0.0  # 0-1
    opinion_collisions: int = 0   # 观点碰撞次数（反驳/补充/质疑）
    turn_taking_pattern: str = "balanced"  # balanced / monopolizing / chaotic
    argument_depth_score: float = 0.0  # 0-1
    consensus_quality: float = 0.0  # 共识达成效率
    
    # LLM 原始输出的证据
    evidence: List[str] = field(default_factory=list)
    
    # 分段对话分析（如果启用批处理）
    segment_analysis: List[dict] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IndividualContribution:
    """个体贡献度"""
    person_id: str
    speaking_contribution: float = 0.0  # 发言贡献
    semantic_contribution: float = 0.0  # 语义相关度贡献
    nonverbal_contribution: float = 0.0  # 非语言投入
    pointing_contribution: float = 0.0  # 指点材料贡献
    gaze_contribution: float = 0.0      # 视线跟随贡献
    
    # 综合得分
    total_score: float = 0.0
    
    # 详细诊断
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_total(self, weights: dict) -> float:
        """根据权重计算总分"""
        self.total_score = (
            weights.get("speaking_time", 0.3) * self.speaking_contribution +
            weights.get("semantic_relevance", 0.3) * self.semantic_contribution +
            weights.get("nonverbal_engagement", 0.2) * self.nonverbal_contribution +
            weights.get("pointing_frequency", 0.1) * self.pointing_contribution +
            weights.get("gaze_attention", 0.1) * self.gaze_contribution
        )
        return self.total_score


@dataclass
class GroupCollaborationReport:
    """小组协作报告"""
    group_id: str
    total_duration: float  # 分析的总时长（秒）
    
    # 成员列表
    member_ids: List[str] = field(default_factory=list)
    
    # 各成员贡献度
    individual_contributions: List[IndividualContribution] = field(default_factory=list)
    
    # 小组整体指标
    balance_score: float = 0.0  # 均衡度（贡献分布的标准差）
    collaboration_mode_score: float = 0.0  # 协作模式质量
    overall_health_score: float = 0.0  # 协作健康分 0-100
    
    # 健康等级
    health_level: CollaborationLevel = CollaborationLevel.FAIR
    
    # 诊断建议
    diagnoses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # 详细数据
    audio_result: Optional[AudioAnalysisResult] = None
    video_result: Optional[VideoAnalysisResult] = None
    semantic_result: Optional[SemanticAnalysisResult] = None
    
    # 元数据
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analysis_version: str = "1.0.0"
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "group_id": self.group_id,
            "total_duration": self.total_duration,
            "member_ids": self.member_ids,
            "overall_health_score": self.overall_health_score,
            "health_level": self.health_level.value,
            "balance_score": self.balance_score,
            "collaboration_mode_score": self.collaboration_mode_score,
            "individual_contributions": [
                {
                    "person_id": ic.person_id,
                    "total_score": ic.total_score,
                    "speaking": ic.speaking_contribution,
                    "semantic": ic.semantic_contribution,
                    "nonverbal": ic.nonverbal_contribution,
                    "pointing": ic.pointing_contribution,
                    "gaze": ic.gaze_contribution,
                    "diagnosis": ic.diagnosis,
                }
                for ic in self.individual_contributions
            ],
            "diagnoses": self.diagnoses,
            "suggestions": self.suggestions,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "analysis_version": self.analysis_version,
        }