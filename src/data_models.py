"""协作学习分析助手 - 数据模型定义

设计约束（针对审计问题 #22 #38 #5 修复）：
1. **降级必须可见**：任何模态缺失、解析失败、回退路径都要在结果对象上留显式标记，
   不允许把失败悄悄变成“中等分”。
2. **往返必须无损**：``to_dict`` / ``from_dict`` / ``save`` / ``load`` 必须对称，
   不得丢失个体贡献度与各模态明细。
3. **个体指标必须逐人**：任何“全组同一个数”填进个人维度的做法都是错误的。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class CollaborationLevel(str, Enum):
    """协作健康等级"""

    EXCELLENT = "excellent"  # 85-100
    GOOD = "good"            # 70-84
    FAIR = "fair"            # 50-69
    POOR = "poor"            # 30-49
    CRITICAL = "critical"    # 0-29

    @classmethod
    def from_score(
        cls, score: float, thresholds: Optional[Dict[str, float]] = None
    ) -> "CollaborationLevel":
        """按切分点判定等级。"""
        t = thresholds or {
            "critical": 0.0,
            "poor": 30.0,
            "fair": 50.0,
            "good": 70.0,
            "excellent": 85.0,
        }
        if score >= t["excellent"]:
            return cls.EXCELLENT
        if score >= t["good"]:
            return cls.GOOD
        if score >= t["fair"]:
            return cls.FAIR
        if score >= t["poor"]:
            return cls.POOR
        return cls.CRITICAL


class _Serializable:
    """提供对称的 dict 往返能力。"""

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - 由子类覆盖
        raise NotImplementedError


# ---------------------------------------------------------------- 语音

@dataclass
class SpeakingSegment:
    """发言片段：谁在什么时候说了什么"""

    speaker_id: str
    start_time: float
    end_time: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakingSegment":
        return cls(
            speaker_id=data["speaker_id"],
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            text=data.get("text", ""),
        )


@dataclass
class AudioAnalysisResult(_Serializable):
    """语音分析结果"""

    segments: List[SpeakingSegment] = field(default_factory=list)
    #: 每个说话人的统计：total_duration / turns / longest_monologue / turn_taking_score
    speaker_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_duration: float = 0.0

    # --- 完整性与降级标记（修复 #22：失败必须可见）---
    #: 是否成功执行了说话人分离
    diarization_applied: bool = False
    #: 说话人分离失败时的原因（None 表示未失败）
    diarization_error: Optional[str] = None
    #: 语音总时长（所有发言片段时长之和）
    total_speaking_time: float = 0.0
    #: 声纹对齐结果：说话人标签 → 学生 ID
    speaker_mapping: Dict[str, str] = field(default_factory=dict)
    #: 本模态产生的降级/失败提示
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "speaker_stats": self.speaker_stats,
            "total_duration": self.total_duration,
            "diarization_applied": self.diarization_applied,
            "diarization_error": self.diarization_error,
            "total_speaking_time": self.total_speaking_time,
            "speaker_mapping": self.speaker_mapping,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioAnalysisResult":
        return cls(
            segments=[SpeakingSegment.from_dict(s) for s in data.get("segments", [])],
            speaker_stats=data.get("speaker_stats", {}),
            total_duration=float(data.get("total_duration", 0.0)),
            diarization_applied=bool(data.get("diarization_applied", False)),
            diarization_error=data.get("diarization_error"),
            total_speaking_time=float(data.get("total_speaking_time", 0.0)),
            speaker_mapping=dict(data.get("speaker_mapping", {})),
            warnings=list(data.get("warnings", [])),
        )


# ---------------------------------------------------------------- 视觉

@dataclass
class AttentionTarget:
    """注意力目标"""

    target_type: str  # "person" / "material"
    target_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttentionTarget":
        allowed = {"target_type", "target_id", "start_time", "end_time"}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class PointingEvent:
    """指点事件"""

    speaker_id: str
    target_type: str  # "material" / "person"
    start_time: float
    end_time: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PointingEvent":
        allowed = {"speaker_id", "target_type", "start_time", "end_time", "confidence"}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class GazeEvent:
    """视线事件"""

    source_id: str
    target_id: str
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GazeEvent":
        allowed = {"source_id", "target_id", "start_time", "end_time"}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class VideoAnalysisResult(_Serializable):
    """视觉分析结果"""

    attention_map: Dict[str, List[AttentionTarget]] = field(default_factory=dict)
    pointing_events: List[PointingEvent] = field(default_factory=list)
    gaze_events: List[GazeEvent] = field(default_factory=list)
    #: 小组凝聚度。未执行视觉分析时为 None（修复 #12：不再用 0.5 冒充）
    cohesion_score: Optional[float] = None
    person_attention_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --- 完整性与降级标记 ---
    #: 是否真的执行了视觉分析（未执行时不应编造默认值）
    analyzed: bool = False
    #: 从视频容器读到的真实帧率
    fps: Optional[float] = None
    #: fps 是否来自兜底值（修复 #16）
    fps_is_fallback: bool = False
    frames_processed: int = 0
    analyzed_duration: float = 0.0
    error: Optional[str] = None
    #: 本模态产生的降级/失败提示（例如帧率兜底、模型不可用）
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attention_map": {
                k: [t.to_dict() for t in v] for k, v in self.attention_map.items()
            },
            "pointing_events": [e.to_dict() for e in self.pointing_events],
            "gaze_events": [e.to_dict() for e in self.gaze_events],
            "cohesion_score": self.cohesion_score,
            "person_attention_stats": self.person_attention_stats,
            "analyzed": self.analyzed,
            "fps": self.fps,
            "fps_is_fallback": self.fps_is_fallback,
            "frames_processed": self.frames_processed,
            "analyzed_duration": self.analyzed_duration,
            "error": self.error,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoAnalysisResult":
        return cls(
            attention_map={
                k: [AttentionTarget.from_dict(t) for t in v]
                for k, v in (data.get("attention_map") or {}).items()
            },
            pointing_events=[
                PointingEvent.from_dict(e) for e in data.get("pointing_events", [])
            ],
            gaze_events=[GazeEvent.from_dict(e) for e in data.get("gaze_events", [])],
            cohesion_score=data.get("cohesion_score"),
            person_attention_stats=data.get("person_attention_stats", {}),
            analyzed=bool(data.get("analyzed", False)),
            fps=data.get("fps"),
            fps_is_fallback=bool(data.get("fps_is_fallback", False)),
            frames_processed=int(data.get("frames_processed", 0)),
            analyzed_duration=float(data.get("analyzed_duration", 0.0)),
            error=data.get("error"),
            warnings=list(data.get("warnings", [])),
        )


# ---------------------------------------------------------------- 语义

@dataclass
class SemanticAnalysisResult(_Serializable):
    """语义分析结果

    ``topic_relevance`` / ``argument_depth_score`` / ``consensus_quality`` 允许为 ``None``，
    表示“该指标不可用”。这修复了旧实现把解析失败静默变成 0.5 中性分的问题：
    不可用的指标必须能被区分出来，并被排除在计分之外，而不是冒充“中等表现”。
    """

    topic_relevance: Optional[float] = None
    opinion_collisions: int = 0
    turn_taking_pattern: str = "unknown"
    argument_depth_score: Optional[float] = None
    consensus_quality: Optional[float] = None
    evidence: List[str] = field(default_factory=list)

    #: 逐说话人语义参与分（0-1）。修复 #5：不能再把全局分数填给每个人。
    per_speaker: Dict[str, float] = field(default_factory=dict)

    # --- 完整性与降级标记（修复 #22：解析失败不再是静默的 0.5）---
    windows_analyzed: int = 0
    windows_failed: int = 0
    complete: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_relevance": self.topic_relevance,
            "opinion_collisions": self.opinion_collisions,
            "turn_taking_pattern": self.turn_taking_pattern,
            "argument_depth_score": self.argument_depth_score,
            "consensus_quality": self.consensus_quality,
            "evidence": self.evidence,
            "per_speaker": self.per_speaker,
            "windows_analyzed": self.windows_analyzed,
            "windows_failed": self.windows_failed,
            "complete": self.complete,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticAnalysisResult":
        def opt_float(key: str) -> Optional[float]:
            value = data.get(key)
            return None if value is None else float(value)

        return cls(
            topic_relevance=opt_float("topic_relevance"),
            opinion_collisions=int(data.get("opinion_collisions", 0)),
            turn_taking_pattern=data.get("turn_taking_pattern", "unknown"),
            argument_depth_score=opt_float("argument_depth_score"),
            consensus_quality=opt_float("consensus_quality"),
            evidence=list(data.get("evidence", [])),
            per_speaker=dict(data.get("per_speaker", {})),
            windows_analyzed=int(data.get("windows_analyzed", 0)),
            windows_failed=int(data.get("windows_failed", 0)),
            complete=bool(data.get("complete", False)),
            warnings=list(data.get("warnings", [])),
        )


# ---------------------------------------------------------------- 个体贡献

@dataclass
class IndividualContribution(_Serializable):
    """个体贡献度（全部维度均为**该成员自己的**测量值）"""

    person_id: str

    # 原始测量
    speaking_seconds: float = 0.0
    speaking_share: float = 0.0   # 占组内总发言时长比例
    turns: int = 0
    longest_monologue: float = 0.0

    # 归一化维度分（0-1）
    speaking_score: float = 0.0
    turn_taking_score: float = 0.0
    semantic_score: float = 0.0
    nonverbal_score: float = 0.0

    total_score: float = 0.0
    diagnosis: Dict[str, Any] = field(default_factory=dict)

    def calculate_total(self, weights: Dict[str, float]) -> float:
        """加权汇总为个体贡献度。

        入参 ``weights`` 是**已在可用维度上重新归一化**的权重：
        缺失的模态会被排除在外（而不是补 0 分），因此只允许出现已知维度的子集，
        且必须满足权重和为 1.0 —— 这样权重写漏或写错都会被立刻发现。
        """
        known = {"speaking", "turn_taking", "semantic", "nonverbal"}
        unknown = set(weights) - known
        if unknown:
            raise ValueError(f"contribution_weights 含未知维度: {sorted(unknown)}")
        if not weights:
            raise ValueError("contribution_weights 不能为空")
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"contribution_weights 之和必须为 1.0，当前为 {total_weight}")

        self.total_score = sum(
            weight * getattr(self, f"{dim}_score") for dim, weight in weights.items()
        )
        return self.total_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "speaking_seconds": round(self.speaking_seconds, 2),
            "speaking_share": round(self.speaking_share, 4),
            "turns": self.turns,
            "longest_monologue": round(self.longest_monologue, 2),
            "speaking_score": round(self.speaking_score, 4),
            "turn_taking_score": round(self.turn_taking_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "nonverbal_score": round(self.nonverbal_score, 4),
            "total_score": round(self.total_score, 4),
            "diagnosis": self.diagnosis,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndividualContribution":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------- 小组报告

@dataclass
class GroupCollaborationReport(_Serializable):
    """小组协作报告"""

    group_id: str
    total_duration: float
    member_ids: List[str]

    audio_result: Optional[AudioAnalysisResult] = None
    video_result: Optional[VideoAnalysisResult] = None
    semantic_result: Optional[SemanticAnalysisResult] = None

    individual_contributions: List[IndividualContribution] = field(default_factory=list)

    # 小组级指标（前五项 0-1，最后一项 0-100）
    #: 活跃度：组内实际发言时长相对于目标发言量的完成度
    participation_activity: float = 0.0
    #: 均衡度：发言份额分布的均匀程度（份额完全均等为 1，一人独占为 0）
    evenness: float = 0.0
    #: 主题相关度（None 表示语义分析未覆盖，该维度未参与计分）
    topic_relevance: Optional[float] = None
    #: 交互深度：观点碰撞 + 论证深度 + 共识质量（None 表示不可用）
    interaction_depth: Optional[float] = None
    #: 轮流发言模式：balanced / monopolizing / chaotic / unknown
    turn_taking_pattern: str = "unknown"
    #: 共识质量（None 表示不可用）
    consensus_quality: Optional[float] = None
    #: 质量分（加权）
    quality_score: float = 0.0
    #: 协作健康分（0-100）
    overall_health_score: float = 0.0
    health_level: CollaborationLevel = CollaborationLevel.CRITICAL

    diagnoses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    #: 各模态是否真的产出了数据（修复 #22：让缺失可见）
    data_completeness: Dict[str, bool] = field(default_factory=dict)
    #: 全流程的降级/失败提示
    warnings: List[str] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "total_duration": round(self.total_duration, 2),
            "member_ids": self.member_ids,
            "participation_activity": round(self.participation_activity, 4),
            "evenness": round(self.evenness, 4),
            "topic_relevance": None if self.topic_relevance is None else round(self.topic_relevance, 4),
            "interaction_depth": None if self.interaction_depth is None else round(self.interaction_depth, 4),
            "turn_taking_pattern": self.turn_taking_pattern,
            "consensus_quality": None if self.consensus_quality is None else round(self.consensus_quality, 4),
            "quality_score": round(self.quality_score, 4),
            "overall_health_score": round(self.overall_health_score, 1),
            "health_level": self.health_level.value,
            "individual_contributions": [c.to_dict() for c in self.individual_contributions],
            "diagnoses": self.diagnoses,
            "suggestions": self.suggestions,
            "data_completeness": self.data_completeness,
            "warnings": self.warnings,
            "created_at": self.created_at,
            "audio_summary": self.audio_result.to_dict() if self.audio_result else None,
            "video_summary": self.video_result.to_dict() if self.video_result else None,
            "semantic_summary": self.semantic_result.to_dict() if self.semantic_result else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupCollaborationReport":
        """从字典完整还原，**不丢失**个体贡献度与各模态明细（修复 #38）。"""
        audio = data.get("audio_summary")
        video = data.get("video_summary")
        semantic = data.get("semantic_summary")

        def opt_float(key: str) -> Optional[float]:
            value = data.get(key)
            return None if value is None else float(value)

        return cls(
            group_id=data["group_id"],
            total_duration=float(data.get("total_duration", 0.0)),
            member_ids=list(data.get("member_ids", [])),
            audio_result=AudioAnalysisResult.from_dict(audio) if audio else None,
            video_result=VideoAnalysisResult.from_dict(video) if video else None,
            semantic_result=SemanticAnalysisResult.from_dict(semantic) if semantic else None,
            individual_contributions=[
                IndividualContribution.from_dict(c)
                for c in data.get("individual_contributions", [])
            ],
            participation_activity=float(data.get("participation_activity", 0.0)),
            evenness=float(data.get("evenness", 0.0)),
            topic_relevance=opt_float("topic_relevance"),
            interaction_depth=opt_float("interaction_depth"),
            turn_taking_pattern=data.get("turn_taking_pattern", "unknown"),
            consensus_quality=opt_float("consensus_quality"),
            quality_score=float(data.get("quality_score", 0.0)),
            overall_health_score=float(data.get("overall_health_score", 0.0)),
            health_level=CollaborationLevel(data.get("health_level", "critical")),
            diagnoses=list(data.get("diagnoses", [])),
            suggestions=list(data.get("suggestions", [])),
            data_completeness=dict(data.get("data_completeness", {})),
            warnings=list(data.get("warnings", [])),
            created_at=data.get("created_at", ""),
        )

    @classmethod
    def load(cls, path: str) -> "GroupCollaborationReport":
        """从文件加载（与 :meth:`save` 严格对称）。"""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
