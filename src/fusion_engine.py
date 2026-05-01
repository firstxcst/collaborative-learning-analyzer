"""多模态融合推理引擎"""

import warnings
warnings.filterwarnings("ignore")

from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime

from .config import get_fusion_config, FusionEngineConfig
from .data_models import (
    AudioAnalysisResult,
    VideoAnalysisResult,
    SemanticAnalysisResult,
    IndividualContribution,
    GroupCollaborationReport,
    CollaborationLevel,
)


class FusionEngine:
    """
    多模态融合推理引擎
    
    核心逻辑链：
    1. 个体贡献度 = 发言时长 × 语义相关度 × 非语言投入系数
    2. 小组均衡度 = 贡献度分布的标准差（越小越均衡）
    3. 协作模式质量 = 均衡度 + 观点碰撞次数 + 结论达成效率
    4. 协作健康分 = 加权映射到 0-100
    """
    
    def __init__(self, config: Optional[FusionEngineConfig] = None):
        self.config = config or get_fusion_config()
        
    def fuse(
        self,
        audio_result: AudioAnalysisResult,
        video_result: VideoAnalysisResult,
        semantic_result: SemanticAnalysisResult,
        group_id: str = "group_1",
        member_ids: Optional[List[str]] = None,
    ) -> GroupCollaborationReport:
        """
        融合多模态分析结果，生成协作报告
        
        Args:
            audio_result: 语音分析结果
            video_result: 视觉分析结果
            semantic_result: 语义分析结果
            group_id: 小组标识
            member_ids: 成员 ID 列表（可选，自动推断）
            
        Returns:
            GroupCollaborationReport: 完整的协作报告
        """
        # 推断成员列表
        if member_ids is None:
            member_ids = self._infer_members(audio_result, video_result)
        
        # 创建报告
        report = GroupCollaborationReport(
            group_id=group_id,
            total_duration=audio_result.total_duration,
            member_ids=member_ids,
            audio_result=audio_result,
            video_result=video_result,
            semantic_result=semantic_result,
        )
        
        # 计算每个成员的贡献度
        report.individual_contributions = self._compute_contributions(
            member_ids, audio_result, video_result, semantic_result
        )
        
        # 计算小组整体指标
        report.balance_score = self._compute_balance_score(
            report.individual_contributions
        )
        
        report.collaboration_mode_score = self._compute_collaboration_mode_score(
            report.balance_score,
            semantic_result
        )
        
        # 计算最终健康分
        report.overall_health_score = self._compute_health_score(
            report.balance_score,
            report.collaboration_mode_score,
            semantic_result
        )
        
        # 确定健康等级
        report.health_level = self._determine_health_level(
            report.overall_health_score
        )
        
        # 生成诊断建议
        report.diagnoses, report.suggestions = self._generate_diagnoses(
            report
        )
        
        return report
    
    def _infer_members(
        self,
        audio_result: AudioAnalysisResult,
        video_result: VideoAnalysisResult
    ) -> List[str]:
        """从分析结果中推断成员列表"""
        members = set()
        
        # 从语音分析结果获取
        for speaker_id in audio_result.speaker_stats.keys():
            members.add(speaker_id)
        
        # 从视觉分析结果获取
        for person_id in video_result.person_attention_stats.keys():
            members.add(person_id)
        
        return sorted(list(members))
    
    def _compute_contributions(
        self,
        member_ids: List[str],
        audio_result: AudioAnalysisResult,
        video_result: VideoAnalysisResult,
        semantic_result: SemanticAnalysisResult,
    ) -> List[IndividualContribution]:
        """计算每个成员的贡献度"""
        contributions = []
        weights = self.config.weights
        
        # 计算总发言时长（用于归一化）
        total_speaking_time = sum(
            stats.get("total_duration", 0)
            for stats in audio_result.speaker_stats.values()
        )
        
        for member_id in member_ids:
            contrib = IndividualContribution(person_id=member_id)
            
            # 1. 发言贡献（归一化）
            speaker_stats = audio_result.speaker_stats.get(member_id, {})
            speaking_time = speaker_stats.get("total_duration", 0)
            contrib.speaking_contribution = (
                speaking_time / total_speaking_time 
                if total_speaking_time > 0 else 0
            )
            
            # 2. 语义贡献（使用全局语义相关度）
            contrib.semantic_contribution = semantic_result.topic_relevance
            
            # 3. 非语言投入（基于注意力统计）
            person_stats = video_result.person_attention_stats.get(member_id, {})
            attention_to_others = person_stats.get("attention_to_others", 0)
            # 归一化：假设理想情况下每分钟关注他人 20 秒
            contrib.nonverbal_contribution = min(1.0, attention_to_others / 20.0)
            
            # 4. 指点贡献
            pointing_freq = person_stats.get("pointing_frequency", 0)
            contrib.pointing_contribution = min(1.0, pointing_freq / 10.0)
            
            # 5. 视线跟随贡献
            contrib.gaze_contribution = video_result.cohesion_score
            
            # 计算总分
            contrib.calculate_total(weights)
            
            # 生成个人诊断
            contrib.diagnosis = self._diagnose_individual(
                contrib, speaker_stats, person_stats
            )
            
            contributions.append(contrib)
        
        return contributions
    
    def _diagnose_individual(
        self,
        contrib: IndividualContribution,
        speaker_stats: dict,
        person_stats: dict,
    ) -> Dict[str, Any]:
        """生成个人诊断信息"""
        diagnosis = {
            "strengths": [],
            "weaknesses": [],
            "details": {},
        }
        
        # 发言分析
        turns = speaker_stats.get("turns", 0)
        longest_monologue = speaker_stats.get("longest_monologue", 0)
        
        if turns == 0:
            diagnosis["weaknesses"].append("全程未发言")
        elif turns < 3:
            diagnosis["weaknesses"].append("发言次数较少")
        else:
            diagnosis["strengths"].append(f"积极参与讨论（{turns} 次发言）")
        
        if longest_monologue > 30:
            diagnosis["weaknesses"].append(f"存在长时间独占发言（{longest_monologue:.1f}秒）")
        
        # 非语言参与
        if contrib.nonverbal_contribution < 0.3:
            diagnosis["weaknesses"].append("非语言参与度低")
        
        diagnosis["details"] = {
            "speaking_turns": turns,
            "longest_monologue": longest_monologue,
            "total_score": contrib.total_score,
        }
        
        return diagnosis
    
    def _compute_balance_score(
        self,
        contributions: List[IndividualContribution]
    ) -> float:
        """计算小组均衡度（贡献分布的标准差，越小越均衡）"""
        if not contributions:
            return 0.0
        
        scores = [c.total_score for c in contributions]
        
        if not scores:
            return 0.0
        
        # 计算标准差
        std = np.std(scores)
        
        # 均衡度 = 1 - 标准差（归一化后）
        # 标准差范围 0-0.5，映射到均衡度 1-0
        balance = max(0, 1 - std * 2)
        
        return balance
    
    def _compute_collaboration_mode_score(
        self,
        balance_score: float,
        semantic_result: SemanticAnalysisResult,
    ) -> float:
        """计算协作模式质量"""
        weights = self.config.collaboration_weights
        
        # 观点碰撞归一化（假设理想情况 5-10 次）
        collision_score = min(1.0, semantic_result.opinion_collisions / 5.0)
        
        score = (
            weights.get("balance_score", 0.4) * balance_score +
            weights.get("opinion_collision", 0.3) * collision_score +
            weights.get("argument_depth", 0.3) * semantic_result.argument_depth_score
        )
        
        return score
    
    def _compute_health_score(
        self,
        balance_score: float,
        collaboration_mode_score: float,
        semantic_result: SemanticAnalysisResult,
    ) -> float:
        """计算最终协作健康分（0-100）"""
        # 加权组合
        raw_score = (
            0.3 * balance_score +
            0.4 * collaboration_mode_score +
            0.3 * semantic_result.topic_relevance
        )
        
        # 映射到 0-100
        health_score = raw_score * 100
        
        # 确保范围
        health_score = max(
            self.config.score_min,
            min(self.config.score_max, health_score)
        )
        
        return health_score
    
    def _determine_health_level(self, score: float) -> CollaborationLevel:
        """确定健康等级"""
        if score >= 85:
            return CollaborationLevel.EXCELLENT
        elif score >= 70:
            return CollaborationLevel.GOOD
        elif score >= 50:
            return CollaborationLevel.FAIR
        elif score >= 30:
            return CollaborationLevel.POOR
        else:
            return CollaborationLevel.CRITICAL
    
    def _generate_diagnoses(
        self,
        report: GroupCollaborationReport,
    ) -> tuple:
        """生成诊断和建议"""
        diagnoses = []
        suggestions = []
        
        # 整体健康诊断
        if report.health_level == CollaborationLevel.EXCELLENT:
            diagnoses.append("✅ 小组协作表现优秀，成员参与均衡，观点交流充分")
        elif report.health_level == CollaborationLevel.GOOD:
            diagnoses.append("👍 小组协作良好，但仍有改进空间")
        elif report.health_level == CollaborationLevel.FAIR:
            diagnoses.append("⚠️ 小组协作一般，存在明显问题需要关注")
        elif report.health_level == CollaborationLevel.POOR:
            diagnoses.append("❌ 小组协作较差，需要教师干预")
        else:
            diagnoses.append("🚨 小组协作严重不足，需要立即干预")
        
        # 均衡度诊断
        if report.balance_score < 0.5:
            diagnoses.append("⚠️ 成员参与度不均衡")
            suggestions.append("建议：鼓励沉默的成员发言，或采用轮流发言机制")
        
        # 个人诊断
        for contrib in report.individual_contributions:
            for weakness in contrib.diagnosis.get("weaknesses", []):
                diagnoses.append(f"👤 {contrib.person_id}: {weakness}")
                
                if "全程未发言" in weakness:
                    suggestions.append(
                        f"建议：关注 {contrib.person_id}，了解其不发言的原因"
                    )
        
        # 语义诊断
        if report.semantic_result:
            if report.semantic_result.topic_relevance < 0.5:
                diagnoses.append("⚠️ 讨论偏离主题")
                suggestions.append("建议：引导学生回到讨论主题")
            
            if report.semantic_result.opinion_collisions < 2:
                diagnoses.append("⚠️ 观点碰撞不足，缺乏深度讨论")
                suggestions.append("建议：提出启发性问题，激发观点交流")
        
        return diagnoses, suggestions
