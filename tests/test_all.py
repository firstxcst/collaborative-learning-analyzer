"""单元测试"""

import pytest
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_models import (
    SpeakingSegment,
    AudioAnalysisResult,
    VideoAnalysisResult,
    SemanticAnalysisResult,
    IndividualContribution,
    GroupCollaborationReport,
    CollaborationLevel,
)
from src.fusion_engine import FusionEngine


class TestDataModels:
    """数据模型测试"""
    
    def test_speaking_segment(self):
        """测试发言片段"""
        segment = SpeakingSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=5.0,
            text="这是测试文本"
        )
        assert segment.duration == 5.0
        assert segment.speaker_id == "speaker_1"
    
    def test_individual_contribution(self):
        """测试个体贡献度"""
        contrib = IndividualContribution(person_id="person_1")
        contrib.speaking_contribution = 0.3
        contrib.semantic_contribution = 0.8
        contrib.nonverbal_contribution = 0.5
        
        weights = {
            "speaking_time": 0.3,
            "semantic_relevance": 0.3,
            "nonverbal_engagement": 0.2,
            "pointing_frequency": 0.1,
            "gaze_attention": 0.1,
        }
        
        total = contrib.calculate_total(weights)
        assert 0 <= total <= 1
    
    def test_collaboration_level(self):
        """测试协作等级"""
        assert CollaborationLevel.EXCELLENT.value == "excellent"
        assert CollaborationLevel.POOR.value == "poor"


class TestFusionEngine:
    """融合引擎测试"""
    
    def test_fuse_basic(self):
        """测试基本融合功能"""
        # 创建模拟数据
        audio_result = AudioAnalysisResult()
        audio_result.segments = [
            SpeakingSegment("speaker_1", 0, 5, "测试发言1"),
            SpeakingSegment("speaker_2", 5, 10, "测试发言2"),
        ]
        audio_result.speaker_stats = {
            "speaker_1": {"total_duration": 5.0, "turns": 1, "longest_monologue": 5.0},
            "speaker_2": {"total_duration": 5.0, "turns": 1, "longest_monologue": 5.0},
        }
        audio_result.total_duration = 10.0
        
        video_result = VideoAnalysisResult()
        video_result.cohesion_score = 0.7
        video_result.person_attention_stats = {
            "speaker_1": {"attention_to_others": 15.0, "pointing_frequency": 2},
            "speaker_2": {"attention_to_others": 15.0, "pointing_frequency": 2},
        }
        
        semantic_result = SemanticAnalysisResult()
        semantic_result.topic_relevance = 0.8
        semantic_result.opinion_collisions = 3
        semantic_result.argument_depth_score = 0.7
        
        # 融合
        engine = FusionEngine()
        report = engine.fuse(
            audio_result,
            video_result,
            semantic_result,
            group_id="test_group",
            member_ids=["speaker_1", "speaker_2"]
        )
        
        # 验证
        assert report.group_id == "test_group"
        assert 0 <= report.overall_health_score <= 100
        assert isinstance(report.health_level, CollaborationLevel)
        assert len(report.individual_contributions) == 2
    
    def test_health_level_determination(self):
        """测试健康等级判定"""
        engine = FusionEngine()
        
        assert engine._determine_health_level(90) == CollaborationLevel.EXCELLENT
        assert engine._determine_health_level(75) == CollaborationLevel.GOOD
        assert engine._determine_health_level(55) == CollaborationLevel.FAIR
        assert engine._determine_health_level(40) == CollaborationLevel.POOR
        assert engine._determine_health_level(20) == CollaborationLevel.CRITICAL


class TestReportSerialization:
    """报告序列化测试"""
    
    def test_to_dict(self):
        """测试字典序列化"""
        report = GroupCollaborationReport(
            group_id="test",
            total_duration=60.0,
            member_ids=["p1", "p2"],
            overall_health_score=75.0,
            health_level=CollaborationLevel.GOOD,
        )
        
        data = report.to_dict()
        
        assert data["group_id"] == "test"
        assert data["overall_health_score"] == 75.0
        assert data["health_level"] == "good"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
