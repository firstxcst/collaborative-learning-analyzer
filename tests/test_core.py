# -*- coding: utf-8 -*-
"""独立测试脚本 - 验证核心模块"""
import sys
import io
from pathlib import Path

# 设置 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加 src 到路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data_models import (
    SpeakingSegment, 
    AudioAnalysisResult, 
    VideoAnalysisResult, 
    SemanticAnalysisResult,
    CollaborationLevel
)
from fusion_engine import FusionEngine
from config import OUTPUT_DIR


def test_data_models():
    """测试数据模型"""
    print("[TEST] 数据模型...")
    
    # 测试 SpeakingSegment
    seg = SpeakingSegment("speaker_1", 0.0, 5.0, "测试文本")
    assert seg.duration == 5.0
    assert seg.to_dict()["duration"] == 5.0
    
    # 测试 CollaborationLevel
    assert CollaborationLevel.EXCELLENT.value == "excellent"
    
    print("  [OK] 数据模型测试通过")


def test_fusion_engine():
    """测试融合引擎"""
    print("[TEST] 融合引擎...")
    
    # 创建测试数据
    audio = AudioAnalysisResult()
    audio.total_duration = 120.0
    audio.segments = [
        SpeakingSegment("speaker_1", 0, 30, "这是发言内容"),
        SpeakingSegment("speaker_2", 35, 55, "我同意这个观点"),
        SpeakingSegment("speaker_3", 60, 85, "我补充一点"),
    ]
    audio.speaker_stats = {
        "speaker_1": {"total_duration": 30.0, "turns": 1, "longest_monologue": 30.0},
        "speaker_2": {"total_duration": 20.0, "turns": 1, "longest_monologue": 20.0},
        "speaker_3": {"total_duration": 25.0, "turns": 1, "longest_monologue": 25.0},
    }
    
    video = VideoAnalysisResult()
    video.cohesion_score = 0.75
    video.person_attention_stats = {
        "speaker_1": {"attention_to_others": 20.0, "pointing_frequency": 3},
        "speaker_2": {"attention_to_others": 25.0, "pointing_frequency": 4},
        "speaker_3": {"attention_to_others": 18.0, "pointing_frequency": 2},
    }
    
    semantic = SemanticAnalysisResult()
    semantic.topic_relevance = 0.85
    semantic.opinion_collisions = 4
    semantic.argument_depth_score = 0.78
    
    # 融合
    engine = FusionEngine()
    report = engine.fuse(audio, video, semantic, group_id="test_group")
    
    # 验证
    assert 0 <= report.overall_health_score <= 100
    assert isinstance(report.health_level, CollaborationLevel)
    assert len(report.individual_contributions) == 3
    
    print(f"  [OK] 协作健康分: {report.overall_health_score:.1f}")
    print(f"  [OK] 健康等级: {report.health_level.value}")
    print(f"  [OK] 成员数: {len(report.individual_contributions)}")
    print("  [OK] 融合引擎测试通过")


def test_report_serialization():
    """测试报告序列化"""
    print("[TEST] 报告序列化...")
    
    audio = AudioAnalysisResult()
    audio.speaker_stats = {"s1": {"total_duration": 10, "turns": 1, "longest_monologue": 10}}
    
    video = VideoAnalysisResult()
    video.cohesion_score = 0.7
    video.person_attention_stats = {"s1": {"attention_to_others": 15}}
    
    semantic = SemanticAnalysisResult()
    semantic.topic_relevance = 0.8
    
    engine = FusionEngine()
    report = engine.fuse(audio, video, semantic, group_id="serialization_test")
    
    # 序列化
    data = report.to_dict()
    assert "group_id" in data
    assert "overall_health_score" in data
    
    json_str = report.to_json()
    assert "serialization_test" in json_str
    
    print("  [OK] 报告序列化测试通过")


def main():
    print("=" * 60)
    print("协作学习分析助手 - 核心模块测试")
    print("=" * 60)
    print()
    
    try:
        test_data_models()
        test_fusion_engine()
        test_report_serialization()
        
        print()
        print("=" * 60)
        print("[SUCCESS] 所有测试通过!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n[FAILED] 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
