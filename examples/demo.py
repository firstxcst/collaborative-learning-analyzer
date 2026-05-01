#!/usr/bin/env python3
"""
快速演示脚本（无需真实音视频文件）

演示数据流和输出格式，用于验证系统是否正常工作。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data_models import (
    SpeakingSegment,
    AudioAnalysisResult,
    VideoAnalysisResult,
    SemanticAnalysisResult,
    CollaborationLevel,
)
from fusion_engine import FusionEngine
from config import OUTPUT_DIR


def create_mock_audio_result() -> AudioAnalysisResult:
    """创建模拟的语音分析结果"""
    result = AudioAnalysisResult()
    result.total_duration = 120.0  # 2 分钟讨论
    
    # 模拟发言片段
    result.segments = [
        SpeakingSegment("speaker_1", 0.0, 15.0, "我觉得这个问题的关键在于理解题意..."),
        SpeakingSegment("speaker_2", 16.0, 28.0, "我同意，但我觉得还有另一个角度..."),
        SpeakingSegment("speaker_3", 30.0, 45.0, "你们说的都有道理，我补充一点..."),
        SpeakingSegment("speaker_1", 47.0, 58.0, "那我们综合一下，结论是..."),
        SpeakingSegment("speaker_4", 60.0, 68.0, "我觉得还需要考虑..."),
        SpeakingSegment("speaker_2", 70.0, 85.0, "对，这点很重要..."),
        SpeakingSegment("speaker_3", 88.0, 100.0, "那我们就这样决定吧..."),
    ]
    
    # 模拟统计
    result.speaker_stats = {
        "speaker_1": {"total_duration": 26.0, "turns": 2, "longest_monologue": 15.0},
        "speaker_2": {"total_duration": 27.0, "turns": 2, "longest_monologue": 13.0},
        "speaker_3": {"total_duration": 27.0, "turns": 2, "longest_monologue": 15.0},
        "speaker_4": {"total_duration": 8.0, "turns": 1, "longest_monologue": 8.0},
    }
    
    return result


def create_mock_video_result() -> VideoAnalysisResult:
    """创建模拟的视觉分析结果"""
    result = VideoAnalysisResult()
    result.cohesion_score = 0.72
    
    result.person_attention_stats = {
        "speaker_1": {"attention_to_others": 18.5, "pointing_frequency": 3},
        "speaker_2": {"attention_to_others": 22.0, "pointing_frequency": 5},
        "speaker_3": {"attention_to_others": 15.0, "pointing_frequency": 2},
        "speaker_4": {"attention_to_others": 10.0, "pointing_frequency": 1},
    }
    
    return result


def create_mock_semantic_result() -> SemanticAnalysisResult:
    """创建模拟的语义分析结果"""
    result = SemanticAnalysisResult()
    result.topic_relevance = 0.85
    result.opinion_collisions = 4
    result.turn_taking_pattern = "balanced"
    result.argument_depth_score = 0.78
    result.consensus_quality = 0.82
    result.evidence = [
        "学生1提出了核心观点并给出理由",
        "学生2进行了补充和扩展",
        "学生3提出了不同角度的思考",
        "小组最终达成了共识"
    ]
    
    return result


def main():
    print("=" * 60)
    print("协作学习分析助手 - 演示模式")
    print("=" * 60)
    print()
    
    # 创建模拟数据
    print("[1/3] 创建模拟分析数据...")
    audio_result = create_mock_audio_result()
    video_result = create_mock_video_result()
    semantic_result = create_mock_semantic_result()
    
    print(f"  ✓ 语音: {len(audio_result.segments)} 个发言片段, {audio_result.total_duration:.0f} 秒")
    print(f"  ✓ 视觉: 凝聚度 {video_result.cohesion_score:.2f}")
    print(f"  ✓ 语义: 主题相关度 {semantic_result.topic_relevance:.2f}, 观点碰撞 {semantic_result.opinion_collisions} 次")
    print()
    
    # 融合分析
    print("[2/3] 多模态融合分析...")
    engine = FusionEngine()
    report = engine.fuse(
        audio_result,
        video_result,
        semantic_result,
        group_id="demo_group"
    )
    
    print(f"  ✓ 协作健康分: {report.overall_health_score:.1f}")
    print(f"  ✓ 健康等级: {report.health_level.value}")
    print(f"  ✓ 均衡度: {report.balance_score:.3f}")
    print()
    
    # 输出详细结果
    print("[3/3] 生成报告...")
    print()
    print("-" * 60)
    print("个人贡献度分析")
    print("-" * 60)
    for contrib in report.individual_contributions:
        print(f"\n{contrib.person_id}:")
        print(f"  发言贡献: {contrib.speaking_contribution:.1%}")
        print(f"  语义贡献: {contrib.semantic_contribution:.1%}")
        print(f"  非语言参与: {contrib.nonverbal_contribution:.1%}")
        print(f"  总分: {contrib.total_score:.3f}")
        
        if contrib.diagnosis.get("weaknesses"):
            print(f"  ⚠️ 问题: {', '.join(contrib.diagnosis['weaknesses'])}")
    
    print()
    print("-" * 60)
    print("诊断与建议")
    print("-" * 60)
    for diagnosis in report.diagnoses:
        print(f"  {diagnosis}")
    
    if report.suggestions:
        print()
        print("改进建议:")
        for suggestion in report.suggestions:
            print(f"  💡 {suggestion}")
    
    # 保存报告
    output_path = OUTPUT_DIR / "demo_report.json"
    output_path.parent.mkdir(exist_ok=True)
    report.save(str(output_path))
    
    print()
    print("=" * 60)
    print(f"✅ 演示完成！报告已保存: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
