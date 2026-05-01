"""协作学习分析助手 - 命令行入口

Usage:
    python -m src --audio discussion.wav
"""

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import json
    
    parser = argparse.ArgumentParser(description="协作学习分析助手")
    parser.add_argument("--audio", help="音频文件路径")
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--members", type=int, default=4, help="小组成员数量")
    parser.add_argument("--output", default="report.json", help="输出文件路径")
    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    args = parser.parse_args()
    
    if args.demo or not args.audio:
        # 演示模式
        from data_models import SpeakingSegment, AudioAnalysisResult, VideoAnalysisResult, SemanticAnalysisResult
        from fusion_engine import FusionEngine
        from config import OUTPUT_DIR
        
        print("=" * 60)
        print("协作学习分析助手 - 演示模式")
        print("=" * 60)
        
        # 模拟数据
        audio = AudioAnalysisResult()
        audio.total_duration = 120.0
        audio.segments = [
            SpeakingSegment("speaker_1", 0, 15, "我觉得..."),
            SpeakingSegment("speaker_2", 16, 28, "我同意..."),
            SpeakingSegment("speaker_3", 30, 45, "补充一点..."),
        ]
        audio.speaker_stats = {
            "speaker_1": {"total_duration": 15.0, "turns": 1, "longest_monologue": 15.0},
            "speaker_2": {"total_duration": 12.0, "turns": 1, "longest_monologue": 12.0},
            "speaker_3": {"total_duration": 15.0, "turns": 1, "longest_monologue": 15.0},
        }
        
        video = VideoAnalysisResult()
        video.cohesion_score = 0.72
        video.person_attention_stats = {
            "speaker_1": {"attention_to_others": 18.5, "pointing_frequency": 3},
            "speaker_2": {"attention_to_others": 22.0, "pointing_frequency": 5},
            "speaker_3": {"attention_to_others": 15.0, "pointing_frequency": 2},
        }
        
        semantic = SemanticAnalysisResult()
        semantic.topic_relevance = 0.85
        semantic.opinion_collisions = 3
        semantic.argument_depth_score = 0.78
        
        # 融合
        engine = FusionEngine()
        report = engine.fuse(audio, video, semantic, group_id="demo")
        
        print(f"\n✅ 协作健康分: {report.overall_health_score:.1f}")
        print(f"   健康等级: {report.health_level.value}")
        
        output = OUTPUT_DIR / args.output
        output.parent.mkdir(exist_ok=True)
        report.save(str(output))
        print(f"\n📄 报告已保存: {output}")
    else:
        print("请使用完整示例脚本: python examples/run_analysis.py")
