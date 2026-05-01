#!/usr/bin/env python3
"""示例：完整分析流程演示

Usage:
    python examples/run_analysis.py --audio data/sample.wav --video data/sample.mp4
"""

import argparse
import json
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.audio_agent import AudioAgent
from src.video_agent import VideoAgent
from src.semantic_agent import SemanticAgent
from src.fusion_engine import FusionEngine
from src.config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="协作学习分析助手")
    parser.add_argument("--audio", type=str, help="音频文件路径")
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--members", type=int, default=4, help="小组成员数量")
    parser.add_argument("--output", type=str, default="report.json", help="输出文件名")
    parser.add_argument("--skip-audio", action="store_true", help="跳过语音分析")
    parser.add_argument("--skip-video", action="store_true", help="跳过视觉分析")
    parser.add_argument("--skip-semantic", action="store_true", help="跳过语义分析")
    args = parser.parse_args()
    
    print("=" * 60)
    print("协作学习分析助手")
    print("=" * 60)
    
    # 初始化智能体
    audio_agent = AudioAgent()
    video_agent = VideoAgent()
    semantic_agent = SemanticAgent()
    fusion_engine = FusionEngine()
    
    # 语音分析
    audio_result = None
    if args.audio and not args.skip_audio:
        print("\n[1/4] 语音分析中...")
        try:
            audio_result = audio_agent.analyze(
                args.audio, 
                num_speakers=args.members,
                skip_diarization=True  # MVP 阶段跳过说话人分离
            )
            print(f"  ✓ 检测到 {len(audio_result.segments)} 个发言片段")
            print(f"  ✓ 总时长: {audio_result.total_duration:.1f} 秒")
        except Exception as e:
            print(f"  ✗ 语音分析失败: {e}")
    
    # 视觉分析
    video_result = None
    if args.video and not args.skip_video:
        print("\n[2/4] 视觉分析中...")
        try:
            video_result = video_agent.analyze(args.video, num_members=args.members)
            print(f"  ✓ 凝聚度: {video_result.cohesion_score:.2f}")
        except Exception as e:
            print(f"  ✗ 视觉分析失败: {e}")
    
    # 语义分析
    semantic_result = None
    if audio_result and not args.skip_semantic:
        print("\n[3/4] 语义分析中...")
        try:
            semantic_result = semantic_agent.analyze(audio_result.segments)
            print(f"  ✓ 主题相关度: {semantic_result.topic_relevance:.2f}")
            print(f"  ✓ 观点碰撞: {semantic_result.opinion_collisions} 次")
            print(f"  ✓ 论证深度: {semantic_result.argument_depth_score:.2f}")
        except Exception as e:
            print(f"  ✗ 语义分析失败: {e}")
    
    # 多模态融合
    report = None
    if audio_result and video_result and semantic_result:
        print("\n[4/4] 多模态融合中...")
        try:
            report = fusion_engine.fuse(
                audio_result, 
                video_result, 
                semantic_result,
                group_id="demo_group"
            )
            print(f"  ✓ 协作健康分: {report.overall_health_score:.1f}")
            print(f"  ✓ 健康等级: {report.health_level.value}")
        except Exception as e:
            print(f"  ✗ 融合分析失败: {e}")
    
    # 输出报告
    if report:
        output_path = OUTPUT_DIR / args.output
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 报告已保存: {output_path}")
        
        # 打印诊断
        print("\n" + "=" * 60)
        print("诊断结果")
        print("=" * 60)
        for diagnosis in report.diagnoses:
            print(f"  {diagnosis}")
        
        if report.suggestions:
            print("\n建议:")
            for suggestion in report.suggestions:
                print(f"  • {suggestion}")
    else:
        print("\n⚠️ 无法生成完整报告，请检查输入文件")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
