# Collaborative Learning Analyzer - 协作学习分析助手

基于多模态 AI 的小组协作过程性评价工具。通过分析课堂小组讨论的音视频数据，自动生成每位成员的参与度、协作质量等过程性指标。

## 项目背景

新课标要求落实合作学习的过程性评价，但教师走进教室时学生往往"表演式讨论"，缺乏真实、持续的参与度数据。本工具从"只看到结果"转向"全过程量化分析"，输出协作健康分与个性化反馈。

## 系统架构

```
[麦克风阵列+摄像头] → 语音智能体 (Speaker Diarization + ASR)
                  → 视觉智能体 (YOLO + Supervision 姿态/交互检测)
                  → 语义智能体 (LLM 分析对话质量)
                              ↓
                   多模态融合推理引擎
                              ↓
                   协作健康分 & 诊断报告
```

三个智能体分别负责：
- **语音智能体**：谁在说话、说了什么、说多久
- **视觉智能体**：身体朝向、手势、材料互动
- **语义智能体**：对话是否围绕任务、是否有观点碰撞

## 安装

```bash
git clone https://github.com/firstxcst/collaborative-learning-analyzer.git
cd collaborative-learning-analyzer
pip install -e .
```

## 快速开始

### 阶段一：基础数据流（单文件演示）

```bash
python examples/run_analysis.py \
    --audio data/sample_discussion.wav \
    --video data/sample_discussion.mp4 \
    --members 4 \
    --output results/report.json
```

### 配置 API 密钥

```bash
# 设置 OpenAI API（用于语义智能体）
export OPENAI_API_KEY="your-key-here"

# 或创建 .env 文件
echo "OPENAI_API_KEY=your-key-here" > .env
```

## 项目结构

```
collaborative-learning-analyzer/
├── src/
│   ├── audio_agent.py      # 语音智能体（声纹分离 + ASR）
│   ├── video_agent.py      # 视觉智能体（YOLO + 姿态分析）
│   ├── semantic_agent.py   # 语义智能体（LLM 对话分析）
│   ├── fusion_engine.py    # 多模态融合推理引擎
│   ├── data_models.py      # 数据模型定义
│   └── config.py           # 配置文件
├── tests/                  # 单元测试
├── examples/               # 示例脚本
├── docs/                   # 开发文档
└── README.md
```

## 开发路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| Stage 1 | MVP 数据流打通（语音+视觉+语义独立输出） | ✅ 进行中 |
| Stage 2 | 多模态时序融合与推理引擎 | 🔄 待开发 |
| Stage 3 | 真实课堂环境适配 | 🔄 待开发 |
| Stage 4 | 迭代优化与评估 | 🔄 待开发 |

## 技术栈

| 模块 | 技术 |
|------|------|
| 音频处理 | pyannote.audio, whisper, librosa |
| 视觉 | ultralytics (YOLOv8), supervision, OpenCV |
| 语义推理 | OpenAI GPT-4o / Qwen / vLLM |
| 数据存储 | SQLite / PostgreSQL |
| 报告可视化 | matplotlib, seaborn, Streamlit |

## 隐私与伦理声明

⚠️ **重要**：所有分析应在本地或校内服务器完成，不向云端传输原始音视频数据。使用前需获得学生及家长的知情同意。

## 许可证

MIT License