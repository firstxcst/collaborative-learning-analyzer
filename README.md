# Collaborative Learning Analyzer - 协作学习分析助手

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-firstxcst%2Fcollaborative--learning--analyzer-black)](https://github.com/firstxcst/collaborative-learning-analyzer)

**基于多模态 AI 的小组协作过程性评价工具**

分析课堂小组讨论音视频 → 自动生成参与度与协作质量指标

</div>

---

## 📖 项目背景

新课标要求落实合作学习的过程性评价，但教师走进教室时学生往往"表演式讨论"，缺乏真实、持续的参与度数据。本工具从"只看到结果"转向"全过程量化分析"，输出协作健康分与个性化反馈。

### 解决的核心问题

| 痛点 | 解决方案 |
|------|----------|
| 只能通过结果评价小组协作 | 全过程音视频分析，量化参与度 |
| 教师难以同时关注多个小组 | 自动化分析，生成数据报告 |
| 缺乏个性化反馈依据 | 多维度诊断，提供改进建议 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    输入：课堂讨论音视频                           │
│              (麦克风阵列 + 摄像头录制)                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│ 语音智能体 │  │ 视觉智能体 │  │ 语义智能体 │
│           │  │           │  │           │
│ • Speaker │  │ • YOLO    │  │ • GPT-4o  │
│   Diarize │  │   Detect  │  │   /Qwen   │
│ • Whisper │  │ • Pose    │  │ • Topic   │
│   ASR     │  │   Estimate│  │   Relevance│
│           │  │ • Track   │  │ • Opinion │
│           │  │           │  │   Collision│
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │
      └──────────────┼──────────────┘
                     ▼
        ┌────────────────────────┐
        │   多模态融合推理引擎    │
        │                        │
        │  贡献度 = 发言 × 语义  │
        │          × 非语言参与  │
        │                        │
        │  均衡度 = std(贡献度)  │
        │                        │
        │  健康分 = 加权映射     │
        └────────────┬───────────┘
                     ▼
        ┌────────────────────────┐
        │       输出报告         │
        │                        │
        │ • 协作健康分 (0-100)   │
        │ • 个体贡献度分析       │
        │ • 诊断建议             │
        └────────────────────────┘
```

---

## 🚀 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/firstxcst/collaborative-learning-analyzer.git
cd collaborative-learning-analyzer

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -e .
```

### 2. 配置 API 密钥

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
# OPENAI_API_KEY=sk-xxx  (必需，用于语义分析)
# PYANNOTE_TOKEN=xxx     (可选，用于说话人分离)
```

> **获取 API 密钥：**
> - OpenAI: https://platform.openai.com/api-keys
> - pyannote: https://huggingface.co/pyannote/speaker-diarization-3.1 (需注册并接受用户协议)

### 3. 运行示例

```bash
# 基础分析（仅 ASR，跳过说话人分离）
python examples/run_analysis.py \
    --audio data/sample.wav \
    --members 4 \
    --skip-video

# 完整分析（语音 + 视觉 + 语义）
python examples/run_analysis.py \
    --audio data/sample.wav \
    --video data/sample.mp4 \
    --members 4 \
    --output results/report.json
```

### 4. Python API 使用

```python
from src.audio_agent import AudioAgent
from src.video_agent import VideoAgent
from src.semantic_agent import SemanticAgent
from src.fusion_engine import FusionEngine

# 初始化智能体
audio_agent = AudioAgent()
video_agent = VideoAgent()
semantic_agent = SemanticAgent()
fusion_engine = FusionEngine()

# 语音分析
audio_result = audio_agent.analyze(
    "discussion.wav",
    num_speakers=4,
    skip_diarization=True  # MVP 阶段可跳过说话人分离
)

# 语义分析
semantic_result = semantic_agent.analyze(audio_result.segments)

# 视觉分析（可选）
video_result = video_agent.analyze("discussion.mp4", num_members=4)

# 融合生成报告
report = fusion_engine.fuse(
    audio_result,
    video_result,
    semantic_result,
    group_id="group_1"
)

# 输出结果
print(f"协作健康分: {report.overall_health_score:.1f}")
print(f"健康等级: {report.health_level.value}")
for diagnosis in report.diagnoses:
    print(f"  {diagnosis}")
```

---

## 📊 输出示例

```json
{
  "group_id": "group_1",
  "total_duration": 180.5,
  "overall_health_score": 72.5,
  "health_level": "good",
  "balance_score": 0.68,
  "collaboration_mode_score": 0.75,
  "individual_contributions": [
    {
      "person_id": "speaker_1",
      "speaking_contribution": 0.35,
      "semantic_contribution": 0.82,
      "nonverbal_contribution": 0.65,
      "total_score": 0.61
    }
  ],
  "diagnoses": [
    "👍 小组协作良好，但仍有改进空间",
    "👤 speaker_2: 发言次数较少"
  ],
  "suggestions": [
    "建议：鼓励沉默的成员发言，或采用轮流发言机制"
  ]
}
```

---

## 📁 项目结构

```
collaborative-learning-analyzer/
├── src/
│   ├── __init__.py          # 模块初始化
│   ├── audio_agent.py       # 语音智能体 (Speaker Diarization + ASR)
│   ├── video_agent.py       # 视觉智能体 (YOLO + Supervision)
│   ├── semantic_agent.py    # 语义智能体 (LLM 对话分析)
│   ├── fusion_engine.py     # 多模态融合推理引擎
│   ├── data_models.py       # 数据模型定义 (Pydantic)
│   └── config.py            # 配置管理
├── collaborative_learning_analyzer/
│   └── __init__.py          # 包入口
├── tests/
│   └── test_all.py          # 单元测试
├── examples/
│   └── run_analysis.py      # 示例脚本
├── data/                    # 数据目录 (gitignored)
├── results/                 # 输出目录 (gitignored)
├── models/                  # 模型缓存 (gitignored)
├── pyproject.toml           # 项目配置
├── requirements.txt         # 依赖列表
├── .env.example             # 环境变量模板
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔧 核心模块详解

### 语音智能体 (AudioAgent)

**功能：** 回答"谁在说话、说了什么、说多久"

| 能力 | 技术 | 说明 |
|------|------|------|
| 说话人分离 | pyannote.audio 3.1 | 识别不同说话人，需 HF Token |
| 语音转文字 | OpenAI Whisper | 多语言 ASR，带时间戳 |
| 声纹注册 | Resemblyzer | 课前注册学生声纹提高准确率 |

```python
audio_agent = AudioAgent()

# 基础使用：仅 ASR
result = audio_agent.analyze("audio.wav", skip_diarization=True)

# 高级使用：说话人分离
result = audio_agent.analyze("audio.wav", num_speakers=4)

# 输出
for segment in result.segments:
    print(f"{segment.speaker_id}: [{segment.start_time:.1f}s] {segment.text}")
```

### 视觉智能体 (VideoAgent)

**功能：** 分析非语言交互（身体朝向、手势、材料互动）

| 能力 | 技术 | 说明 |
|------|------|------|
| 人员检测 | YOLOv8 | 实时检测画面中的人 |
| 目标追踪 | ByteTrack | 跨帧追踪，分配唯一 ID |
| 姿态估计 | YOLOv8-pose | 17 关键点检测 |
| 交互分析 | Supervision | 指点检测、视线推断 |

```python
video_agent = VideoAgent()

# 完整分析
result = video_agent.analyze("video.mp4", num_members=4)

# 输出
print(f"凝聚度: {result.cohesion_score:.2f}")
for person_id, stats in result.person_attention_stats.items():
    print(f"{person_id}: 关注他人 {stats['attention_to_others']:.1f}s")
```

### 语义智能体 (SemanticAgent)

**功能：** 判断对话是否围绕任务、是否有深度观点碰撞

| 能力 | 输出 | 说明 |
|------|------|------|
| 主题相关度 | 0-1 | 对话是否围绕任务主题 |
| 观点碰撞 | 次数 | 反驳、补充、质疑等深度交互 |
| 论证深度 | 0-1 | 观点是否有充分理由支撑 |
| 轮流模式 | balanced/monopolizing/chaotic | 发言均衡性 |

```python
semantic_agent = SemanticAgent()

# 分析对话
result = semantic_agent.analyze(audio_result.segments)

# 输出
print(f"主题相关度: {result.topic_relevance:.2f}")
print(f"观点碰撞: {result.opinion_collisions} 次")
print(f"论证深度: {result.argument_depth_score:.2f}")
```

### 融合引擎 (FusionEngine)

**功能：** 多模态融合，计算协作健康分

**计算公式：**

```
个体贡献度 = 发言时长(30%) × 语义相关度(30%) 
           + 非语言参与(20%) + 指点频率(10%) + 视线跟随(10%)

均衡度 = 1 - std(贡献度分布)

协作模式质量 = 均衡度(40%) + 观点碰撞(30%) + 论证深度(30%)

协作健康分 = 均衡度(30%) + 协作模式(40%) + 主题相关(30%) × 100
```

**健康等级：**

| 分数范围 | 等级 | 说明 |
|----------|------|------|
| 85-100 | excellent | 协作优秀，无需干预 |
| 70-84 | good | 协作良好，有小改进空间 |
| 50-69 | fair | 协作一般，需关注 |
| 30-49 | poor | 协作较差，需干预 |
| 0-29 | critical | 协作严重不足，立即干预 |

---

## 🛠️ 开发路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| **Stage 1** | MVP 数据流打通（语音+视觉+语义独立输出） | ✅ 完成 |
| **Stage 2** | 多模态时序融合与推理引擎优化 | 🔄 进行中 |
| **Stage 3** | 真实课堂环境适配（多摄像头、噪音处理） | 📅 计划中 |
| **Stage 4** | 迭代优化与教育专家评估 | 📅 计划中 |
| **Stage 5** | Web 可视化界面 + 实时分析 | 📅 计划中 |

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_all.py -v

# 带覆盖率
pytest tests/ --cov=src --cov-report=html
```

---

## 🔒 隐私与伦理声明

⚠️ **重要提醒：**

1. **本地处理**：所有分析应在本地或校内服务器完成，不向云端传输原始音视频数据
2. **知情同意**：使用前需获得学生及家长的知情同意
3. **数据保护**：原始数据应加密存储，定期删除
4. **教育伦理**：分析结果仅供教学参考，不应作为唯一评价依据

---

## 📦 依赖说明

### 核心依赖

| 库 | 版本 | 用途 |
|---|------|------|
| `openai-whisper` | ≥20231117 | 语音识别 |
| `ultralytics` | ≥8.0.0 | YOLO 检测 |
| `supervision` | ≥0.18.0 | 视觉标注 |
| `openai` | ≥1.0.0 | LLM API |
| `numpy` | ≥1.24.0 | 数值计算 |
| `pydub` | ≥0.25.0 | 音频处理 |

### 可选依赖

| 库 | 用途 |
|---|------|
| `pyannote.audio` | 说话人分离（需 HF Token） |
| `resemblyzer` | 声纹注册 |
| `dashscope` | 阿里云 Qwen |
| `streamlit` | Web 界面 |

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📮 联系方式

- GitHub Issues: https://github.com/firstxcst/collaborative-learning-analyzer/issues
- 项目主页: https://github.com/firstxcst/collaborative-learning-analyzer

---

<div align="center">

**Made with ❤️ for Education**

</div>
