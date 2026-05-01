"""协作学习分析助手 - 配置模块"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

# ============ 路径配置 ============
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ============ API 配置 ============
@dataclass
class APIConfig:
    """API 配置"""
    openai_api_key: Optional[str] = field(default=None)
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.3
    
    qwen_api_key: Optional[str] = None
    qwen_model: str = "qwen2.5-72b-instruct"
    
    # 可选：本地 vLLM 配置
    vllm_endpoint: Optional[str] = None
    vllm_model: Optional[str] = None


def load_api_config() -> APIConfig:
    """从环境变量或 .env 文件加载 API 配置"""
    config = APIConfig()
    
    # 从环境变量读取
    config.openai_api_key = os.getenv("OPENAI_API_KEY")
    config.qwen_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    config.vllm_endpoint = os.getenv("VLLM_ENDPOINT")
    
    # 从 .env 文件读取（如果存在）
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "OPENAI_API_KEY":
                    config.openai_api_key = config.openai_api_key or value
                elif key in ("DASHSCOPE_API_KEY", "QWEN_API_KEY"):
                    config.qwen_api_key = config.qwen_api_key or value
                elif key == "VLLM_ENDPOINT":
                    config.vllm_endpoint = value
    
    return config


# ============ 语音智能体配置 ============
@dataclass
class AudioAgentConfig:
    """语音智能体配置"""
    # pyannote 配置（需要从 huggingface.co 注册获取 token）
    pyannote_token: Optional[str] = None
    
    # Whisper 配置
    whisper_model: str = "base"  # tiny/base/small/medium/large
    whisper_device: str = "auto"  # auto/cpu/cuda
    
    # 说话人注册：课前每位学生朗读固定文本生成的声纹向量存储路径
    speaker_profiles_dir: Path = field(default_factory=lambda: MODELS_DIR / "speaker_profiles")
    
    # ASR 语言
    asr_language: str = "zh"  # 中文


# ============ 视觉智能体配置 ============
@dataclass
class VideoAgentConfig:
    """视觉智能体配置"""
    # YOLO 模型
    yolo_model: str = "yolov8n.pt"  # yolov8n/yolov8s/yolov8m/yolov8l/yolov8x
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    
    # 追踪器
    tracker: str = "bytetrack"  # bytetrack / botsort
    tracker_conf_threshold: float = 0.3
    
    # 姿态估计（可选，用于更精细的朝向分析）
    pose_model: str = "yolov8n-pose.pt"
    pose_keypoint_threshold: float = 0.5
    
    # 交互检测阈值
    pointing_duration_ms: int = 1000  # 指点动作持续时间阈值
    gaze_angle_threshold: float = 30  # 视线朝向判定阈值（度）
    
    # 检测目标类别
    detected_classes: list = field(default_factory=lambda: ["person"])
    object_classes: list = field(default_factory=lambda: ["book", "paper", "pen"])


# ============ 语义智能体配置 ============
@dataclass
class SemanticAgentConfig:
    """语义智能体配置"""
    # LLM 提供者：openai / dashscope / vllm
    provider: str = "openai"
    
    # 批处理配置
    batch_size: int = 5  # 每次送入 LLM 的对话段落数
    compress_before_llm: bool = True  # 是否在送入前对转录进行摘要压缩
    
    # Prompt 模板路径
    prompt_template_path: Optional[Path] = None


# ============ 融合引擎配置 ============
@dataclass
class FusionEngineConfig:
    """多模态融合推理引擎配置"""
    # 时间窗口大小（秒）
    time_window_seconds: float = 30.0
    
    # 个体贡献度权重
    weights: dict = field(default_factory=lambda: {
        "speaking_time": 0.3,
        "semantic_relevance": 0.3,
        "nonverbal_engagement": 0.2,
        "pointing_frequency": 0.1,
        "gaze_attention": 0.1,
    })
    
    # 协作模式权重
    collaboration_weights: dict = field(default_factory=lambda: {
        "balance_score": 0.4,
        "opinion_collision": 0.3,
        "argument_depth": 0.3,
    })
    
    # 最终分数映射范围
    score_min: float = 0.0
    score_max: float = 100.0


# ============ 全局配置单例 ============
_config: Optional[APIConfig] = None


def get_config() -> APIConfig:
    """获取 API 配置单例"""
    global _config
    if _config is None:
        _config = load_api_config()
    return _config


def get_audio_config() -> AudioAgentConfig:
    """获取语音智能体配置"""
    return AudioAgentConfig(
        pyannote_token=os.getenv("PYANNOTE_TOKEN"),
    )


def get_video_config() -> VideoAgentConfig:
    """获取视觉智能体配置"""
    return VideoAgentConfig()


def get_semantic_config() -> SemanticAgentConfig:
    """获取语义智能体配置"""
    provider = os.getenv("LLM_PROVIDER", "openai")
    return SemanticAgentConfig(provider=provider)


def get_fusion_config() -> FusionEngineConfig:
    """获取融合引擎配置"""
    return FusionEngineConfig()