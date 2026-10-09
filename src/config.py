"""协作学习分析助手 - 配置模块

设计约束（针对审计问题 #31 #32 #34 #35 修复）：
1. 本模块导入时**不得**产生任何副作用（不建目录、不改全局告警设置）。
2. 数据目录解析必须同时兼容 editable 安装与常规 wheel 安装：
   常规安装时 ``__file__`` 位于 site-packages，绝不能把数据写进 site-packages。
3. 所有路径与阈值均可通过环境变量覆盖，便于复现实验。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ============ 路径解析（无副作用） ============

ENV_PROJECT_ROOT = "CLA_PROJECT_ROOT"
ENV_DATA_DIR = "CLA_DATA_DIR"
ENV_OUTPUT_DIR = "CLA_OUTPUT_DIR"
ENV_MODELS_DIR = "CLA_MODELS_DIR"

#: 打包资源根目录的判定标记
_MARKER = "pyproject.toml"


def _looks_like_project_root(path: Path) -> bool:
    return (path / _MARKER).is_file()


def resolve_project_root() -> Path:
    """解析“项目根目录”。

    优先级：
    1. 环境变量 ``CLA_PROJECT_ROOT``；
    2. 从 ``__file__`` 向上回溯，找到含 ``pyproject.toml`` 的目录；
    3. 当前工作目录。

    绝不会返回 site-packages：若 ``__file__`` 位于 site-packages 内，
    回溯不到标记文件，此时退化为当前工作目录，而不是污染依赖目录。
    """
    env = os.getenv(ENV_PROJECT_ROOT)
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if _looks_like_project_root(candidate):
            return candidate

    # 安装形态（site-packages）或裁剪环境：退化为 cwd，绝不写依赖目录
    return Path.cwd().resolve()


def _resolve_dir(env_name: str, default: Path) -> Path:
    env = os.getenv(env_name)
    if env:
        return Path(env).expanduser().resolve()
    return default


@dataclass(frozen=True)
class Paths:
    """运行时目录。仅在显式调用 :meth:`ensure` 时才创建目录。"""

    project_root: Path
    data: Path
    output: Path
    models: Path

    def ensure(self) -> "Paths":
        """按需创建目录。这是唯一会写文件系统的入口。"""
        for d in (self.data, self.output, self.models):
            d.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(create: bool = False) -> Paths:
    """获取运行时目录配置。

    Args:
        create: 是否立即创建目录。默认 False，避免导入期副作用。
    """
    root = resolve_project_root()
    paths = Paths(
        project_root=root,
        data=_resolve_dir(ENV_DATA_DIR, root / "data"),
        output=_resolve_dir(ENV_OUTPUT_DIR, root / "results"),
        models=_resolve_dir(ENV_MODELS_DIR, root / "models"),
    )
    return paths.ensure() if create else paths


# 兼容旧调用点：惰性代理，访问属性时才解析，不触发 IO
class _LazyPath:
    """向后兼容 ``from config import DATA_DIR`` 的惰性路径代理。"""

    __slots__ = ("_attr",)

    def __init__(self, attr: str) -> None:
        self._attr = attr

    def _path(self) -> Path:
        return getattr(get_paths(), self._attr)

    def __fspath__(self) -> str:
        return str(self._path())

    def __str__(self) -> str:
        return str(self._path())

    def __repr__(self) -> str:
        return f"<lazy path {self._attr} -> {self._path()}>"

    def __truediv__(self, other: str) -> Path:
        return self._path() / other

    def __getattr__(self, item: str):
        return getattr(self._path(), item)


DATA_DIR = _LazyPath("data")
OUTPUT_DIR = _LazyPath("output")
MODELS_DIR = _LazyPath("models")


# ============ .env 加载（不依赖 python-dotenv，无副作用） ============

def _read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


# ============ API 配置 ============

@dataclass
class APIConfig:
    """API 配置"""

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.0

    qwen_api_key: Optional[str] = None
    qwen_model: str = "qwen2.5-72b-instruct"

    vllm_endpoint: Optional[str] = None
    vllm_model: Optional[str] = None

    #: 单次 LLM 请求超时（秒）
    request_timeout: float = 120.0
    #: 失败重试次数
    max_retries: int = 3
    #: 单次请求最大输出 token
    max_output_tokens: int = 2048
    #: 送入 LLM 的最大字符数（超出则按窗口分块，而不是丢弃中段）
    max_prompt_chars: int = 12000


def load_api_config() -> APIConfig:
    """从环境变量 / .env 加载 API 配置。

    环境变量优先于 .env 文件。
    """
    config = APIConfig()
    env_file = _read_env_file(get_paths().project_root / ".env")

    def pick(key: str) -> Optional[str]:
        return os.getenv(key) or env_file.get(key)

    config.openai_api_key = pick("OPENAI_API_KEY")
    config.qwen_api_key = pick("DASHSCOPE_API_KEY") or pick("QWEN_API_KEY")
    config.vllm_endpoint = pick("VLLM_ENDPOINT")
    config.vllm_model = pick("VLLM_MODEL")

    if model := pick("OPENAI_MODEL"):
        config.openai_model = model
    if model := pick("QWEN_MODEL"):
        config.qwen_model = model
    if value := pick("LLM_TEMPERATURE"):
        config.openai_temperature = float(value)
    if value := pick("LLM_MAX_RETRIES"):
        config.max_retries = int(value)
    if value := pick("LLM_MAX_PROMPT_CHARS"):
        config.max_prompt_chars = int(value)

    return config


# ============ 语音智能体配置 ============

@dataclass
class AudioAgentConfig:
    """语音智能体配置"""

    pyannote_token: Optional[str] = None

    whisper_model: str = "base"
    #: 默认交由 Whisper 自行选择设备（None 表示自动）。
    #: 注意：不能传字符串 "auto" —— torch.device("auto") 会抛 RuntimeError。
    whisper_device: Optional[str] = None

    speaker_profiles_dir: Optional[Path] = None

    asr_language: str = "zh"

    #: 说话人分离失败时是否允许降级为纯 ASR。降级会在结果里留下显式标记。
    allow_diarization_fallback: bool = True

    #: 声纹注册的余弦相似度阈值，低于该值视为未匹配
    speaker_match_threshold: float = 0.75


# ============ 视觉智能体配置 ============

@dataclass
class VideoAgentConfig:
    """视觉智能体配置"""

    yolo_model: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45

    tracker: str = "bytetrack"

    pose_model: str = "yolov8n-pose.pt"
    pose_keypoint_threshold: float = 0.5

    #: 朝向判定的角度容差（度）：头-肩向量与目标方向的夹角小于该值视为“看向”
    gaze_angle_threshold: float = 35.0

    #: 指点动作需要持续的最短时长（秒），短于该值的事件在收尾阶段被过滤，
    #: 用于抑制手部抖动造成的假阳性（与帧率无关，比旧版的“连续帧数”更稳）
    pointing_min_duration_seconds: float = 0.4

    #: 指点判定：手腕到目标框中心距离 / 目标框对角线 的阈值
    pointing_distance_ratio: float = 1.25

    #: 采样步长（帧）。>1 表示抽帧推理，用于控制长视频的内存与耗时。
    frame_stride: int = 2

    #: 允许的最大处理帧数，超出则进一步增大采样步长（防止内存爆炸）
    max_processed_frames: int = 30000

    #: 未显式指定时的默认帧率（仅当容器读不出 fps 时才使用，且会在结果中标记）
    fallback_fps: float = 30.0

    #: 材料类目标检测的执行间隔（秒）。逐帧跑两个模型代价过高。
    object_detection_interval_seconds: float = 3.0

    #: 需要检测的材料类别（COCO 类名，小写）
    object_classes: List[str] = field(
        default_factory=lambda: ["book", "cell phone", "laptop", "cup"]
    )


# ============ 语义智能体配置 ============

@dataclass
class SemanticAgentConfig:
    """语义智能体配置"""

    provider: str = "openai"

    #: 模型名。留空则回落到 APIConfig 里对应 provider 的默认模型。
    #: （修复问题 #26：旧实现在 dashscope 分支引用了不存在的 self.config.qwen_model）
    model: Optional[str] = None

    #: 分块窗口大小（秒）。长对话按窗口逐块分析后聚合，而不是截断丢弃中段。
    window_seconds: float = 120.0

    #: 相邻窗口的重叠秒数，避免跨窗口的观点碰撞被切断
    window_overlap_seconds: float = 15.0

    #: 并发请求数上限（控制速率与成本）
    max_concurrency: int = 2

    prompt_template_path: Optional[Path] = None


# ============ 融合引擎配置 ============

@dataclass
class FusionEngineConfig:
    """多模态融合推理引擎配置

    计分模型（修复问题 #1-#9）采用**乘性活跃度门控**：

        health = 100 * activity * quality

        activity  = 组内发言总时长 / (时长 × 目标发言占比)，截断到 [0, 1]
        quality   = w_e * evenness + w_t * topic_relevance + w_d * interaction_depth
        evenness  = (1/HHI - 1) / (n - 1),  HHI = Σ p_i²,  p_i 为发言时长份额

    关键性质：
    * 全员沉默 → activity = 0 → health = 0（可达 critical，修复“沉默得 90 分”）
    * 发言份额完全均等 → evenness = 1；一人独占 → evenness = 0
    * evenness 对全体成员的**水平高低**不敏感，只对**份额分布**敏感（修复偏移量稀释问题）
    """

    #: 目标发言占比：组内实际说话时长 / 总时长。低于该值视为讨论不充分。
    target_talk_ratio: float = 0.45

    #: 质量分内部权重（三项之和应为 1.0）
    quality_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "evenness": 0.40,
            "topic_relevance": 0.25,
            "interaction_depth": 0.35,
        }
    )

    #: 观点碰撞次数的归一化参考值（每分钟）
    collision_reference_per_minute: float = 1.0

    #: 交互深度的子项权重（之和应为 1.0）。缺失子项会在可用项上重新归一化。
    depth_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "collision": 0.45,
            "argument": 0.35,
            "consensus": 0.20,
        }
    )

    #: “关注他人”时长速率的归一化参考值（秒/分钟）
    attention_reference_per_minute: float = 20.0

    #: 指点频率的归一化参考值（次/分钟）
    pointing_reference_per_minute: float = 5.0

    #: 个体贡献度的维度权重（之和应为 1.0）
    contribution_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "speaking": 0.35,
            "turn_taking": 0.15,
            "semantic": 0.35,
            "nonverbal": 0.15,
        }
    )

    #: 等级切分点（下限，升序）
    level_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "critical": 0.0,
            "poor": 30.0,
            "fair": 50.0,
            "good": 70.0,
            "excellent": 85.0,
        }
    )

    score_min: float = 0.0
    score_max: float = 100.0

    def validate(self) -> None:
        """校验权重配置，避免无声的模型漂移。"""
        qw = sum(self.quality_weights.values())
        if abs(qw - 1.0) > 1e-6:
            raise ValueError(f"quality_weights 之和必须为 1.0，当前为 {qw}")
        cw = sum(self.contribution_weights.values())
        if abs(cw - 1.0) > 1e-6:
            raise ValueError(f"contribution_weights 之和必须为 1.0，当前为 {cw}")
        dw = sum(self.depth_weights.values())
        if abs(dw - 1.0) > 1e-6:
            raise ValueError(f"depth_weights 之和必须为 1.0，当前为 {dw}")
        if not 0.0 < self.target_talk_ratio <= 1.0:
            raise ValueError("target_talk_ratio 必须位于 (0, 1]")


# ============ 全局配置单例 ============

_config: Optional[APIConfig] = None


def get_config() -> APIConfig:
    """获取 API 配置单例。"""
    global _config
    if _config is None:
        _config = load_api_config()
    return _config


def reset_config() -> None:
    """清除配置缓存（测试用）。"""
    global _config
    _config = None


def get_audio_config() -> AudioAgentConfig:
    return AudioAgentConfig(
        pyannote_token=os.getenv("PYANNOTE_TOKEN"),
        whisper_device=os.getenv("WHISPER_DEVICE") or None,
        speaker_profiles_dir=get_paths().models / "speaker_profiles",
    )


def get_video_config() -> VideoAgentConfig:
    return VideoAgentConfig()


def get_semantic_config() -> SemanticAgentConfig:
    return SemanticAgentConfig(provider=os.getenv("LLM_PROVIDER", "openai"))


def get_fusion_config() -> FusionEngineConfig:
    config = FusionEngineConfig()
    config.validate()
    return config
