"""语音智能体 - 说话人分离 + ASR + 声纹身份对齐

修复的审计问题：
* #30 whisper 设备参数：不再传字符串 ``"auto"``（``torch.device("auto")`` 会抛
  RuntimeError，旧默认配置开箱即崩）；
* #31 移除模块级 ``warnings.filterwarnings("ignore")`` —— 库不得污染宿主进程的告警设置；
* #43 声纹注册从死代码变为真正可用：``align_speakers`` 用声纹把说话人标签对齐到学生身份，
  这是跨模态（语音 ↔ 视觉）身份对齐的前提；
* #22 说话人分离失败时留下显式标记，不再静默降级成“全体同一个人”；
* 文本对齐改为“最大重叠归属”，不再把同一句 ASR 文本重复分配给多个说话人轮次。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # 作为包导入
    from .config import AudioAgentConfig, get_audio_config, get_paths
    from .data_models import AudioAnalysisResult, SpeakingSegment
except ImportError:  # 独立运行
    from config import AudioAgentConfig, get_audio_config, get_paths
    from data_models import AudioAnalysisResult, SpeakingSegment

__all__ = ["AudioAgent"]

#: 无法区分说话人时使用的占位 ID
UNKNOWN_SPEAKER = "unknown"


def _normalize_device(device: Optional[str]) -> Optional[str]:
    """把设备参数规范化为 Whisper 能接受的形式。

    ``whisper.load_model(name, device=None)`` 会自动选择 cuda/cpu；
    而 ``device="auto"`` 会让 ``torch.device("auto")`` 抛 RuntimeError。
    这里把空串与 ``"auto"`` 统一转换为 None。
    """
    if device is None:
        return None
    cleaned = device.strip().lower()
    if cleaned in {"", "auto"}:
        return None
    return device


def _probe_duration(path: str) -> Optional[float]:
    """用 soundfile 读取音频真实时长（不解码整个文件，代价极低）。"""
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


class AudioAgent:
    """语音智能体：回答“谁在说话、说了什么、说多久”"""

    def __init__(self, config: Optional[AudioAgentConfig] = None):
        self.config = config or get_audio_config()
        self._whisper_model = None
        self._diarization_pipeline = None
        self._voice_encoder = None

    # ------------------------------------------------------------ 转录导入

    @staticmethod
    def load_transcript(path: str) -> Tuple[List[SpeakingSegment], List[str]]:
        """从 JSON 文件导入已有转录。

        支持两种结构：
        1. ``{"segments": [{"speaker_id": "S1", "start": 0.0, "end": 3.2, "text": "..."}]}``
        2. 直接是上述 segments 的数组。

        典型用途：分会场麦克风（每个座位一个声道，说话人身份天然已知）、
        教师自带字幕、或已由外部服务完成的 ASR 结果。

        Returns:
            (segments, warnings)
        """
        warnings: List[str] = []
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            items = raw.get("segments") or []
            if raw.get("source"):
                warnings.append(f"转录来源：{raw['source']}")
        elif isinstance(raw, list):
            items = raw
        else:
            raise ValueError("转录文件格式无法识别：应为对象（含 segments）或数组")

        segments: List[SpeakingSegment] = []
        skipped = 0
        for item in items:
            try:
                # speaker_id 是必需字段：缺失时若默默填成 unknown，
                # 会把“数据有问题”伪装成“有效但无法区分说话人”，这里选择显式报错。
                speaker = str(item["speaker_id"]).strip()
                start = float(item["start"])
                end = float(item["end"])
                text = str(item.get("text") or "")
                if not speaker:
                    raise ValueError("speaker_id 为空")
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if end < start:
                start, end = end, start
            segments.append(
                SpeakingSegment(speaker_id=speaker, start_time=start, end_time=end, text=text)
            )

        if skipped:
            warnings.append(f"转录音条中有 {skipped} 条缺少必需字段，已跳过。")
        if not segments:
            warnings.append("转录文件未包含任何有效发言片段。")
        segments.sort(key=lambda s: s.start_time)
        return segments, warnings

    # ------------------------------------------------------------ 模型加载

    def _load_whisper(self):
        if self._whisper_model is None:
            import whisper

            device = _normalize_device(self.config.whisper_device)
            self._whisper_model = whisper.load_model(self.config.whisper_model, device=device)
        return self._whisper_model

    def _load_diarization(self):
        if self._diarization_pipeline is None:
            from pyannote.audio import Pipeline

            if not self.config.pyannote_token:
                raise ValueError(
                    "需要 pyannote token。请在 https://huggingface.co/pyannote/speaker-diarization-3.1 "
                    "注册并接受用户协议后获取 token，再设置环境变量 PYANNOTE_TOKEN"
                )
            self._diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.config.pyannote_token,
            )
        return self._diarization_pipeline

    # ---------------------------------------------------------------- 转写

    def transcribe(self, audio_path: str) -> List[SpeakingSegment]:
        """转录音频，返回带时间戳的文本片段。"""
        model = self._load_whisper()
        result = model.transcribe(
            audio_path,
            language=self.config.asr_language,
            word_timestamps=True,
            verbose=False,
        )
        return [
            SpeakingSegment(
                speaker_id=UNKNOWN_SPEAKER,
                start_time=float(seg["start"]),
                end_time=float(seg["end"]),
                text=str(seg["text"]).strip(),
            )
            for seg in result.get("segments", [])
        ]

    # ------------------------------------------------------------ 说话人分离

    def diarize(
        self, audio_path: str, num_speakers: Optional[int] = None
    ) -> List[SpeakingSegment]:
        """说话人分离，返回带说话人标签的片段。"""
        pipeline = self._load_diarization()
        diarization = pipeline(audio_path, num_speakers=num_speakers)
        return [
            SpeakingSegment(
                speaker_id=str(speaker),
                start_time=float(turn.start),
                end_time=float(turn.end),
                text="",
            )
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

    @staticmethod
    def _overlap(a: SpeakingSegment, b: SpeakingSegment) -> float:
        return max(0.0, min(a.end_time, b.end_time) - max(a.start_time, b.start_time))

    def _merge_segments(
        self, dia_segments: List[SpeakingSegment], asr_segments: List[SpeakingSegment]
    ) -> List[SpeakingSegment]:
        """把 ASR 文本归属到说话人轮次。

        修复旧实现的问题：旧代码把与某个说话人轮次**有任何重叠**的 ASR 片段全部拼接，
        同一句话会被重复分配给相邻的两个说话人。这里改为：
        1. 每个 ASR 片段只归属到重叠最多的那个说话人轮次；
        2. 重叠需超过该 ASR 片段时长的一半，否则丢弃（避免把语气词错配）。
        """
        if not dia_segments:
            return list(asr_segments)

        buckets: Dict[int, List[str]] = {i: [] for i in range(len(dia_segments))}
        for asr in asr_segments:
            if not asr.text:
                continue
            best_idx, best_overlap = -1, 0.0
            for idx, dia in enumerate(dia_segments):
                ov = self._overlap(asr, dia)
                if ov > best_overlap:
                    best_idx, best_overlap = idx, ov
            if best_idx >= 0 and best_overlap >= 0.5 * max(asr.duration, 1e-6):
                buckets[best_idx].append(asr.text)

        merged: List[SpeakingSegment] = []
        for idx, dia in enumerate(dia_segments):
            merged.append(
                SpeakingSegment(
                    speaker_id=dia.speaker_id,
                    start_time=dia.start_time,
                    end_time=dia.end_time,
                    text=" ".join(buckets[idx]),
                )
            )
        return merged

    # ------------------------------------------------------------ 声纹对齐

    def _load_voice_encoder(self):
        if self._voice_encoder is None:
            from resemblyzer import VoiceEncoder

            self._voice_encoder = VoiceEncoder()
        return self._voice_encoder

    def register_speaker(self, audio_path: str, speaker_id: str):
        """注册说话人声纹（课前朗读固定文本的音频）。

        Returns:
            声纹嵌入向量（numpy.ndarray）

        Raises:
            ImportError: 未安装 resemblyzer
        """
        try:
            import librosa
            from resemblyzer import preprocess_wav
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise ImportError(
                "声纹功能需要额外依赖，请安装：pip install resemblyzer librosa"
            ) from exc

        encoder = self._load_voice_encoder()
        wav, _sr = librosa.load(audio_path, sr=16000)
        embedding = encoder.embed_utterance(preprocess_wav(wav))

        profile_dir = self.config.speaker_profiles_dir
        if profile_dir is None:
            profile_dir = get_paths(create=True).models / "speaker_profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)

        import numpy as np

        np.save(profile_dir / f"{speaker_id}.npy", embedding)
        return embedding

    def load_speaker_profiles(self) -> Dict[str, Any]:
        """加载已注册的说话人声纹。"""
        import numpy as np

        profile_dir = self.config.speaker_profiles_dir
        if profile_dir is None or not profile_dir.exists():
            return {}
        return {f.stem: np.load(f) for f in sorted(profile_dir.glob("*.npy"))}

    @staticmethod
    def _cosine(a, b) -> float:
        import numpy as np

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def align_speakers(
        self, audio_path: str, dia_segments: List[SpeakingSegment]
    ) -> Tuple[Dict[str, str], List[str]]:
        """用声纹把分离出的说话人标签对齐到已注册的学生身份。

        这是跨模态身份对齐的基础：只有拿到“说话人标签 → 学生 ID”的映射，
        视觉模块检测到的 ``person_*`` 才能与真实学生对上（修复 #43 / #37）。

        Returns:
            (mapping, warnings)。未注册任何声纹或缺少依赖时返回空映射并给出原因，
            调用方据此判定“身份未对齐”，而不是编造对应关系。
        """
        warnings: List[str] = []
        try:
            profiles = self.load_speaker_profiles()
        except Exception as exc:  # pragma: no cover
            return {}, [f"声纹加载失败：{exc}"]

        if not profiles:
            return {}, [
                "未找到任何已注册声纹，说话人身份未对齐（可用 register_speaker 课前注册）。"
            ]

        try:
            import librosa
            import numpy as np
            from resemblyzer import preprocess_wav

            encoder = self._load_voice_encoder()
            wav, _sr = librosa.load(audio_path, sr=16000)
            wav = preprocess_wav(wav)
        except ImportError as exc:
            return {}, [f"声纹对齐需要 resemblyzer/librosa，当前不可用：{exc}"]

        mapping: Dict[str, str] = {}
        for label in {s.speaker_id for s in dia_segments}:
            spans = [(s.start_time, s.end_time) for s in dia_segments if s.speaker_id == label]
            chunks = []
            for start, end in spans:
                i0, i1 = int(start * 16000), int(end * 16000)
                if i1 > i0:
                    chunks.append(wav[i0:i1])
            if not chunks:
                continue
            embedding = encoder.embed_utterance(np.concatenate(chunks))
            best_id, best_sim = None, 0.0
            for student_id, profile in profiles.items():
                sim = self._cosine(embedding, profile)
                if sim > best_sim:
                    best_id, best_sim = student_id, sim
            if best_id is not None and best_sim >= self.config.speaker_match_threshold:
                mapping[label] = best_id
            else:
                warnings.append(
                    f"说话人 {label} 与已注册声纹的最高相似度为 {best_sim:.3f}，"
                    f"低于阈值 {self.config.speaker_match_threshold}，未对齐。"
                )
        return mapping, warnings

    # ------------------------------------------------------------ 统计与入口

    def _compute_turn_taking_score(self, segments: List[SpeakingSegment]) -> float:
        """轮流发言顺畅度：发言轮次之间的平均停顿越小越顺畅。"""
        if len(segments) < 2:
            return 1.0
        gaps = [
            segments[i].start_time - segments[i - 1].end_time
            for i in range(1, len(segments))
        ]
        positive = [g for g in gaps if g > 0]
        if not positive:
            return 1.0
        avg_gap = sum(positive) / len(positive)
        return max(0.0, min(1.0, 1.0 - avg_gap / 2.0))

    def _compute_speaker_stats(
        self, segments: List[SpeakingSegment]
    ) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, List[SpeakingSegment]] = {}
        for seg in segments:
            grouped.setdefault(seg.speaker_id, []).append(seg)

        stats: Dict[str, Dict[str, Any]] = {}
        for speaker_id, segs in grouped.items():
            segs = sorted(segs, key=lambda s: s.start_time)
            stats[speaker_id] = {
                "total_duration": sum(s.duration for s in segs),
                "turns": len(segs),
                "longest_monologue": max((s.duration for s in segs), default=0.0),
                "turn_taking_score": self._compute_turn_taking_score(segs),
                "text_chars": sum(len(s.text or "") for s in segs),
            }
        return stats

    def analyze(
        self,
        audio_path: Optional[str] = None,
        num_speakers: Optional[int] = None,
        skip_diarization: bool = False,
        align_profiles: bool = False,
        transcript_path: Optional[str] = None,
    ) -> AudioAnalysisResult:
        """完整分析：说话人分离 + ASR（+ 可选声纹对齐）。

        Args:
            audio_path: 音频文件路径。当 ``transcript_path`` 已提供时可为 None
                （此时用转录里的时间轴，不再读音频）。
            num_speakers: 已知说话人人数
            skip_diarization: 跳过说话人分离（仅 ASR）。**注意**：跳过之后无法区分成员，
                个体参与度会退化为全组单一数值，结果对象会带显式告警。
            align_profiles: 是否用声纹把说话人标签对齐到学生身份
            transcript_path: 已有转录文件（JSON）路径。适用于分会场麦克风、
                教师自带字幕或外部 ASR 产出的场景，可跳过本地 ASR 推理。

        Returns:
            AudioAnalysisResult（声纹映射见 ``speaker_mapping``，降级提示见 ``warnings``）
        """
        warnings: List[str] = []

        if transcript_path:
            segments, transcript_warnings = self.load_transcript(transcript_path)
            warnings.extend(transcript_warnings)
            duration = max((s.end_time for s in segments), default=0.0)
            if audio_path:
                probed = _probe_duration(audio_path)
                if probed:
                    duration = max(duration, probed)
            return AudioAnalysisResult(
                segments=segments,
                speaker_stats=self._compute_speaker_stats(segments),
                total_duration=duration,
                diarization_applied=len(
                    {s.speaker_id for s in segments} - {UNKNOWN_SPEAKER}
                ) > 1,
                diarization_error=None,
                total_speaking_time=sum(s.duration for s in segments),
                speaker_mapping={},
                warnings=warnings,
            )

        if not audio_path:
            raise ValueError("必须提供 audio_path 或 transcript_path 之一")

        asr_segments = self.transcribe(audio_path)

        diarization_applied = False
        diarization_error: Optional[str] = None
        mapping: Dict[str, str] = {}

        if skip_diarization:
            warnings.append(
                "已按要求跳过说话人分离：所有发言归入同一个 unknown 成员，无法区分个体。"
            )
            if num_speakers:
                warnings.append(
                    f"传入的 num_speakers={num_speakers} 在跳过说话人分离时不会被使用。"
                )
            merged = asr_segments
        else:
            try:
                dia_segments = self.diarize(audio_path, num_speakers)
                merged = self._merge_segments(dia_segments, asr_segments)
                diarization_applied = True
                if align_profiles:
                    mapping, align_warnings = self.align_speakers(audio_path, dia_segments)
                    warnings.extend(align_warnings)
                    for seg in merged:
                        if seg.speaker_id in mapping:
                            seg.speaker_id = mapping[seg.speaker_id]
            except Exception as exc:
                diarization_error = f"{type(exc).__name__}: {exc}"
                if not self.config.allow_diarization_fallback:
                    raise
                warnings.append(
                    f"说话人分离失败，已降级为纯 ASR（成员区分不可靠）：{diarization_error}"
                )
                merged = asr_segments

        real_duration = _probe_duration(audio_path)
        if real_duration is None and merged:
            warnings.append("未能读取音频真实时长，已用最后一个发言片段的结束时间代替。")
        fallback_duration = max((s.end_time for s in merged), default=0.0)

        return AudioAnalysisResult(
            segments=merged,
            speaker_stats=self._compute_speaker_stats(merged),
            total_duration=real_duration if real_duration else fallback_duration,
            diarization_applied=diarization_applied,
            diarization_error=diarization_error,
            total_speaking_time=sum(s.duration for s in merged),
            speaker_mapping=mapping,
            warnings=warnings,
        )
