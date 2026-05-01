"""语音智能体 - 说话人分离 + ASR"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from .config import get_audio_config, AudioAgentConfig
from .data_models import AudioAnalysisResult, SpeakingSegment


class AudioAgent:
    """
    语音智能体：回答"谁在说话、说了什么、说多久"
    
    核心能力：
    1. 说话人分离（Speaker Diarization）- 识别不同说话人
    2. 语音转文字（ASR）- 生成带时间戳的转录
    3. 声纹注册 - 课前注册学生声纹提高准确率
    """
    
    def __init__(self, config: Optional[AudioAgentConfig] = None):
        self.config = config or get_audio_config()
        self._whisper_model = None
        self._diarization_pipeline = None
        
    def _load_whisper(self):
        """懒加载 Whisper 模型"""
        if self._whisper_model is None:
            import whisper
            self._whisper_model = whisper.load_model(
                self.config.whisper_model,
                device=self.config.whisper_device
            )
        return self._whisper_model
    
    def _load_diarization(self):
        """懒加载 pyannote 说话人分离模型"""
        if self._diarization_pipeline is None:
            from pyannote.audio import Pipeline
            if self.config.pyannote_token:
                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.config.pyannote_token
                )
            else:
                raise ValueError(
                    "需要 pyannote token。请在 https://huggingface.co/pyannote/speaker-diarization-3.1 "
                    "注册并获取 token，然后设置环境变量 PYANNOTE_TOKEN"
                )
        return self._diarization_pipeline
    
    def transcribe(self, audio_path: str) -> List[SpeakingSegment]:
        """
        转录音频文件，返回带时间戳的文本片段
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            List[SpeakingSegment]: 发言片段列表
        """
        model = self._load_whisper()
        result = model.transcribe(
            audio_path,
            language=self.config.asr_language,
            word_timestamps=True
        )
        
        segments = []
        for seg in result.get("segments", []):
            segments.append(SpeakingSegment(
                speaker_id="unknown",  # 后续由 diarization 填充
                start_time=seg["start"],
                end_time=seg["end"],
                text=seg["text"].strip()
            ))
        
        return segments
    
    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> List[SpeakingSegment]:
        """
        说话人分离，识别不同说话人
        
        Args:
            audio_path: 音频文件路径
            num_speakers: 已知说话人数量（可选）
            
        Returns:
            List[SpeakingSegment]: 带说话人 ID 的片段列表
        """
        pipeline = self._load_diarization()
        
        # 运行说话人分离
        diarization = pipeline(audio_path, num_speakers=num_speakers)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakingSegment(
                speaker_id=speaker,
                start_time=turn.start,
                end_time=turn.end,
                text=""  # 文本由 ASR 填充
            ))
        
        return segments
    
    def analyze(
        self, 
        audio_path: str, 
        num_speakers: Optional[int] = None,
        skip_diarization: bool = False
    ) -> AudioAnalysisResult:
        """
        完整分析：说话人分离 + ASR
        
        Args:
            audio_path: 音频文件路径
            num_speakers: 已知说话人数量
            skip_diarization: 跳过说话人分离（仅 ASR）
            
        Returns:
            AudioAnalysisResult: 分析结果
        """
        # ASR 转录
        asr_segments = self.transcribe(audio_path)
        
        if not skip_diarization:
            try:
                # 说话人分离
                dia_segments = self.diarize(audio_path, num_speakers)
                
                # 合并：将 ASR 文本对齐到说话人片段
                merged = self._merge_segments(dia_segments, asr_segments)
            except Exception as e:
                print(f"说话人分离失败: {e}，仅使用 ASR 结果")
                merged = asr_segments
        else:
            merged = asr_segments
        
        # 计算统计指标
        result = AudioAnalysisResult(segments=merged)
        result.speaker_stats = self._compute_speaker_stats(merged)
        result.total_duration = max(s.end_time for s in merged) if merged else 0.0
        
        return result
    
    def _merge_segments(
        self, 
        dia_segments: List[SpeakingSegment],
        asr_segments: List[SpeakingSegment]
    ) -> List[SpeakingSegment]:
        """将 ASR 文本对齐到说话人分离片段"""
        merged = []
        
        for dia in dia_segments:
            # 找到时间重叠的 ASR 片段
            overlapping_text = []
            for asr in asr_segments:
                # 检查时间重叠
                if asr.start_time < dia.end_time and asr.end_time > dia.start_time:
                    overlapping_text.append(asr.text)
            
            merged.append(SpeakingSegment(
                speaker_id=dia.speaker_id,
                start_time=dia.start_time,
                end_time=dia.end_time,
                text=" ".join(overlapping_text)
            ))
        
        return merged
    
    def _compute_speaker_stats(self, segments: List[SpeakingSegment]) -> Dict[str, dict]:
        """计算每个说话人的统计指标"""
        stats = {}
        
        # 按说话人分组
        speaker_segments: Dict[str, List[SpeakingSegment]] = {}
        for seg in segments:
            if seg.speaker_id not in speaker_segments:
                speaker_segments[seg.speaker_id] = []
            speaker_segments[seg.speaker_id].append(seg)
        
        for speaker_id, segs in speaker_segments.items():
            total_duration = sum(s.duration for s in segs)
            turns = len(segs)
            longest_monologue = max(s.duration for s in segs) if segs else 0
            
            # 计算轮流发言顺畅度
            turn_taking_score = self._compute_turn_taking_score(segs)
            
            stats[speaker_id] = {
                "total_duration": total_duration,
                "turns": turns,
                "longest_monologue": longest_monologue,
                "turn_taking_score": turn_taking_score,
            }
        
        return stats
    
    def _compute_turn_taking_score(self, segments: List[SpeakingSegment]) -> float:
        """计算轮流发言顺畅度（简化版）"""
        if len(segments) < 2:
            return 1.0
        
        # 计算片段间的间隔
        gaps = []
        for i in range(1, len(segments)):
            gap = segments[i].start_time - segments[i-1].end_time
            gaps.append(gap)
        
        # 间隔越小越顺畅
        avg_gap = np.mean(gaps) if gaps else 0
        # 将间隔映射到 0-1 分数（间隔 0-2 秒为 1-0）
        score = max(0, 1 - avg_gap / 2.0)
        return score
    
    def register_speaker(self, audio_path: str, speaker_id: str) -> np.ndarray:
        """
        注册说话人声纹
        
        Args:
            audio_path: 说话人朗读固定文本的音频
            speaker_id: 说话人标识
            
        Returns:
            np.ndarray: 声纹嵌入向量
        """
        # 使用 Resemblyzer 或 pyannote 提取声纹嵌入
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            import librosa
            
            encoder = VoiceEncoder()
            wav, sr = librosa.load(audio_path, sr=16000)
            wav_processed = preprocess_wav(wav)
            embedding = encoder.embed_utterance(wav_processed)
            
            # 保存声纹
            profile_dir = self.config.speaker_profiles_dir
            profile_dir.mkdir(exist_ok=True, parents=True)
            np.save(profile_dir / f"{speaker_id}.npy", embedding)
            
            return embedding
        except ImportError:
            raise ImportError("请安装 resemblyzer: pip install resemblyzer")
    
    def load_speaker_profiles(self) -> Dict[str, np.ndarray]:
        """加载已注册的说话人声纹"""
        profiles = {}
        profile_dir = self.config.speaker_profiles_dir
        
        if not profile_dir.exists():
            return profiles
        
        for file in profile_dir.glob("*.npy"):
            speaker_id = file.stem
            profiles[speaker_id] = np.load(file)
        
        return profiles
