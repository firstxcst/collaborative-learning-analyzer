#!/usr/bin/env python3
"""生成用于端到端验证的**真实媒体夹具**。

为什么需要它
------------
仓库不可能附带真实课堂录像（涉及学生隐私，也会让仓库体积失控）。但没有媒体文件，
任何人 clone 下来都跑不出任何结果 —— 这正是审计问题 #60 指出的。

本脚本生成的是**真实媒体**，不是“模拟数据”：
* ``discussion.wav`` —— 用 Windows 内置中文语音合成引擎（SAPI）逐句合成的**真实音频**，
  多位不同音色分别扮演小组成员，可直接被 Whisper 转写；
* ``discussion.transcript.json`` —— 与上述音频逐句对应的**真实时间轴与文本**
  （由合成过程实测得到，不是手编的假数据），供 ``--transcript`` 路径使用；
* ``speaker_references/*.wav`` —— 每位成员的真实参考音频，可用于声纹注册测试；
* ``discussion.mp4`` —— 真实视频文件（真实像素、真实帧率、真实编码），
  用于验证视频解码、帧率读取与标注输出。

**明确的局限（请勿误用）**
--------------------------
1. 合成语音 ≠ 真实课堂录音：没有远场混响、没有重叠说话、没有环境噪声。
   因此在本夹具上测出的 ASR / 说话人分离表现**显著优于**真实课堂，不可作为精度证据。
2. ``discussion.mp4`` 中**没有人**，只有几何图形。YOLO 人体/姿态检测不会命中它，
   所以它只能验证视频 IO 与聚合逻辑，**不能**验证人体检测或视线估计精度。
3. 本夹具的用途是**回归验证与冒烟测试**，不是效果评估，更不是效度证据。

关于音色数量
------------
SAPI 可见的音色集合取决于调用进程的宿主环境：在部分环境下只能看到 1 个中文音色，
在另一些环境下能看到 4 个。本脚本会**自适应**可用音色数，并用语速差异增强区分度，
同时在音色不足时打印告警。用 ``--list-voices`` 可以查看当前环境能看到什么。

用法::

    python tools/make_fixtures.py --list-voices
    python tools/make_fixtures.py --out fixtures
    python tools/make_fixtures.py --out fixtures --tts-dir <已合成的 utt_*.wav 目录>
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent

#: 小组成员：标识 / 优先选用的音色 / 语速
SPEAKERS: List[Dict[str, Any]] = [
    {"id": "stu_A", "prefer": "Huihui Desktop", "rate": 0},
    {"id": "stu_B", "prefer": "Kangkang", "rate": -2},
    {"id": "stu_C", "prefer": "Yaoyao", "rate": 2},
    {"id": "stu_D", "prefer": "Huihui", "rate": 0},
]

#: 讨论脚本：围绕明确主题、含真实观点交锋的四人讨论
DISCUSSION_TOPIC = "为什么天空是蓝色的"
#: 主题关键词（离线规则基线需要关键词而不是整句，理由见 src/semantic_agent.py）
DISCUSSION_KEYWORDS = "蓝光, 散射, 大气, 波长, 晚霞, 太阳光"

SCRIPT: List[Dict[str, str]] = [
    {"speaker": "stu_A", "text": "我觉得这个问题要从光的散射来解释，太阳光里包含了各种颜色。"},
    {"speaker": "stu_B", "text": "但是为什么偏偏是蓝色散射得最厉害，而不是红色呢？"},
    {"speaker": "stu_C", "text": "因为蓝光的波长比较短，遇到空气中的分子以后更容易被散射出去。"},
    {"speaker": "stu_A", "text": "对，所以从各个方向看过去都是蓝光进入我们的眼睛。"},
    {"speaker": "stu_D", "text": "我补充一点，傍晚的时候太阳角度低，光要穿过更厚的大气层。"},
    {"speaker": "stu_B", "text": "那按这个道理，蓝光是不是在傍晚都被散射完了，所以才剩下红橙色？"},
    {"speaker": "stu_C", "text": "没错，这就是为什么晚霞是红色的，这说明推理是自洽的。"},
    {"speaker": "stu_D", "text": "那么我们可以总结一下，蓝光散射强和路径变长这两条共同解释了天空的颜色变化。"},
    {"speaker": "stu_A", "text": "我同意这个结论，不过我还有一个疑问，太空里为什么是黑的。"},
    {"speaker": "stu_B", "text": "因为太空几乎没有大气分子，没有东西可以散射光线。"},
    {"speaker": "stu_C", "text": "所以如果在月球上看天空，应该一直是黑色的，这个可以验证。"},
    {"speaker": "stu_D", "text": "那我们就用散射强度和大气厚度这两个变量来解释所有情况。"},
]

GAP_AFTER = 0.35
LEAD_SILENCE = 1.0
TAIL_SILENCE = 1.0
TARGET_SR = 16000
FADE_SECONDS = 0.02  # 句首句尾淡入淡出，避免拼接爆音

POWERSHELL_CANDIDATES = [
    "powershell",
    r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    r"C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe",
]

_LIST_SCRIPT = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.GetInstalledVoices() | ForEach-Object { "
    "$_.VoiceInfo.Name + '|' + $_.VoiceInfo.Culture.Name }; "
    "$s.Dispose()"
)


def _require(module: str):
    try:
        return __import__(module)
    except ImportError:
        print(f"缺少依赖 {module}。请安装：pip install numpy soundfile", file=sys.stderr)
        raise SystemExit(1)


def list_voices() -> List[Tuple[str, str]]:
    """列出所有 PowerShell 宿主可见的 SAPI 音色，返回 (名称, 区域) 去重列表。"""
    found: Dict[str, str] = {}
    for exe in POWERSHELL_CANDIDATES:
        try:
            result = subprocess.run(
                [exe, "-NoProfile", "-NonInteractive", "-Command", _LIST_SCRIPT],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode != 0:
            continue
        for line in (result.stdout or "").splitlines():
            if "|" not in line:
                continue
            name, _, culture = line.strip().partition("|")
            if name:
                found[name.strip()] = culture.strip()
    return sorted(found.items())


def pick_voices(
    available: Sequence[Tuple[str, str]], speakers: Sequence[Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """为每位成员选一个音色，并在音色不足时给出告警。"""
    warnings: List[str] = []
    if not available:
        raise RuntimeError("未找到任何 SAPI 音色，无法生成音频夹具。")

    zh = [name for name, culture in available if culture.lower().startswith("zh")]
    pool = zh or [name for name, _ in available]
    if not zh:
        warnings.append("当前环境没有中文音色，将使用非中文音色合成（转写语言需相应调整）。")
    if len(pool) < len(speakers):
        warnings.append(
            f"可选音色只有 {len(pool)} 个，少于成员数 {len(speakers)}；"
            "重复使用音色并靠语速差异增加区分度，这会让说话人分离变难。"
        )

    names = [name for name, _ in available]
    chosen: List[str] = []
    used: set[str] = set()
    for index, speaker in enumerate(speakers):
        preference = str(speaker["prefer"]).lower()
        match = next(
            (n for n in pool if preference in n.lower() and n not in used),
            None,
        )
        if match is None:
            match = next((n for n in pool if n not in used), pool[index % len(pool)])
        used.add(match)
        chosen.append(match if match in names else pool[index % len(pool)])
    return chosen, warnings


def synthesize(jobs: List[Dict[str, Any]], tmp_dir: Path) -> None:
    """调用 PowerShell + SAPI 把每句话合成到独立 wav。"""
    jobs_file = tmp_dir / "jobs.json"
    jobs_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")

    last_error = ""
    for exe in POWERSHELL_CANDIDATES:
        try:
            result = subprocess.run(
                [
                    exe, "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(SCRIPT_DIR / "synth_tts.ps1"),
                    "-JobsFile", str(jobs_file),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=900,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = f"{exe}: {exc}"
            continue
        if result.returncode == 0 and "SYNTH_DONE" in (result.stdout or ""):
            return
        last_error = f"{exe}: rc={result.returncode}\n{result.stdout}\n{result.stderr}"
    raise RuntimeError(f"语音合成失败。最后错误：\n{last_error}")


def _resample(samples, src_sr: int, dst_sr: int):
    np = _require("numpy")
    data = np.asarray(samples, dtype=np.float32)
    if src_sr == dst_sr or data.size == 0:
        return data
    target_len = int(round(data.size / src_sr * dst_sr))
    if target_len <= 0:
        return data[:0]
    return np.interp(
        np.linspace(0.0, data.size - 1, num=target_len),
        np.arange(data.size),
        data,
    ).astype(np.float32)


def read_wav_mono(path: Path, target_sr: int):
    sf = _require("soundfile")
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return _resample(data.mean(axis=1), int(sr), target_sr), target_sr


def write_wav(path: Path, samples, sr: int) -> None:
    sf = _require("soundfile")
    np = _require("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0),
        sr,
        subtype="PCM_16",
    )


def build_audio(
    out_dir: Path, tmp_dir: Path, tts_dir: Optional[Path] = None
) -> Dict[str, Any]:
    np = _require("numpy")
    warnings: List[str] = []

    plan: List[Dict[str, Any]] = [
        {
            "index": index,
            "speaker": line["speaker"],
            "text": line["text"],
            "path": (tts_dir or tmp_dir) / f"utt_{index:03d}.wav",
        }
        for index, line in enumerate(SCRIPT)
    ]

    voice_info: Dict[str, Any] = {}
    if tts_dir is None:
        available = list_voices()
        if not available:
            raise RuntimeError("未找到任何 SAPI 音色，无法生成音频夹具。")
        chosen, warnings = pick_voices(available, SPEAKERS)
        speaker_voice = {s["id"]: chosen[i] for i, s in enumerate(SPEAKERS)}
        jobs = [
            {
                "voice": speaker_voice[item["speaker"]],
                "rate": next(s["rate"] for s in SPEAKERS if s["id"] == item["speaker"]),
                "text": item["text"],
                "out": str(item["path"]),
            }
            for item in plan
        ]
        synthesize(jobs, tmp_dir)
        voice_info = {
            "available": [f"{n} ({c})" for n, c in available],
            "assigned": speaker_voice,
            "warning": warnings or None,
        }
    else:
        missing = [str(item["path"]) for item in plan if not item["path"].is_file()]
        if missing:
            raise RuntimeError(f"--tts-dir 中缺少 {len(missing)} 个片段，例如：{missing[0]}")
        voice_info = {"reused_tts_dir": str(tts_dir), "warning": warnings or None}

    gap = np.zeros(int(GAP_AFTER * TARGET_SR), dtype=np.float32)
    lead = np.zeros(int(LEAD_SILENCE * TARGET_SR), dtype=np.float32)
    fade = max(1, int(FADE_SECONDS * TARGET_SR))

    pieces: List[Any] = [lead]
    segments: List[Dict[str, Any]] = []
    per_speaker: Dict[str, List[Any]] = {s["id"]: [] for s in SPEAKERS}
    cursor = float(LEAD_SILENCE)

    for item in plan:
        audio, _ = read_wav_mono(item["path"], TARGET_SR)
        if audio.size == 0:
            continue
        if audio.size > 2 * fade:
            audio = audio.copy()
            audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        segments.append(
            {
                "speaker_id": item["speaker"],
                "start": round(cursor, 3),
                "end": round(cursor + audio.size / TARGET_SR, 3),
                "text": item["text"],
            }
        )
        pieces.append(audio)
        per_speaker[item["speaker"]].append(audio)
        pieces.append(gap)
        cursor += audio.size / TARGET_SR + GAP_AFTER

    pieces.append(np.zeros(int(TAIL_SILENCE * TARGET_SR), dtype=np.float32))
    timeline = np.concatenate(pieces)
    write_wav(out_dir / "discussion.wav", timeline, TARGET_SR)

    (out_dir / "discussion.transcript.json").write_text(
        json.dumps(
            {
                "source": (
                    "tools/make_fixtures.py 使用 Windows SAPI 合成；"
                    "时间轴由合成过程实测得到（非人工编造）"
                ),
                "language": "zh",
                "topic": DISCUSSION_TOPIC,
                "topic_keywords": DISCUSSION_KEYWORDS,
                "synthesized": True,
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for speaker_id, chunks in per_speaker.items():
        if chunks:
            write_wav(
                out_dir / "speaker_references" / f"{speaker_id}.wav",
                np.concatenate(chunks),
                TARGET_SR,
            )

    return {
        "duration": round(timeline.size / TARGET_SR, 2),
        "segments": len(segments),
        "speakers": len({s["speaker_id"] for s in segments}),
        "sample_rate": TARGET_SR,
        "voices": voice_info,
        "warnings": warnings,
    }


def build_video(
    out_dir: Path, duration: float, fps: float = 25.0
) -> Optional[Dict[str, Any]]:
    """生成真实视频文件（画面为几何图形，不含真人）。"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("提示：未安装 opencv，跳过视频夹具生成。", file=sys.stderr)
        return None

    width, height = 640, 480
    path = out_dir / "discussion.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        print("提示：视频写入器不可用，跳过视频夹具生成。", file=sys.stderr)
        return None

    total_frames = max(1, int(duration * fps))
    try:
        for frame_index in range(total_frames):
            t = frame_index / fps
            frame = np.full((height, width, 3), 245, dtype=np.uint8)
            for k in range(4):
                phase = t * (0.6 + 0.15 * k) + k * 1.3
                cx = int(width * (0.2 + 0.6 * (0.5 + 0.5 * math.sin(phase))))
                cy = int(height * (0.55 + 0.08 * math.cos(phase * 1.7)))
                cv2.circle(frame, (cx, cy), 34, (200, 120, 90), -1)
                cv2.putText(
                    frame, f"blob{k}", (cx - 28, cy - 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        writer.release()

    return {
        "fps": fps,
        "frames": total_frames,
        "width": width,
        "height": height,
        "note": "画面为几何图形，不含真人；仅用于验证视频 IO 与标注输出",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成真实媒体夹具（Windows 平台）")
    parser.add_argument("--out", default="fixtures", help="输出目录")
    parser.add_argument("--list-voices", action="store_true", help="列出可见的 SAPI 音色后退出")
    parser.add_argument("--no-video", action="store_true", help="跳过视频夹具")
    parser.add_argument(
        "--tts-dir",
        help="复用已合成的 utt_*.wav 目录，跳过语音合成（用于在合成环境受限时分两步执行）",
    )
    args = parser.parse_args()

    if args.list_voices:
        voices = list_voices()
        print(json.dumps(
            {"count": len(voices), "voices": [f"{n} ({c})" for n, c in voices]},
            ensure_ascii=False, indent=2,
        ))
        return 0

    if sys.platform != "win32":
        print("本脚本依赖 Windows SAPI 语音合成，仅支持 Windows。", file=sys.stderr)
        return 2

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tts_dir = Path(args.tts_dir).resolve() if args.tts_dir else None
    if tts_dir is not None:
        audio_info = build_audio(out_dir, tts_dir, tts_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="cla_fixtures_") as tmp:
            audio_info = build_audio(out_dir, Path(tmp))

    video_info = None if args.no_video else build_video(out_dir, audio_info["duration"])

    manifest = {
        "generated_by": "tools/make_fixtures.py",
        "is_synthetic_media": True,
        "audio": audio_info,
        "video": video_info,
        "limitations": [
            "合成语音不含远场混响、重叠说话与环境噪声，ASR/说话人分离表现会显著优于真实课堂。",
            "视频中不含真人，不能用于验证人体检测或视线估计精度。",
            "本夹具用于回归验证与冒烟测试，不可作为精度或效度证据。",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(out_dir), **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
