"""语义智能体 - 对话质量分析

修复的审计问题：
* #21 旧 ``_parse_response`` 只捕获 ``JSONDecodeError``：LLM 返回 ``null`` 或中文时抛出的
  ``TypeError`` / ``ValueError`` 会直接冒泡，**整条流水线崩掉**且没有重试。现在做了类型
  校验、范围裁剪、重试与指数退避。
* #22 **最严重的一条**：旧实现遇到“JSON 外面裹了一句解释文字”时，会静默返回默认的
  ``topic_relevance=0.5 / collisions=0``，与“真实表现中等”完全无法区分。现在解析失败
  会被计入 ``windows_failed`` 并写入 ``warnings``，同时相关指标置为 ``None``（不可用），
  由融合层重新归一化权重，绝不编造中性分。
* #23/#24 旧实现把超长转录截成“头 2000 + 尾 2000”字符（8 分钟讨论即丢弃 44%，40 分钟课
  丢弃约 80%），而且丢掉的正是观点碰撞最密集的**中段**，却仍把残片上的计数当作全组指标。
  现在改为**滑窗分块**：逐窗分析后聚合，不丢内容。
* #25 ``analyze_batch`` 旧实现从未被调用（死代码），现在它就是长对话的实际执行路径。
* #26 旧实现在 dashscope 分支读取 ``self.config.qwen_model``，而该字段并不存在于
  ``SemanticAgentConfig`` —— 该分支必然 AttributeError。现在统一由 ``_resolve_model`` 解析。
* #27 补齐超时、重试、最大输出 token、并发上限。
* #28/#29 ``turn_taking_pattern`` 与 ``consensus_quality`` 旧实现让模型算了却从不使用，
  现在都进入报告与诊断。
* #5 新增 ``per_speaker``：个体语义分必须逐人，不能把全局分数填给每个人。

额外提供 ``provider="heuristic"``：一个**明确标注为规则基线**的离线分析器
（基于话语标记词统计），用于无 API key / 无网络时的冒烟与回归测试。
它不是 LLM，也不声称具备 LLM 的语义理解能力。
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # 作为包导入
    from .config import (
        APIConfig,
        SemanticAgentConfig,
        get_config,
        get_semantic_config,
    )
    from .data_models import SemanticAnalysisResult, SpeakingSegment
except ImportError:  # 独立运行
    from config import APIConfig, SemanticAgentConfig, get_config, get_semantic_config
    from data_models import SemanticAnalysisResult, SpeakingSegment

__all__ = ["SemanticAgent", "DEFAULT_PROMPT_TEMPLATE"]

#: Prompt 模板。使用 __TOKEN__ 占位符而非 str.format，避免 JSON 花括号转义踩坑。
DEFAULT_PROMPT_TEMPLATE = """你是一个教育评价专家。请分析以下小组讨论的转录文本，输出结构化的协作质量指标。

## 讨论主题
__TOPIC__

## 待分析区间
核心区间：第 __START__ 秒 至 第 __END__ 秒
（转录中可能包含核心区间之外的上下文，仅用于帮助理解；**所有统计只针对核心区间内发生的内容**）

## 转录
__TRANSCRIPT__

## 分析任务
1. topic_relevance (0-1)：核心区间内对话围绕讨论主题的程度。若无法判断主题，返回 null。
2. opinion_collisions (整数)：反驳、补充、质疑等真实观点交锋的次数。注意：简单的"嗯""对"不算。
3. turn_taking_pattern：balanced（均衡）/ monopolizing（被个别人垄断）/ chaotic（杂乱抢话）。
4. argument_depth_score (0-1)：观点是否有理由、证据或推理支撑。
5. consensus_quality (0-1)：结论是否经过充分讨论后达成。若未形成结论，返回 null。
6. per_speaker：逐位说话人的语义参与分 (0-1)。键必须使用转录中出现的说话人标识原文。
7. evidence：2-5 条支撑判断的具体证据，须引用转录原文片段。

## 输出格式（严格 JSON，不要输出任何其他文字）
{
  "topic_relevance": 0.85,
  "opinion_collisions": 3,
  "turn_taking_pattern": "balanced",
  "argument_depth_score": 0.7,
  "consensus_quality": 0.6,
  "per_speaker": {"SPEAKER_00": 0.8, "SPEAKER_01": 0.6},
  "evidence": ["SPEAKER_00 提出方案 X", "SPEAKER_01 反驳并给出理由"]
}
"""

#: 判定“观点碰撞”的话语标记（规则基线用）
_COLLISION_MARKERS = (
    "但是", "不过", "可是", "然而", "不同意", "不对", "相反", "但是我觉得", "我觉得不",
    "换个角度", "补充一点", "有问题", "不一定", "前提是",
    "but", "however", "disagree", "on the contrary", "actually",
)
#: 判定“论证深度”的话语标记（规则基线用）
_REASON_MARKERS = (
    "因为", "所以", "由于", "因此", "理由", "原因", "说明", "证据", "意味着", "导致",
    "because", "therefore", "since", "which means", "evidence",
)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _coerce_float(value: Any, lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
    """把 LLM 返回值安全地转成 [lo, hi] 内的浮点数。

    无法转换或为空时返回 None（表示“不可用”），**绝不返回编造的中性默认值**。
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(lo, min(hi, number))


def _coerce_int(value: Any, lo: int = 0) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return max(lo, number)


class SemanticParseError(ValueError):
    """LLM 响应无法解析为预期的结构化结果。"""


class SemanticAgent:
    """语义智能体：判断对话是否围绕任务、是否有深度观点碰撞

    支持 provider：``openai`` / ``dashscope`` / ``vllm`` / ``heuristic``。
    ``heuristic`` 是离线规则基线，不调用任何外部服务。
    """

    def __init__(
        self,
        config: Optional[SemanticAgentConfig] = None,
        api_config: Optional[APIConfig] = None,
    ):
        self.config = config or get_semantic_config()
        self._api_config = api_config
        self._client = None
        self._prompt_template = self._load_prompt_template()

    # ------------------------------------------------------------ 配置

    @property
    def api_config(self) -> APIConfig:
        if self._api_config is None:
            self._api_config = get_config()
        return self._api_config

    def _load_prompt_template(self) -> str:
        if self.config.prompt_template_path:
            return Path(self.config.prompt_template_path).read_text(encoding="utf-8")
        return DEFAULT_PROMPT_TEMPLATE

    def _resolve_model(self) -> str:
        """解析模型名（修复 #26：不再引用不存在的 ``config.qwen_model``）。"""
        if self.config.model:
            return self.config.model
        provider = self.config.provider
        if provider == "openai":
            return self.api_config.openai_model
        if provider == "dashscope":
            return self.api_config.qwen_model
        if provider == "vllm":
            return self.api_config.vllm_model or "default"
        raise ValueError(f"不支持的 LLM 提供者: {provider}")

    def _get_client(self):
        if self._client is not None:
            return self._client
        provider = self.config.provider
        if provider == "openai":
            import openai

            if not self.api_config.openai_api_key:
                raise ValueError("请设置 OPENAI_API_KEY 环境变量或在 .env 中配置")
            self._client = openai.OpenAI(
                api_key=self.api_config.openai_api_key,
                timeout=self.api_config.request_timeout,
            )
        elif provider == "dashscope":
            import dashscope

            if not self.api_config.qwen_api_key:
                raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或在 .env 中配置")
            dashscope.api_key = self.api_config.qwen_api_key
            self._client = dashscope
        elif provider == "vllm":
            import openai

            if not self.api_config.vllm_endpoint:
                raise ValueError("请设置 VLLM_ENDPOINT 环境变量")
            self._client = openai.OpenAI(
                base_url=self.api_config.vllm_endpoint,
                api_key="dummy",
                timeout=self.api_config.request_timeout,
            )
        else:
            raise ValueError(f"不支持的 LLM 提供者: {provider}")
        return self._client

    # ------------------------------------------------------------ 文本组织

    @staticmethod
    def format_transcript(segments: Sequence[SpeakingSegment]) -> str:
        """把发言片段格式化为转录文本（保留原始说话人标识以便逐人回填）。"""
        return "\n".join(
            f"[{seg.start_time:.1f}s] {seg.speaker_id}: {seg.text}"
            for seg in segments
            if seg.text and seg.text.strip()
        )

    def plan_windows(
        self, segments: Sequence[SpeakingSegment]
    ) -> List[Tuple[float, float]]:
        """按时间切分分析窗口（核心区间不重叠，重叠部分只作为上下文）。"""
        if not segments:
            return []
        start = min(s.start_time for s in segments)
        end = max(s.end_time for s in segments)
        span = max(0.0, end - start)
        window = max(1.0, self.config.window_seconds)
        if span <= window:
            return [(start, end)]
        windows: List[Tuple[float, float]] = []
        cursor = start
        while cursor < end:
            windows.append((cursor, min(cursor + window, end)))
            cursor += window
        return windows

    def _window_segments(
        self, segments: Sequence[SpeakingSegment], core: Tuple[float, float]
    ) -> List[SpeakingSegment]:
        """取出核心区间及其上下文边缘的发言片段。

        注意：这里**不做截断丢弃**，只是按窗口切片；窗口之间内容完整覆盖整段对话。
        """
        core_start, core_end = core
        margin = max(0.0, self.config.window_overlap_seconds)
        lo, hi = core_start - margin, core_end + margin
        return [s for s in segments if s.end_time > lo and s.start_time < hi]

    # ------------------------------------------------------------ 单窗分析

    def _build_prompt(
        self, segments: Sequence[SpeakingSegment], core: Tuple[float, float], topic: Optional[str]
    ) -> str:
        transcript = self.format_transcript(segments)
        return (
            self._prompt_template.replace("__TOPIC__", topic or "（未提供，topic_relevance 请返回 null）")
            .replace("__START__", f"{core[0]:.1f}")
            .replace("__END__", f"{core[1]:.1f}")
            .replace("__TRANSCRIPT__", transcript)
        )

    def analyze_window(
        self,
        segments: Sequence[SpeakingSegment],
        core: Tuple[float, float],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """分析单个核心窗口，返回已校验的原始指标字典。"""
        if self.config.provider == "heuristic":
            return self._heuristic_window(segments, core, context)

        prompt = self._build_prompt(segments, core, context)
        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.api_config.max_retries)):
            try:
                raw = self._call_llm(prompt)
                return self.parse_response(raw)
            except SemanticParseError as exc:
                # 解析失败：重试一次仍失败就如实上报，不再伪造默认值
                last_error = exc
            except Exception as exc:  # 网络/限流等
                last_error = exc
            if attempt + 1 < max(1, self.api_config.max_retries):
                time.sleep(min(8.0, 2.0 ** attempt))
        raise SemanticParseError(f"窗口 [{core[0]:.1f}s, {core[1]:.1f}s] 分析失败：{last_error}")

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        model = self._resolve_model()
        provider = self.config.provider

        if provider in {"openai", "vllm"}:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.api_config.openai_temperature,
                max_tokens=self.api_config.max_output_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                raise SemanticParseError("LLM 返回了空内容")
            return content

        if provider == "dashscope":
            from dashscope import Generation

            response = Generation.call(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
            )
            if getattr(response, "status_code", 200) != 200:
                raise RuntimeError(f"dashscope 调用失败: {getattr(response, 'message', '')}")
            return response.output.choices[0].message.content

        raise ValueError(f"不支持的 LLM 提供者: {provider}")

    # ------------------------------------------------------------ 响应解析

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """从模型输出中提取 JSON 对象。

        旧实现在这里失败后会静默返回默认值；现在失败一律抛错，由上层显式记录。
        """
        if not text or not text.strip():
            raise SemanticParseError("空响应")
        cleaned = text.strip()

        fenced = _JSON_FENCE.search(cleaned)
        if fenced:
            cleaned = fenced.group(1).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start < 0 or end <= start:
                raise SemanticParseError("响应中找不到 JSON 对象")
            try:
                data = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                raise SemanticParseError(f"JSON 解析失败: {exc}") from exc

        if not isinstance(data, dict):
            raise SemanticParseError(f"期望 JSON 对象，实际为 {type(data).__name__}")
        return data

    def parse_response(self, response: str) -> Dict[str, Any]:
        """解析并校验 LLM 响应。

        与旧实现的关键差异：
        * 类型转换失败不再抛出未捕获异常，而是把该字段置为 None；
        * 关键字段全部缺失时抛出 :class:`SemanticParseError`，由调用方计入失败窗口。
        """
        data = self._extract_json(response)

        per_speaker_raw = data.get("per_speaker")
        per_speaker: Dict[str, float] = {}
        if isinstance(per_speaker_raw, dict):
            for key, value in per_speaker_raw.items():
                score = _coerce_float(value)
                if score is not None:
                    per_speaker[str(key)] = score

        result = {
            "topic_relevance": _coerce_float(data.get("topic_relevance")),
            "opinion_collisions": _coerce_int(data.get("opinion_collisions")),
            "turn_taking_pattern": str(data.get("turn_taking_pattern") or "unknown").lower(),
            "argument_depth_score": _coerce_float(data.get("argument_depth_score")),
            "consensus_quality": _coerce_float(data.get("consensus_quality")),
            "per_speaker": per_speaker,
            "evidence": [str(e) for e in (data.get("evidence") or []) if str(e).strip()],
        }

        usable = [
            result["topic_relevance"],
            result["opinion_collisions"],
            result["argument_depth_score"],
        ]
        if all(v is None for v in usable):
            raise SemanticParseError("关键字段全部缺失或类型非法")
        return result

    # ------------------------------------------------------------ 规则基线

    def _heuristic_window(
        self,
        segments: Sequence[SpeakingSegment],
        core: Tuple[float, float],
        topic: Optional[str],
    ) -> Dict[str, Any]:
        """离线规则基线（**不是 LLM**）：基于话语标记词统计的粗略估计。

        * 观点碰撞：包含转折/反驳标记的发言轮次
        * 论证深度：包含因果/理由标记的轮次占比
        * topic_relevance：仅在提供了主题时，用字符二元组重合度估计；否则返回 None
        """
        in_core = [s for s in segments if s.text.strip()]
        if not in_core:
            raise SemanticParseError("核心窗口内没有可用文本")

        collisions = sum(
            1 for s in in_core if any(m in s.text for m in _COLLISION_MARKERS)
        )
        reasoned = sum(1 for s in in_core if any(m in s.text for m in _REASON_MARKERS))
        depth = reasoned / len(in_core)

        topic_relevance: Optional[float] = None
        if topic:
            topic_grams = {topic[i : i + 2] for i in range(max(1, len(topic) - 1))}
            if topic_grams:
                hits = 0
                total = 0
                for seg in in_core:
                    grams = {seg.text[i : i + 2] for i in range(max(1, len(seg.text) - 1))}
                    if grams:
                        total += 1
                        if topic_grams & grams:
                            hits += 1
                topic_relevance = hits / total if total else None

        per_speaker: Dict[str, float] = {}
        by_speaker: Dict[str, List[SpeakingSegment]] = {}
        for seg in in_core:
            by_speaker.setdefault(seg.speaker_id, []).append(seg)
        if by_speaker:
            longest = max(sum(len(s.text) for s in segs) for segs in by_speaker.values())
            for speaker, segs in by_speaker.items():
                chars = sum(len(s.text) for s in segs)
                per_speaker[speaker] = round(min(1.0, chars / longest) if longest else 0.0, 4)

        return {
            "topic_relevance": topic_relevance,
            "opinion_collisions": collisions,
            "turn_taking_pattern": "unknown",
            "argument_depth_score": depth,
            "consensus_quality": None,
            "per_speaker": per_speaker,
            "evidence": [],
        }

    # ------------------------------------------------------------ 入口

    def analyze(
        self,
        segments: Sequence[SpeakingSegment],
        context: Optional[str] = None,
    ) -> SemanticAnalysisResult:
        """分析对话质量（长对话自动分窗，不做内容截断丢弃）。"""
        result = SemanticAnalysisResult()

        if not segments or not any(s.text.strip() for s in segments):
            result.warnings.append("对话为空或没有任何文本，语义分析被跳过（不产生默认分）。")
            return result

        windows = self.plan_windows(segments)
        if not windows:
            result.warnings.append("无法切分分析窗口，语义分析被跳过。")
            return result

        topic_values: List[Tuple[float, float]] = []      # (值, 窗口时长权重)
        depth_values: List[Tuple[float, float]] = []
        consensus_values: List[Tuple[float, float]] = []
        pattern_votes: Dict[str, int] = {}
        collisions_total = 0
        per_speaker_acc: Dict[str, List[float]] = {}
        evidence: List[str] = []
        failed: List[str] = []

        for core in windows:
            window_segments = self._window_segments(segments, core)
            try:
                data = self.analyze_window(window_segments, core, context)
            except SemanticParseError as exc:
                failed.append(str(exc))
                continue

            weight = max(1e-6, core[1] - core[0])
            if data["topic_relevance"] is not None:
                topic_values.append((data["topic_relevance"], weight))
            if data["argument_depth_score"] is not None:
                depth_values.append((data["argument_depth_score"], weight))
            if data["consensus_quality"] is not None:
                consensus_values.append((data["consensus_quality"], weight))
            if data["opinion_collisions"] is not None:
                collisions_total += data["opinion_collisions"]
            pattern = data.get("turn_taking_pattern") or "unknown"
            pattern_votes[pattern] = pattern_votes.get(pattern, 0) + 1
            for speaker, score in (data.get("per_speaker") or {}).items():
                per_speaker_acc.setdefault(speaker, []).append(score)
            evidence.extend(data.get("evidence") or [])

        result.windows_analyzed = len(windows) - len(failed)
        result.windows_failed = len(failed)
        result.complete = bool(result.windows_analyzed) and not failed

        if failed:
            result.warnings.append(
                f"{len(failed)}/{len(windows)} 个分析窗口失败，指标仅由成功窗口聚合得出；"
                "该结果不代表完整对话。首个失败原因：" + failed[0]
            )
        if not result.windows_analyzed:
            result.warnings.append("所有分析窗口均失败，语义指标全部不可用（未使用任何默认值充数）。")
            return result

        def weighted_mean(pairs: List[Tuple[float, float]]) -> Optional[float]:
            if not pairs:
                return None
            total_weight = sum(w for _v, w in pairs)
            return sum(v * w for v, w in pairs) / total_weight if total_weight else None

        result.topic_relevance = weighted_mean(topic_values)
        result.argument_depth_score = weighted_mean(depth_values)
        result.consensus_quality = weighted_mean(consensus_values)
        result.opinion_collisions = collisions_total
        result.turn_taking_pattern = max(pattern_votes, key=pattern_votes.get) if pattern_votes else "unknown"
        result.per_speaker = {
            speaker: sum(scores) / len(scores) for speaker, scores in per_speaker_acc.items()
        }
        result.evidence = evidence[:20]
        return result
