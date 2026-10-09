"""多模态融合推理引擎

计分模型（修复审计问题 #1-#9、#37 幽灵成员、#5 个体指标失真）

旧实现的致命缺陷是 ``均衡度 = 1 - 2 * std(贡献度)``：标准差度量的是**离散程度**，
不是**水平高低**，于是“所有成员都是 0 分”反而得到满分均衡度，实测全员沉默的小组
拿到 90.5 分（excellent），比正常均衡小组還高；同时由于每个成员的贡献度里都叠加了
同一批常数项（全局语义分、全局凝聚度），标准差被系统性稀释，均衡度恒偏向 1。

新模型采用**乘性活跃度门控**：

    activity  = min(1, 组内发言总时长 / (总时长 × 目标发言占比))
    evenness  = (1/HHI - 1) / (n - 1),   HHI = Σ p_i²,  p_i = t_i / Σt
    depth     = 0.5 × min(1, 每分钟观点碰撞 / 参考值) + 0.5 × 论证深度
    quality   = w_e·evenness + w_t·topic_relevance + w_d·depth
    health    = 100 × activity × quality

性质：
* 全员沉默 → activity = 0 → health = 0，可达 critical；
* 发言份额完全均等 → evenness = 1；一人独占 → evenness = 0；
* 健康分取值区间完全覆盖 0-100（由 ``test_score_range_is_fully_reachable`` 断言）；
* 缺失模态**不被编造**：可用维度重新归一化权重，并在报告中显式标注缺失。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

try:  # 作为包导入
    from .config import FusionEngineConfig, get_fusion_config
    from .data_models import (
        AudioAnalysisResult,
        CollaborationLevel,
        GroupCollaborationReport,
        IndividualContribution,
        SemanticAnalysisResult,
        VideoAnalysisResult,
    )
except ImportError:  # 独立运行
    from config import FusionEngineConfig, get_fusion_config
    from data_models import (
        AudioAnalysisResult,
        CollaborationLevel,
        GroupCollaborationReport,
        IndividualContribution,
        SemanticAnalysisResult,
        VideoAnalysisResult,
    )

__all__ = ["FusionEngine", "hhi_evenness"]

CONTRIBUTION_DIMS = ("speaking", "turn_taking", "semantic", "nonverbal")


def hhi_evenness(shares: List[float]) -> float:
    """基于 Herfindahl-Hirschman 指数的归一化均衡度。

    Args:
        shares: 各成员的份额，应非负；内部会归一化。

    Returns:
        [0, 1]。份额完全均等返回 1；全部份额集中在一人返回 0；
        份额总和为 0（例如全员沉默）返回 0 —— 这一点是关键修复，
        旧实现在这种情况返回 1.0（满分）。
    """
    total = sum(shares)
    if total <= 0 or not shares:
        return 0.0
    n = len(shares)
    if n == 1:
        return 1.0
    probs = [max(0.0, s) / total for s in shares]
    hhi = sum(p * p for p in probs)
    # HHI 取值范围 [1/n, 1]，线性映射到均衡度 [1, 0]
    return max(0.0, min(1.0, (1.0 / hhi - 1.0) / (n - 1)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _dedupe(items: List[str]) -> List[str]:
    """按出现顺序去重，避免同一个降级原因在各阶段被重复上报。"""
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


class FusionEngine:
    """多模态融合推理引擎"""

    def __init__(self, config: Optional[FusionEngineConfig] = None):
        self.config = config or get_fusion_config()
        self.config.validate()

    # ------------------------------------------------------------------ 入口

    def fuse(
        self,
        audio_result: AudioAnalysisResult,
        video_result: Optional[VideoAnalysisResult] = None,
        semantic_result: Optional[SemanticAnalysisResult] = None,
        group_id: str = "group_1",
        member_ids: Optional[List[str]] = None,
        member_mapping: Optional[Dict[str, str]] = None,
    ) -> GroupCollaborationReport:
        """融合多模态分析结果，生成协作报告。

        Args:
            audio_result: 语音分析结果（**唯一**的成员名册来源之一）
            video_result: 视觉分析结果，可为 None
            semantic_result: 语义分析结果，可为 None
            group_id: 小组标识
            member_ids: 显式成员名册；缺省时从语音结果推断
            member_mapping: 视觉 ID → 成员 ID 的对齐表。仅有视觉 ID 而无对齐时，
                不会把视觉 ID 当作成员（修复 #37 幽灵成员问题）。

        Returns:
            GroupCollaborationReport
        """
        video_result = video_result or VideoAnalysisResult()
        semantic_result = semantic_result or SemanticAnalysisResult()

        warnings: List[str] = []
        completeness: Dict[str, bool] = {}

        members, roster_warnings = self._resolve_members(
            audio_result, video_result, member_ids, member_mapping
        )
        warnings.extend(roster_warnings)
        warnings.extend(audio_result.warnings)

        # ---- 语音维度 ----
        audio_ok = bool(audio_result.speaker_stats)
        completeness["audio"] = audio_ok
        if not audio_ok:
            warnings.append("语音分析未产出任何说话人统计，个体参与度无法计算。")
        if audio_result.diarization_error:
            warnings.append(
                f"说话人分离失败并已降级为纯 ASR：{audio_result.diarization_error}"
                "（成员区分不可靠）"
            )

        # ---- 视觉维度 ----
        video_ok = bool(video_result.analyzed)
        completeness["video"] = video_ok
        warnings.extend(video_result.warnings)
        if not video_ok:
            warnings.append(
                "视觉分析未产出数据，非语言参与度维度将不参与计分（不会被 0.5 之类的默认值填充）。"
            )
        if video_result.error and not any("视觉" in w for w in video_result.warnings):
            # 避免与 video_result.warnings 里已有的同一原因重复上报
            warnings.append(f"视觉分析报错：{video_result.error}")

        # ---- 语义维度 ----
        # 修复 #22：只有真正解析成功的窗口才算“有语义数据”，
        # 解析失败不会再产生一个看起来正常的 0.5 分。
        semantic_usable = semantic_result.windows_analyzed > 0
        completeness["semantic"] = semantic_usable and semantic_result.complete
        warnings.extend(semantic_result.warnings)
        if not semantic_usable:
            warnings.append("语义分析未产出任何可用窗口，主题相关度与交互深度将不参与计分。")
        elif not semantic_result.complete:
            warnings.append(
                f"语义分析覆盖率不完整（成功 {semantic_result.windows_analyzed} 个 / "
                f"失败 {semantic_result.windows_failed} 个窗口），指标仅由成功窗口聚合。"
            )

        # ---- 组成员指标 ----
        shares = self._speaking_shares(members, audio_result)
        activity = self._participation_activity(audio_result)
        evenness = hhi_evenness(shares)
        # 语义不可用时不得计算交互深度：空 SemanticAnalysisResult 的
        # opinion_collisions 默认是 0（合法计数值），若不加这道门，
        # 会得到一个“0 分”的深度并混进质量分 —— 那正是我们要消灭的编造值。
        depth = (
            self._interaction_depth(semantic_result, audio_result.total_duration)
            if semantic_usable
            else None
        )

        components: Dict[str, float] = {}
        if audio_ok:
            components["evenness"] = evenness
        if semantic_usable and semantic_result.topic_relevance is not None:
            components["topic_relevance"] = _clamp01(semantic_result.topic_relevance)
        if depth is not None:
            components["interaction_depth"] = depth

        quality, used_components = self._quality_score(components)
        if not used_components:
            warnings.append("没有任何维度可用，质量分置 0；该报告不具备评价意义。")

        health = self._health_score(activity, quality)

        # ---- 个体贡献度 ----
        contributions = self._compute_contributions(
            members=members,
            shares=shares,
            audio_result=audio_result,
            video_result=video_result,
            semantic_result=semantic_result,
            member_mapping=member_mapping or {},
            video_ok=video_ok,
            semantic_usable=semantic_usable,
        )

        report = GroupCollaborationReport(
            group_id=group_id,
            total_duration=audio_result.total_duration,
            member_ids=members,
            audio_result=audio_result,
            video_result=video_result,
            semantic_result=semantic_result,
            individual_contributions=contributions,
            participation_activity=activity,
            evenness=evenness,
            topic_relevance=components.get("topic_relevance"),
            interaction_depth=depth,
            turn_taking_pattern=semantic_result.turn_taking_pattern if semantic_usable else "unknown",
            consensus_quality=semantic_result.consensus_quality if semantic_usable else None,
            quality_score=quality,
            overall_health_score=health,
            health_level=CollaborationLevel.from_score(
                health, self.config.level_thresholds
            ),
            data_completeness=completeness,
            warnings=_dedupe(warnings),
        )

        report.diagnoses, report.suggestions = self._generate_diagnoses(report)
        return report

    # -------------------------------------------------------------- 成员名册

    def _resolve_members(
        self,
        audio_result: AudioAnalysisResult,
        video_result: VideoAnalysisResult,
        member_ids: Optional[List[str]],
        member_mapping: Optional[Dict[str, str]],
    ) -> Tuple[List[str], List[str]]:
        """确定小组成员名册。

        修复 #37：旧实现把语音的 ``unknown`` 和视觉的 ``person_1`` 直接并集，
        凭空造出“全程未发言”的幽灵成员，而真正发言的人被说成“独占发言”。
        新规则：
        1. 显式传入的 ``member_ids`` 优先；
        2. 否则以**语音结果**为准（会说话的人才是小组成员）；
        3. 视觉 ID 只有在给出 ``member_mapping`` 时才会被认作成员。
        """
        warnings: List[str] = []

        if member_ids:
            return list(dict.fromkeys(member_ids)), warnings

        members = list(audio_result.speaker_stats.keys())
        if members:
            if "unknown" in members:
                warnings.append(
                    "语音结果中所有发言被归入 unknown（未做说话人分离），"
                    "个体参与度将退化为全组单一数值。"
                )
            return members, warnings

        # 无语音数据时，才退到视觉 ID
        members = list(video_result.person_attention_stats.keys())
        if members:
            warnings.append("无语音数据，成员名册回退到视觉追踪 ID，无法得知谁说了什么。")
        else:
            warnings.append("既无语音也无视觉数据，无法确定成员名册。")
        return members, warnings

    # -------------------------------------------------------------- 指标计算

    def _speaking_shares(
        self, members: List[str], audio_result: AudioAnalysisResult
    ) -> List[float]:
        return [
            float(audio_result.speaker_stats.get(m, {}).get("total_duration", 0.0) or 0.0)
            for m in members
        ]

    def _participation_activity(self, audio_result: AudioAnalysisResult) -> float:
        """活跃度：组内实际发言时长 / 目标发言量。

        全员沉默 → 0；达到目标发言占比 → 1。
        """
        duration = audio_result.total_duration
        if duration <= 0:
            return 0.0
        speaking = audio_result.total_speaking_time
        if speaking <= 0:
            speaking = sum(
                float(s.get("total_duration", 0.0) or 0.0)
                for s in audio_result.speaker_stats.values()
            )
        target = duration * self.config.target_talk_ratio
        if target <= 0:
            return 0.0
        return _clamp01(speaking / target)

    def _interaction_depth(
        self, semantic_result: SemanticAnalysisResult, duration: float
    ) -> Optional[float]:
        """交互深度 = 观点碰撞频率 + 论证深度 + 共识质量。

        三个子项各自可能不可用（LLM 未给出或解析失败）。不可用的子项被排除、
        其余子项重新归一化，而不是补 0 分；全部不可用时返回 ``None``，
        由上层把该维度整体排除在计分之外。

        这里同时修复了 #28/#29：``turn_taking_pattern`` 与 ``consensus_quality``
        过去让模型算了却从不使用。
        """
        subs: Dict[str, float] = {}
        minutes = duration / 60.0
        ref = self.config.collision_reference_per_minute
        if minutes > 0 and ref > 0:
            subs["collision"] = _clamp01((semantic_result.opinion_collisions / minutes) / ref)
        if semantic_result.argument_depth_score is not None:
            subs["argument"] = _clamp01(semantic_result.argument_depth_score)
        if semantic_result.consensus_quality is not None:
            subs["consensus"] = _clamp01(semantic_result.consensus_quality)
        if not subs:
            return None
        weight_sum = sum(self.config.depth_weights[k] for k in subs)
        if weight_sum <= 0:
            return None
        return _clamp01(
            sum(self.config.depth_weights[k] / weight_sum * v for k, v in subs.items())
        )

    def _quality_score(self, components: Dict[str, float]) -> Tuple[float, Set[str]]:
        """加权质量分。缺失维度**重新归一化**权重，而不是用默认值填充。"""
        used = {k for k in components if k in self.config.quality_weights}
        if not used:
            return 0.0, set()
        weight_sum = sum(self.config.quality_weights[k] for k in used)
        if weight_sum <= 0:
            return 0.0, set()
        score = sum(
            self.config.quality_weights[k] / weight_sum * _clamp01(components[k])
            for k in used
        )
        return _clamp01(score), used

    def _health_score(self, activity: float, quality: float) -> float:
        raw = 100.0 * _clamp01(activity) * _clamp01(quality)
        return max(self.config.score_min, min(self.config.score_max, raw))

    # -------------------------------------------------------------- 个体贡献

    def _compute_contributions(
        self,
        members: List[str],
        shares: List[float],
        audio_result: AudioAnalysisResult,
        video_result: VideoAnalysisResult,
        semantic_result: SemanticAnalysisResult,
        member_mapping: Dict[str, str],
        video_ok: bool,
        semantic_usable: bool,
    ) -> List[IndividualContribution]:
        n = max(1, len(members))
        total_speaking = sum(shares)
        equal_share = 1.0 / n
        total_turns = sum(
            int(audio_result.speaker_stats.get(m, {}).get("turns", 0) or 0) for m in members
        )
        expected_turns = total_turns / n if total_turns else 0.0

        # 时长（分钟），用于把“关注他人秒数”“指点次数”换算成每分钟速率
        minutes = (audio_result.total_duration or 0.0) / 60.0

        contributions: List[IndividualContribution] = []
        for member, seconds in zip(members, shares):
            stats = audio_result.speaker_stats.get(member, {})
            turns = int(stats.get("turns", 0) or 0)
            contrib = IndividualContribution(
                person_id=member,
                speaking_seconds=seconds,
                speaking_share=(seconds / total_speaking) if total_speaking > 0 else 0.0,
                turns=turns,
                longest_monologue=float(stats.get("longest_monologue", 0.0) or 0.0),
            )

            available: Set[str] = set()

            # 1) 发言（相对均等份额归一化）
            if total_speaking > 0:
                contrib.speaking_score = _clamp01((seconds / total_speaking) / equal_share)
                available.add("speaking")
            else:
                contrib.speaking_score = 0.0

            # 2) 轮流发言 = 发言轮次充足度(60%) + 轮次衔接顺畅度(40%)
            #    顺畅度由语音智能体计算（修复：旧实现的 turn_taking_score 算了但从没用过）
            if expected_turns > 0:
                turn_share = _clamp01(turns / expected_turns)
                smoothness = float(stats.get("turn_taking_score", 0.5) or 0.5)
                contrib.turn_taking_score = 0.6 * turn_share + 0.4 * _clamp01(smoothness)
                available.add("turn_taking")

            # 3) 语义（**逐人**，不再用全局分数冒充）
            if semantic_usable and member in semantic_result.per_speaker:
                contrib.semantic_score = _clamp01(semantic_result.per_speaker[member])
                available.add("semantic")

            # 4) 非语言（必须完成 ID 对齐，否则不归因到人）
            video_person_id = None
            if member_mapping:
                reverse = {v: k for k, v in member_mapping.items()}
                video_person_id = reverse.get(member)
            if video_ok and video_person_id:
                pstats = video_result.person_attention_stats.get(video_person_id)
                if pstats:
                    contrib.nonverbal_score = self._nonverbal_score(pstats, minutes)
                    available.add("nonverbal")

            contrib.calculate_total(self._renormalized_weights(available))
            contrib.diagnosis = self._diagnose_individual(
                contrib, stats, available_extra=available
            )
            contributions.append(contrib)

        return contributions

    def _nonverbal_score(self, pstats: Dict[str, Any], minutes: float) -> float:
        """把“关注他人时长”和“指点次数”换算成每分钟速率后归一化到 0-1。"""
        per_minute = 1.0 / minutes if minutes > 0 else 0.0
        attention_rate = float(pstats.get("attention_to_others", 0.0) or 0.0) * per_minute
        pointing_rate = float(pstats.get("pointing_frequency", 0.0) or 0.0) * per_minute

        attn_score = _clamp01(attention_rate / self.config.attention_reference_per_minute)
        point_score = _clamp01(pointing_rate / self.config.pointing_reference_per_minute)
        return 0.7 * attn_score + 0.3 * point_score

    def _renormalized_weights(self, available: Set[str]) -> Dict[str, float]:
        """在可用维度上重新归一化权重（缺失维度不参与，也不补 0 分）。"""
        used = [d for d in CONTRIBUTION_DIMS if d in available]
        if not used:
            # 无任何可用维度：平均分配，结果必然为 0 分，但保证权重合法
            return {d: 1.0 / len(CONTRIBUTION_DIMS) for d in CONTRIBUTION_DIMS}
        total = sum(self.config.contribution_weights[d] for d in used)
        return {d: self.config.contribution_weights[d] / total for d in used}

    # -------------------------------------------------------------- 诊断

    def _diagnose_individual(
        self,
        contrib: IndividualContribution,
        speaker_stats: Dict[str, Any],
        available_extra: Set[str],
    ) -> Dict[str, Any]:
        """生成个人诊断。"""
        diagnosis: Dict[str, Any] = {"strengths": [], "weaknesses": [], "details": {}}
        turns = contrib.turns
        longest = contrib.longest_monologue

        if contrib.speaking_seconds <= 0:
            diagnosis["weaknesses"].append("全程未发言")
        elif turns == 0:
            diagnosis["weaknesses"].append("有发言时长但未记录到独立发言轮次")
        else:
            diagnosis["strengths"].append(f"参与讨论（{turns} 次发言）")

        if longest > 30:
            diagnosis["weaknesses"].append(f"存在长时间独占发言（{longest:.1f}秒）")
        if contrib.speaking_share > 0.6:
            diagnosis["weaknesses"].append(
                f"发言份额达 {contrib.speaking_share:.0%}，显著高于均等份额"
            )
        if "nonverbal" not in available_extra:
            diagnosis["details"]["nonverbal_unavailable"] = True
        if "semantic" not in available_extra:
            diagnosis["details"]["semantic_unavailable"] = True

        diagnosis["details"].update(
            {
                "speaking_seconds": round(contrib.speaking_seconds, 2),
                "speaking_share": round(contrib.speaking_share, 4),
                "speaking_turns": turns,
                "longest_monologue": longest,
                "total_score": contrib.total_score,
                "scored_dimensions": sorted(available_extra),
            }
        )
        return diagnosis

    def _generate_diagnoses(self, report: GroupCollaborationReport) -> Tuple[List[str], List[str]]:
        diagnoses: List[str] = []
        suggestions: List[str] = []

        level_text = {
            CollaborationLevel.EXCELLENT: "小组协作表现优秀，成员参与均衡，观点交流充分",
            CollaborationLevel.GOOD: "小组协作良好，仍有改进空间",
            CollaborationLevel.FAIR: "小组协作一般，存在需要关注的问题",
            CollaborationLevel.POOR: "小组协作较差，需要教师干预",
            CollaborationLevel.CRITICAL: "小组协作严重不足，需要立即干预",
        }
        diagnoses.append(
            f"[{report.health_level.value}] {level_text[report.health_level]}"
            f"（健康分 {report.overall_health_score:.1f}）"
        )

        if report.participation_activity < 0.5:
            diagnoses.append(
                f"组内发言量偏低（活跃度 {report.participation_activity:.0%}），"
                "讨论时间未被充分利用"
            )
            suggestions.append("建议：明确讨论时间节点，或采用限时轮流发言推动参与")

        if report.evenness < 0.5:
            diagnoses.append(f"发言份额不均衡（均衡度 {report.evenness:.2f}）")
            suggestions.append("建议：鼓励沉默成员发言，或采用轮流发言机制")

        if report.turn_taking_pattern == "monopolizing":
            diagnoses.append("轮流模式为「个别人垄断」，多数成员难以插话")
            suggestions.append("建议：设定单人连续发言时长上限")
        elif report.turn_taking_pattern == "chaotic":
            diagnoses.append("轮流模式为「杂乱抢话」，缺乏有序交流")
            suggestions.append("建议：建立发言顺序规则")

        if report.topic_relevance is not None and report.topic_relevance < 0.6:
            diagnoses.append(f"讨论主题相关度偏低（{report.topic_relevance:.2f}）")
            suggestions.append("建议：引导学生回到讨论主题")

        if report.interaction_depth is not None and report.interaction_depth < 0.4:
            diagnoses.append(f"交互深度不足（{report.interaction_depth:.2f}）")
            suggestions.append("建议：提出启发性问题，激发观点碰撞")

        if report.consensus_quality is not None and report.consensus_quality < 0.5:
            diagnoses.append(f"共识质量偏低（{report.consensus_quality:.2f}），讨论未充分收敛")
            suggestions.append("建议：预留结论汇总环节，要求小组明确给出结论与依据")

        for contrib in report.individual_contributions:
            for weakness in contrib.diagnosis.get("weaknesses", []):
                diagnoses.append(f"{contrib.person_id}: {weakness}")
                if "全程未发言" in weakness:
                    suggestions.append(f"建议：关注 {contrib.person_id}，了解其不发言的原因")

        # 缺失模态必须写进结论，避免“看起来分析过了”
        missing = [k for k, v in report.data_completeness.items() if not v]
        if missing:
            diagnoses.append(f"以下模态未产出有效数据，未纳入计分：{', '.join(missing)}")

        return diagnoses, suggestions
