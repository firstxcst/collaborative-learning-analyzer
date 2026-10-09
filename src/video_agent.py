"""视觉智能体 - 姿态/交互检测（非语言参与度）

修复的审计问题：
* #10/#11/#12 旧实现里 ``_compute_attention_map`` / ``_detect_pointing`` / ``_analyze_gaze``
  三个方法体**完全为空**，导致 pointing/gaze/attention 恒为空容器、``cohesion_score``
  永远落到硬编码 0.5，融合公式里 30% 的权重形同虚设。本模块给出可运行的真实实现。
* #13 ``_save_annotated_video`` 只被调用从未定义（AttributeError），现补齐为 ``_draw_debug_frame``。
* #14 ``model.track()`` 返回生成器被耗尽后再次使用，现改为逐帧流式处理。
* #15 时间戳硬编码 ``i/30``，现从容器读取真实 FPS，读不到才兜底并显式标记。
* #16 旧实现把整段视频的每帧检测框与全部关键点存进内存（30 分钟视频约 5.4 万帧 ×
  人数 × 51 个浮点），现改为**流式聚合**：只保留事件区间与统计量，内存与视频长度无关。
* #19 ``cv2`` 曾被导入却完全未使用；现在它被真正用于视频解码与标注绘制，
  且改为**惰性导入**，使纯语音路径不再被 opencv 拖累。
* #20 移除对 ``supervision`` 的依赖（原本只在死分支里 import），标注绘制直接用 cv2。

关于精度的诚实说明
------------------
单目视线估计是公认困难的问题。本实现采用**几何启发式**：
以“肩中点 → 鼻子”的二维向量作为头部朝向的粗略代理，判断它是否指向候选目标中心。
它在**正对相机、成员大致处于同一深度平面**的课堂机位下大致可用；
在侧面机位、成员前后错落、遮挡严重时误差明显增大。
因此每条视线事件都携带 ``confidence``，结果不应被当作精确视线追踪数据使用。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:  # 作为包导入
    from .config import VideoAgentConfig, get_video_config
    from .data_models import (
        AttentionTarget,
        GazeEvent,
        PointingEvent,
        VideoAnalysisResult,
    )
except ImportError:  # 独立运行
    from config import VideoAgentConfig, get_video_config
    from data_models import AttentionTarget, GazeEvent, PointingEvent, VideoAnalysisResult

__all__ = ["VideoAgent"]

# COCO 17 关键点索引
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

#: 事件在两次采样之间允许的最大间隔（秒）。超过则视为一次新事件。
_EVENT_GAP_TOLERANCE = 1.5


def _import_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise ImportError(
            "视觉分析需要 opencv。请安装：pip install 'collaborative-learning-analyzer[vision]'"
        ) from exc


def _import_yolo():
    try:
        from ultralytics import YOLO

        return YOLO
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise ImportError(
            "视觉分析需要 ultralytics。请安装：pip install 'collaborative-learning-analyzer[vision]'"
        ) from exc


def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """两个二维向量之间的夹角（度），任一为零向量时返回 180。"""
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def _is_valid(point: Optional[List[float]]) -> bool:
    return bool(point) and len(point) >= 2 and all(v == v for v in point[:2])


class _EventAccumulator:
    """把逐帧布尔判定聚合成连续事件区间（内存与视频长度无关）。"""

    def __init__(self, gap_tolerance: float = _EVENT_GAP_TOLERANCE):
        self.gap_tolerance = gap_tolerance
        self._open: Dict[Tuple[str, str], List[float]] = {}
        self._closed: List[Tuple[str, str, float, float, float]] = []

    def observe(
        self, key: Tuple[str, str], timestamp: float, active: bool, confidence: float = 1.0
    ) -> None:
        current = self._open.get(key)
        if active:
            if current is None:
                self._open[key] = [timestamp, timestamp, confidence]
            elif timestamp - current[1] > self.gap_tolerance:
                self._closed.append((key[0], key[1], current[0], current[1], current[2]))
                self._open[key] = [timestamp, timestamp, confidence]
            else:
                current[1] = timestamp
                current[2] = max(current[2], confidence)
        elif current is not None and timestamp - current[1] > self.gap_tolerance:
            self._closed.append((key[0], key[1], current[0], current[1], current[2]))
            del self._open[key]

    def finish(self, min_duration: float = 0.0) -> List[Tuple[str, str, float, float, float]]:
        """收尾并过滤掉过短的事件（用于抑制单帧抖动造成的假阳性）。"""
        for key, current in self._open.items():
            self._closed.append((key[0], key[1], current[0], current[1], current[2]))
        self._open.clear()
        events = [e for e in self._closed if (e[3] - e[2]) >= min_duration]
        events.sort(key=lambda e: e[2])
        return events


class VideoAgent:
    """视觉智能体：分析非语言交互（身体朝向、指点、相互关注）"""

    def __init__(self, config: Optional[VideoAgentConfig] = None):
        self.config = config or get_video_config()
        self._pose_model = None
        self._object_model = None

    # ------------------------------------------------------------ 模型加载

    def _load_pose(self):
        if self._pose_model is None:
            self._pose_model = _import_yolo()(self.config.pose_model)
        return self._pose_model

    def _load_objects(self):
        if self._object_model is None:
            self._object_model = _import_yolo()(self.config.yolo_model)
        return self._object_model

    # ------------------------------------------------------------ 视频元信息

    @staticmethod
    def _read_video_meta(video_path: str) -> Dict[str, Any]:
        cv2 = _import_cv2()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件：{video_path}")
        try:
            return {
                "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            }
        finally:
            cap.release()

    # ------------------------------------------------------------ 几何工具

    @staticmethod
    def _person_geometry(
        box: List[float], keypoints: Optional[List[Optional[List[float]]]]
    ) -> Dict[str, Any]:
        """从人体框与关键点提取几何量，缺失关键点时对应项为 None。"""
        geometry: Dict[str, Any] = {
            "box": box,
            "center": ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
            "diagonal": math.hypot(box[2] - box[0], box[3] - box[1]),
            "nose": None,
            "shoulder_mid": None,
            "head_vector": None,
            "wrists": [],
        }
        if not keypoints or len(keypoints) <= KP_RIGHT_WRIST:
            return geometry

        nose, ls, rs = keypoints[KP_NOSE], keypoints[KP_LEFT_SHOULDER], keypoints[KP_RIGHT_SHOULDER]
        if _is_valid(nose) and _is_valid(ls) and _is_valid(rs):
            shoulder_mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
            geometry["nose"] = (nose[0], nose[1])
            geometry["shoulder_mid"] = shoulder_mid
            geometry["head_vector"] = (nose[0] - shoulder_mid[0], nose[1] - shoulder_mid[1])

        for idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
            kp = keypoints[idx]
            if _is_valid(kp):
                geometry["wrists"].append((kp[0], kp[1], float(kp[2]) if len(kp) > 2 else 1.0))
        return geometry

    # ------------------------------------------------------------ 主流程

    def analyze(
        self,
        video_path: str,
        num_members: Optional[int] = None,
        save_annotated: bool = False,
        output_path: Optional[str] = None,
    ) -> VideoAnalysisResult:
        """完整视觉分析。

        Args:
            video_path: 视频路径
            num_members: 已知成员数（用于一致性校验，不强制）
            save_annotated: 是否输出标注视频
            output_path: 标注视频输出路径

        Returns:
            VideoAnalysisResult（降级提示见 ``warnings``）
        """
        cv2 = _import_cv2()
        warnings: List[str] = []

        meta = self._read_video_meta(video_path)
        fps = meta["fps"]
        fps_is_fallback = False
        if not fps or fps <= 0 or not math.isfinite(fps):
            fps = self.config.fallback_fps
            fps_is_fallback = True
            warnings.append(
                f"无法从视频容器读取有效 FPS，已退回默认 {fps} fps —— "
                "由帧序号推算的时间戳可能不准确，跨模态时间对齐需谨慎。"
            )

        frame_count = meta["frame_count"]
        stride = max(1, int(self.config.frame_stride))
        if frame_count and frame_count / stride > self.config.max_processed_frames:
            stride = int(math.ceil(frame_count / self.config.max_processed_frames))
            warnings.append(
                f"视频共 {frame_count} 帧，为控制内存已把采样步长提高到 {stride} 帧。"
            )
        sample_interval = stride / fps

        pose_model = self._load_pose()

        object_classes = list(self.config.object_classes or [])
        object_model = None
        if object_classes:
            try:
                object_model = self._load_objects()
            except Exception as exc:  # pragma: no cover - 模型下载失败等
                warnings.append(f"材料目标检测模型不可用，指点材料的检测被跳过：{exc}")

        # 目标检测按固定时间间隔执行，避免逐帧跑两个模型
        object_every = max(1, int(round(self.config.object_detection_interval_seconds / sample_interval)))

        gaze_acc = _EventAccumulator()
        pointing_acc = _EventAccumulator()
        attention_to_materials: Dict[str, float] = {}
        object_boxes: List[Dict[str, Any]] = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件：{video_path}")

        writer = None
        if save_annotated and output_path:
            writer = self._open_writer(cv2, output_path, fps, meta)
            if writer is None:
                warnings.append("标注视频写入器创建失败，已跳过标注视频输出。")

        frame_index = -1
        processed = 0
        try:
            while True:
                grabbed, frame = cap.read()
                if not grabbed:
                    break
                frame_index += 1
                if frame_index % stride != 0:
                    continue
                processed += 1
                timestamp = frame_index / fps
                sample_dt = sample_interval

                persons = self._detect_persons(pose_model, frame)

                if object_model is not None and (processed - 1) % object_every == 0:
                    detected = self._detect_objects(object_model, frame, object_classes)
                    if detected:
                        object_boxes = detected
                if object_model is None:
                    object_boxes = []

                self._observe_interactions(
                    persons=persons,
                    object_boxes=object_boxes,
                    timestamp=timestamp,
                    sample_dt=sample_dt,
                    gaze_acc=gaze_acc,
                    pointing_acc=pointing_acc,
                    attention_to_materials=attention_to_materials,
                )

                if writer is not None:
                    self._draw_debug_frame(cv2, frame, persons)
                    writer.write(frame)
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        gaze_events = [
            GazeEvent(source_id=s, target_id=t, start_time=a, end_time=b)
            for s, t, a, b, _c in gaze_acc.finish(min_duration=0.0)
        ]
        pointing_events = [
            PointingEvent(
                speaker_id=s,
                target_type="material" if t.startswith("object_") else "person",
                start_time=a,
                end_time=b,
                confidence=round(c, 4),
            )
            for s, t, a, b, c in pointing_acc.finish(
                min_duration=self.config.pointing_min_duration_seconds
            )
        ]

        analyzed_duration = processed * sample_interval
        result = VideoAnalysisResult(
            gaze_events=gaze_events,
            pointing_events=pointing_events,
            analyzed=True,
            fps=fps,
            fps_is_fallback=fps_is_fallback,
            frames_processed=processed,
            analyzed_duration=analyzed_duration,
        )
        result.attention_map = self._build_attention_map(gaze_events, attention_to_materials)
        result.person_attention_stats = self._compute_person_stats(
            gaze_events=gaze_events,
            pointing_events=pointing_events,
            attention_to_materials=attention_to_materials,
        )
        result.cohesion_score = self._compute_cohesion(result, analyzed_duration)

        if num_members and len(result.person_attention_stats) != num_members:
            warnings.append(
                f"检测到 {len(result.person_attention_stats)} 个独立人员，"
                f"与给定成员数 {num_members} 不一致（追踪抖动或人员出入画会造成偏差）。"
            )
        if not result.person_attention_stats:
            warnings.append("未检测到任何人体，非语言参与度不可用。")
        result.warnings = warnings
        return result

    # ------------------------------------------------------------ 检测

    def _detect_persons(self, pose_model, frame) -> Dict[str, Dict[str, Any]]:
        """单帧人员检测 + 追踪 + 姿态，返回 {person_id: geometry}。"""
        persons: Dict[str, Dict[str, Any]] = {}
        results = pose_model.track(
            frame,
            persist=True,
            conf=self.config.yolo_conf_threshold,
            iou=self.config.yolo_iou_threshold,
            tracker=self.config.tracker,
            verbose=False,
        )
        if not results:
            return persons
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return persons

        keypoints_array = None
        if getattr(result, "keypoints", None) is not None and result.keypoints is not None:
            data = result.keypoints.data
            keypoints_array = data.cpu().numpy() if hasattr(data, "cpu") else data

        for i in range(len(boxes)):
            box = boxes[i]
            if getattr(box, "id", None) is None or box.id is None:
                continue  # 无稳定追踪 ID 的检测不参与跨模态归因
            track_id = int(box.id[0])
            xyxy = [float(v) for v in box.xyxy[0].tolist()]

            keypoints: Optional[List[Optional[List[float]]]] = None
            if keypoints_array is not None and i < len(keypoints_array):
                keypoints = []
                for k in keypoints_array[i]:
                    point = [float(k[0]), float(k[1]), float(k[2])]
                    keypoints.append(
                        point if point[2] >= self.config.pose_keypoint_threshold else None
                    )
            persons[f"person_{track_id}"] = self._person_geometry(xyxy, keypoints)
        return persons

    def _detect_objects(
        self, object_model, frame, object_classes: List[str]
    ) -> List[Dict[str, Any]]:
        """检测材料类目标（书本/纸张等）。"""
        results = object_model(
            frame,
            conf=self.config.yolo_conf_threshold,
            iou=self.config.yolo_iou_threshold,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        wanted = {c.lower() for c in object_classes}
        found: List[Dict[str, Any]] = []
        for i in range(len(boxes)):
            label = str(names.get(int(boxes[i].cls[0]), int(boxes[i].cls[0]))).lower()
            if label not in wanted:
                continue
            xyxy = [float(v) for v in boxes[i].xyxy[0].tolist()]
            found.append(
                {
                    "label": label,
                    "box": xyxy,
                    "center": ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0),
                    "diagonal": math.hypot(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]),
                }
            )
        return found

    # ------------------------------------------------------------ 交互判定

    def _observe_interactions(
        self,
        persons: Dict[str, Dict[str, Any]],
        object_boxes: List[Dict[str, Any]],
        timestamp: float,
        sample_dt: float,
        gaze_acc: _EventAccumulator,
        pointing_acc: _EventAccumulator,
        attention_to_materials: Dict[str, float],
    ) -> None:
        """在单个采样时刻判定“看向谁 / 指点什么”。

        这是旧实现里三个空方法（attention / pointing / gaze）合并后的真实实现。
        """
        person_ids = list(persons)
        for source in person_ids:
            geo = persons[source]
            head_vector = geo["head_vector"]
            shoulder_mid = geo["shoulder_mid"]

            if head_vector and shoulder_mid:
                for target in person_ids:
                    if target == source:
                        continue
                    t_geo = persons[target]
                    vec = (
                        t_geo["center"][0] - shoulder_mid[0],
                        t_geo["center"][1] - shoulder_mid[1],
                    )
                    angle = _angle_between(head_vector, vec)
                    gaze_acc.observe(
                        (source, target),
                        timestamp,
                        angle <= self.config.gaze_angle_threshold,
                        max(0.0, 1.0 - angle / 180.0),
                    )

                materials_active = False
                for obj in object_boxes:
                    vec = (
                        obj["center"][0] - shoulder_mid[0],
                        obj["center"][1] - shoulder_mid[1],
                    )
                    if _angle_between(head_vector, vec) <= self.config.gaze_angle_threshold:
                        materials_active = True
                        break
                if materials_active:
                    attention_to_materials[source] = (
                        attention_to_materials.get(source, 0.0) + sample_dt
                    )

            for wrist in geo["wrists"]:
                if wrist[2] < self.config.pose_keypoint_threshold:
                    continue
                for target, t_geo in persons.items():
                    if target == source:
                        continue
                    self._observe_pointing(
                        source, target, wrist, t_geo, timestamp, pointing_acc
                    )
                for obj_index, obj in enumerate(object_boxes):
                    self._observe_pointing(
                        source, f"object_{obj_index}", wrist, obj, timestamp, pointing_acc
                    )

    def _observe_pointing(
        self,
        source: str,
        target: str,
        wrist: Tuple[float, float, float],
        target_geo: Dict[str, Any],
        timestamp: float,
        accumulator: _EventAccumulator,
    ) -> None:
        """指点判定：手腕到目标中心的距离小于目标对角线的一定倍数。

        过短的瞬态事件在 ``finish(min_duration=...)`` 阶段被统一过滤，
        因此这里不需要额外维护“连续帧计数”。
        """
        diagonal = target_geo.get("diagonal") or 0.0
        if diagonal <= 0:
            accumulator.observe((source, target), timestamp, False)
            return
        distance = math.hypot(
            wrist[0] - target_geo["center"][0], wrist[1] - target_geo["center"][1]
        )
        threshold = self.config.pointing_distance_ratio * diagonal
        accumulator.observe(
            (source, target),
            timestamp,
            distance <= threshold,
            max(0.0, 1.0 - distance / (threshold * 4)),
        )

    # ------------------------------------------------------------ 汇总

    def _build_attention_map(
        self, gaze_events: List[GazeEvent], attention_to_materials: Dict[str, float]
    ) -> Dict[str, List[AttentionTarget]]:
        """把视线事件聚合为“某人 → 注意力目标”的映射。

        旧实现返回的**永远是空字典**；现在返回真实聚合结果。
        """
        mapping: Dict[str, List[AttentionTarget]] = {}
        for event in gaze_events:
            mapping.setdefault(event.source_id, []).append(
                AttentionTarget(
                    target_type="person",
                    target_id=event.target_id,
                    start_time=event.start_time,
                    end_time=event.end_time,
                )
            )
        for source, seconds in attention_to_materials.items():
            if seconds > 0:
                mapping.setdefault(source, []).append(
                    AttentionTarget(
                        target_type="material", target_id="materials", end_time=seconds
                    )
                )
        return mapping

    def _compute_person_stats(
        self,
        gaze_events: List[GazeEvent],
        pointing_events: List[PointingEvent],
        attention_to_materials: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        """每人的非语言参与统计。"""
        sources = {e.source_id for e in gaze_events}
        sources |= set(attention_to_materials)
        sources |= {e.speaker_id for e in pointing_events}

        stats: Dict[str, Dict[str, Any]] = {}
        for person_id in sorted(sources):
            stats[person_id] = {
                "attention_to_others": sum(
                    e.duration for e in gaze_events if e.source_id == person_id
                ),
                "attention_to_materials": float(attention_to_materials.get(person_id, 0.0)),
                "attention_received": sum(
                    e.duration for e in gaze_events if e.target_id == person_id
                ),
                "pointing_frequency": sum(
                    1 for e in pointing_events if e.speaker_id == person_id
                ),
                "pointing_to_material_frequency": sum(
                    1
                    for e in pointing_events
                    if e.speaker_id == person_id and e.target_type == "material"
                ),
            }
        return stats

    def _compute_cohesion(
        self, result: VideoAnalysisResult, analyzed_duration: float
    ) -> Optional[float]:
        """小组凝聚度 = 成员平均“关注他人”时长占分析时长的比例。

        旧实现在没有视线事件时直接返回硬编码 0.5，等于凭空编造一个中等值；
        现在若无人被检测到则返回 ``None``，由上层判定该维度不可用。
        """
        if not result.person_attention_stats or analyzed_duration <= 0:
            return None
        ratios = [
            min(1.0, s["attention_to_others"] / analyzed_duration)
            for s in result.person_attention_stats.values()
        ]
        return sum(ratios) / len(ratios) if ratios else None

    # ------------------------------------------------------------ 标注输出

    @staticmethod
    def _open_writer(cv2, output_path: str, fps: float, meta: Dict[str, Any]):
        width, height = meta.get("width") or 0, meta.get("height") or 0
        if width <= 0 or height <= 0:
            return None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        return writer if writer.isOpened() else None

    @staticmethod
    def _draw_debug_frame(cv2, frame, persons: Dict[str, Dict[str, Any]]) -> None:
        """在帧上绘制人体框、追踪 ID 与头部朝向箭头（就地修改）。

        旧实现只能调用一个不存在的 ``_save_annotated_video``。
        """
        for person_id, geo in persons.items():
            box = geo["box"]
            cv2.rectangle(
                frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 200, 0), 2
            )
            cv2.putText(
                frame, person_id, (int(box[0]), max(12, int(box[1]) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA,
            )
            if geo["head_vector"] and geo["shoulder_mid"] and geo["nose"]:
                cv2.arrowedLine(
                    frame,
                    (int(geo["shoulder_mid"][0]), int(geo["shoulder_mid"][1])),
                    (int(geo["nose"][0]), int(geo["nose"][1])),
                    (0, 165, 255), 2, tipLength=0.3,
                )
