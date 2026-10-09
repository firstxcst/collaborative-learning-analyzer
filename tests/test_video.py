"""视觉模块测试。

重点：这些测试**全部不需要 YOLO / torch**，因为被验证的是几何与事件聚合逻辑本身。
真实的人体检测与视线估计精度无法在没有真实课堂素材的情况下验证 —— 这一点在
``fixtures/manifest.json`` 与 README 中都已明确声明，本文件不做任何精度声明。

覆盖的审计问题：#10 #11 #12 #13 #14 #15 #16 #19 #20
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from collaborative_learning_analyzer.data_models import GazeEvent, PointingEvent
from collaborative_learning_analyzer.video_agent import (
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_NOSE,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
    VideoAgent,
    _angle_between,
    _EventAccumulator,
)

HAS_CV2 = importlib.util.find_spec("cv2") is not None


@pytest.fixture
def agent() -> VideoAgent:
    return VideoAgent()


def keypoints(
    nose: tuple[float, float] | None = None,
    left_shoulder: tuple[float, float] | None = None,
    right_shoulder: tuple[float, float] | None = None,
    left_wrist: tuple[float, float] | None = None,
    right_wrist: tuple[float, float] | None = None,
    conf: float = 0.9,
) -> list:
    """构造 COCO-17 关键点数组（未指定的点置 None，模拟低置信度被丢弃）。"""
    points: list = [None] * 17
    for index, value in (
        (KP_NOSE, nose),
        (KP_LEFT_SHOULDER, left_shoulder),
        (KP_RIGHT_SHOULDER, right_shoulder),
        (KP_LEFT_WRIST, left_wrist),
        (KP_RIGHT_WRIST, right_wrist),
    ):
        if value is not None:
            points[index] = [value[0], value[1], conf]
    return points


def person_with_head_aimed_at(
    box: list[float], target: tuple[float, float]
) -> dict:
    """构造一个“头朝向指向 target”的人体几何（用真实几何函数生成，保证结构一致）。"""
    shoulder_mid = ((box[0] + box[2]) / 2, box[1] + 40)
    direction = (target[0] - shoulder_mid[0], target[1] - shoulder_mid[1])
    norm = math.hypot(*direction) or 1.0
    unit = (direction[0] / norm, direction[1] / norm)
    nose = (shoulder_mid[0] + unit[0] * 30, shoulder_mid[1] + unit[1] * 30)
    return VideoAgent._person_geometry(
        box,
        keypoints(
            nose=nose,
            left_shoulder=(shoulder_mid[0] - 25, shoulder_mid[1]),
            right_shoulder=(shoulder_mid[0] + 25, shoulder_mid[1]),
        ),
    )


# ---------------------------------------------------------------- 基础几何

def test_angle_between_basic():
    assert _angle_between((1, 0), (1, 0)) == pytest.approx(0.0)
    assert _angle_between((1, 0), (-1, 0)) == pytest.approx(180.0)
    assert _angle_between((1, 0), (0, 1)) == pytest.approx(90.0)


def test_angle_between_zero_vector_is_180():
    assert _angle_between((0, 0), (1, 0)) == 180.0


def test_person_geometry_extracts_head_vector():
    geometry = VideoAgent._person_geometry(
        [0, 0, 100, 200],
        keypoints(
            nose=(50, 60),
            left_shoulder=(25, 100),
            right_shoulder=(75, 100),
            left_wrist=(10, 150),
        ),
    )
    assert geometry["shoulder_mid"] == pytest.approx((50.0, 100.0))
    assert geometry["head_vector"] == pytest.approx((0.0, -40.0))
    assert geometry["center"] == pytest.approx((50.0, 100.0))
    assert geometry["diagonal"] == pytest.approx(math.hypot(100, 200))
    assert len(geometry["wrists"]) == 1


def test_person_geometry_handles_missing_keypoints():
    geometry = VideoAgent._person_geometry([0, 0, 10, 10], None)
    assert geometry["head_vector"] is None
    assert geometry["wrists"] == []


def test_person_geometry_ignores_dropped_low_confidence_points():
    """低置信度关键点在检测阶段已被置空，几何提取必须容忍 None。"""
    geometry = VideoAgent._person_geometry(
        [0, 0, 10, 10],
        keypoints(nose=(5, 5), left_shoulder=(2, 8), right_shoulder=None),
    )
    assert geometry["head_vector"] is None


# ---------------------------------------------------------- 事件聚合器

def test_event_accumulator_merges_contiguous_observations():
    acc = _EventAccumulator(gap_tolerance=1.5)
    for t in (0.0, 0.5, 1.0, 1.5, 2.0):
        acc.observe(("a", "b"), t, True)
    events = acc.finish()
    assert len(events) == 1
    source, target, start, end, _conf = events[0]
    assert (source, target) == ("a", "b")
    assert start == 0.0 and end == 2.0


def test_event_accumulator_splits_on_long_gap():
    acc = _EventAccumulator(gap_tolerance=1.0)
    for t in (0.0, 0.5, 1.0, 5.0, 5.5):
        acc.observe(("a", "b"), t, True)
    assert len(acc.finish()) == 2


def test_event_accumulator_closes_when_inactive():
    acc = _EventAccumulator(gap_tolerance=1.0)
    acc.observe(("a", "b"), 0.0, True)
    acc.observe(("a", "b"), 0.5, True)
    acc.observe(("a", "b"), 3.0, False)
    events = acc.finish()
    assert len(events) == 1
    assert events[0][3] == 0.5


def test_event_accumulator_filters_short_events():
    acc = _EventAccumulator(gap_tolerance=1.0)
    acc.observe(("a", "b"), 0.0, True)   # 单次观测 → 时长 0
    acc.observe(("c", "d"), 0.0, True)
    acc.observe(("c", "d"), 0.8, True)   # 连续 → 时长 0.8
    events = acc.finish(min_duration=0.4)
    assert [e[:2] for e in events] == [("c", "d")]


def test_event_accumulator_keeps_max_confidence():
    acc = _EventAccumulator()
    acc.observe(("a", "b"), 0.0, True, 0.2)
    acc.observe(("a", "b"), 0.5, True, 0.9)
    assert acc.finish()[0][4] == pytest.approx(0.9)


# ---------------------------------------------------------- 交互判定

def test_gaze_event_created_when_head_points_at_other_person(agent):
    """修复 #10/#11：旧实现里 _analyze_gaze 方法体为空，视线事件永远是空列表。"""
    target_box = [300, 0, 400, 200]
    source_box = [0, 0, 100, 200]
    target_center = (350.0, 100.0)
    people = {
        "person_1": person_with_head_aimed_at(source_box, target_center),
        "person_2": VideoAgent._person_geometry(target_box, None),
    }
    gaze_acc = _EventAccumulator()
    pointing_acc = _EventAccumulator()
    for t in (0.0, 0.5, 1.0):
        agent._observe_interactions(
            persons=people,
            object_boxes=[],
            timestamp=t,
            sample_dt=0.5,
            gaze_acc=gaze_acc,
            pointing_acc=pointing_acc,
            attention_to_materials={},
        )
    events = gaze_acc.finish()
    assert ("person_1", "person_2") in {e[:2] for e in events}


def test_no_gaze_event_when_head_points_away(agent):
    target_box = [300, 0, 400, 200]
    source_box = [0, 0, 100, 200]
    people = {
        # 头部朝向左上方（远离目标）
        "person_1": person_with_head_aimed_at(source_box, (-500.0, -500.0)),
        "person_2": VideoAgent._person_geometry(target_box, None),
    }
    gaze_acc = _EventAccumulator()
    agent._observe_interactions(
        persons=people, object_boxes=[], timestamp=0.0, sample_dt=0.5,
        gaze_acc=gaze_acc, pointing_acc=_EventAccumulator(), attention_to_materials={},
    )
    assert gaze_acc.finish() == []


def test_pointing_event_when_wrist_near_target(agent):
    """修复 #10：旧实现 _detect_pointing 方法体为空，指点事件永远是空列表。"""
    target_box = [100, 100, 200, 200]
    target_geo = VideoAgent._person_geometry(target_box, None)
    accumulator = _EventAccumulator()
    # 手腕落在目标框中心附近
    for t in (0.0, 0.5, 1.0, 1.5):
        agent._observe_pointing("person_1", "person_2", (150.0, 150.0, 0.9), target_geo, t, accumulator)
    events = accumulator.finish(min_duration=0.4)
    assert len(events) == 1
    assert events[0][0] == "person_1" and events[0][1] == "person_2"


def test_no_pointing_when_wrist_far_from_target(agent):
    target_geo = VideoAgent._person_geometry([100, 100, 200, 200], None)
    accumulator = _EventAccumulator()
    for t in (0.0, 0.5, 1.0):
        agent._observe_pointing("p1", "p2", (900.0, 900.0, 0.9), target_geo, t, accumulator)
    assert accumulator.finish(min_duration=0.4) == []


def test_pointing_requires_minimum_duration(agent):
    """连续不足最短时长的手部抖动不应该产生指点事件（修复 #13 抖动抑制）。"""
    target_geo = VideoAgent._person_geometry([100, 100, 200, 200], None)
    accumulator = _EventAccumulator()
    agent._observe_pointing("p1", "p2", (150.0, 150.0, 0.9), target_geo, 0.0, accumulator)
    assert accumulator.finish(min_duration=0.4) == []


def test_material_attention_accumulates_time(agent):
    box = [0, 0, 100, 200]
    material = {"label": "book", "box": [200, 100, 260, 160], "center": (230.0, 130.0),
                "diagonal": math.hypot(60, 60)}
    people = {"person_1": person_with_head_aimed_at(box, material["center"])}
    attention: dict[str, float] = {}
    for t in (0.0, 0.5, 1.0):
        agent._observe_interactions(
            persons=people, object_boxes=[material], timestamp=t, sample_dt=0.5,
            gaze_acc=_EventAccumulator(), pointing_acc=_EventAccumulator(),
            attention_to_materials=attention,
        )
    assert attention.get("person_1", 0.0) == pytest.approx(1.5)


# ---------------------------------------------------------- 汇总

def test_attention_map_is_populated(agent):
    gaze_events = [GazeEvent("p1", "p2", 0.0, 3.0), GazeEvent("p1", "p3", 4.0, 5.0)]
    mapping = agent._build_attention_map(gaze_events, {"p1": 2.0})
    assert set(mapping) == {"p1"}
    assert len(mapping["p1"]) == 3
    assert any(t.target_type == "material" for t in mapping["p1"])


def test_person_stats_aggregation(agent):
    gaze_events = [GazeEvent("p1", "p2", 0.0, 3.0), GazeEvent("p2", "p1", 1.0, 2.5)]
    pointing_events = [
        PointingEvent("p1", "material", 0.0, 1.0, 0.9),
        PointingEvent("p1", "person", 2.0, 2.5, 0.8),
    ]
    stats = agent._compute_person_stats(gaze_events, pointing_events, {"p1": 1.5})
    assert stats["p1"]["attention_to_others"] == pytest.approx(3.0)
    assert stats["p1"]["attention_received"] == pytest.approx(1.5)
    assert stats["p1"]["attention_to_materials"] == pytest.approx(1.5)
    assert stats["p1"]["pointing_frequency"] == 2
    assert stats["p1"]["pointing_to_material_frequency"] == 1


def test_cohesion_is_none_without_data(agent):
    """修复 #12：没有任何视觉数据时必须返回 None，而不是旧的硬编码 0.5。"""
    from collaborative_learning_analyzer.data_models import VideoAnalysisResult

    assert agent._compute_cohesion(VideoAnalysisResult(), 100.0) is None
    assert agent._compute_cohesion(VideoAnalysisResult(analyzed=True), 0.0) is None


def test_cohesion_is_ratio_of_attention_time(agent):
    from collaborative_learning_analyzer.data_models import VideoAnalysisResult

    result = VideoAnalysisResult(
        person_attention_stats={
            "p1": {"attention_to_others": 50.0},
            "p2": {"attention_to_others": 150.0},
        }
    )
    # 平均比例 = ((50/200) + min(1, 150/200)) / 2
    assert agent._compute_cohesion(result, 200.0) == pytest.approx((0.25 + 0.75) / 2)


# ---------------------------------------------------------- 视频 IO（需要 cv2）

@pytest.mark.skipif(not HAS_CV2, reason="需要安装 opencv（pip install 'collaborative-learning-analyzer[vision]'）")
def test_read_video_meta_reads_real_fps(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (64, 48))
    for _ in range(24):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    meta = VideoAgent._read_video_meta(str(path))
    assert meta["fps"] == pytest.approx(24.0, abs=0.5)
    assert meta["frame_count"] == 24
    assert (meta["width"], meta["height"]) == (64, 48)


@pytest.mark.skipif(not HAS_CV2, reason="需要安装 opencv")
def test_missing_video_raises_io_error():
    with pytest.raises(IOError):
        VideoAgent._read_video_meta("不存在的视频文件.mp4")


@pytest.mark.skipif(not HAS_CV2, reason="需要安装 opencv")
def test_annotated_video_writer_creates_file(tmp_path):
    """修复 #13：旧实现调用了一个从未定义的 _save_annotated_video。"""
    import cv2
    import numpy as np

    from collaborative_learning_analyzer.video_agent import _EventAccumulator  # noqa: F401

    source = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (64, 48))
    for _ in range(12):
        writer.write(np.full((48, 64, 3), 200, dtype=np.uint8))
    writer.release()

    out = tmp_path / "annotated.mp4"
    meta = VideoAgent._read_video_meta(str(source))
    writer2 = VideoAgent._open_writer(cv2, str(out), meta["fps"], meta)
    assert writer2 is not None
    writer2.release()
    assert out.is_file()


def test_cv2_is_not_imported_at_module_level():
    """修复 #19/#33：视觉依赖必须惰性导入，纯语音路径不应被 opencv 拖累。"""
    import inspect

    from collaborative_learning_analyzer import video_agent

    source = inspect.getsource(video_agent)
    top = source.split("class VideoAgent", 1)[0]
    assert "\nimport cv2" not in top
    assert "\nfrom ultralytics" not in top
