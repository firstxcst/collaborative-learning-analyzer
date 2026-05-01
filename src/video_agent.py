"""视觉智能体 - YOLO + Supervision 姿态/交互检测"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .config import get_video_config, VideoAgentConfig
from .data_models import (
    VideoAnalysisResult, 
    AttentionTarget,
    PointingEvent,
    GazeEvent
)


class VideoAgent:
    """
    视觉智能体：分析非语言交互
    
    核心能力：
    1. 人员检测与追踪（YOLO + ByteTrack）
    2. 姿态估计（YOLO-pose）
    3. 身体朝向分析
    4. 材料指点检测
    5. 视线关注推断
    """
    
    def __init__(self, config: Optional[VideoAgentConfig] = None):
        self.config = config or get_video_config()
        self._yolo_model = None
        self._pose_model = None
        
    def _load_yolo(self):
        """懒加载 YOLO 模型"""
        if self._yolo_model is None:
            from ultralytics import YOLO
            self._yolo_model = YOLO(self.config.yolo_model)
        return self._yolo_model
    
    def _load_pose(self):
        """懒加载姿态估计模型"""
        if self._pose_model is None:
            from ultralytics import YOLO
            self._pose_model = YOLO(self.config.pose_model)
        return self._pose_model
    
    def detect_and_track(
        self, 
        video_path: str,
        save_annotated: bool = False,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        检测并追踪视频中的目标
        
        Args:
            video_path: 视频文件路径
            save_annotated: 是否保存标注视频
            output_path: 输出视频路径
            
        Returns:
            Dict: 检测结果，包含每帧的检测框和追踪 ID
        """
        model = self._load_yolo()
        
        # 运行检测 + 追踪
        results = model.track(
            video_path,
            conf=self.config.yolo_conf_threshold,
            iou=self.config.yolo_iou_threshold,
            tracker=f"trackers/{self.config.tracker}.yaml",
            persist=True,
            verbose=False
        )
        
        # 解析结果
        frames_data = []
        for i, result in enumerate(results):
            frame_data = {
                "frame_id": i,
                "timestamp": i / 30.0,  # 假设 30fps
                "boxes": [],
                "track_ids": [],
            }
            
            if result.boxes is not None:
                for box in result.boxes:
                    frame_data["boxes"].append(box.xyxy[0].tolist())
                    if box.id is not None:
                        frame_data["track_ids"].append(int(box.id[0]))
            
            frames_data.append(frame_data)
        
        # 保存标注视频
        if save_annotated and output_path:
            import supervision as sv
            # 使用 supervision 进行可视化
            self._save_annotated_video(video_path, results, output_path)
        
        return {"frames": frames_data, "total_frames": len(frames_data)}
    
    def analyze_pose(self, video_path: str) -> List[Dict[str, Any]]:
        """
        姿态估计
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List: 每帧的姿态关键点
        """
        model = self._load_pose()
        
        results = model(video_path, verbose=False)
        
        poses = []
        for i, result in enumerate(results):
            frame_poses = {
                "frame_id": i,
                "timestamp": i / 30.0,
                "keypoints": [],
            }
            
            if result.keypoints is not None:
                for kps in result.keypoints:
                    # COCO 17 关键点格式
                    # 0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders,
                    # 7-8: elbows, 9-10: wrists, 11-12: hips, 13-14: knees, 15-16: ankles
                    keypoints_data = kps.data[0].tolist()  # [17, 3] (x, y, conf)
                    frame_poses["keypoints"].append(keypoints_data)
            
            poses.append(frame_poses)
        
        return poses
    
    def analyze(
        self,
        video_path: str,
        num_members: Optional[int] = None,
        save_annotated: bool = False,
        output_path: Optional[str] = None
    ) -> VideoAnalysisResult:
        """
        完整视觉分析
        
        Args:
            video_path: 视频文件路径
            num_members: 小组成员数量
            save_annotated: 是否保存标注视频
            output_path: 输出视频路径
            
        Returns:
            VideoAnalysisResult: 分析结果
        """
        # 检测 + 追踪
        detection_result = self.detect_and_track(
            video_path, 
            save_annotated, 
            output_path
        )
        
        # 姿态估计
        poses = self.analyze_pose(video_path)
        
        # 分析交互
        result = VideoAnalysisResult()
        
        # 计算注意力分布
        result.attention_map = self._compute_attention_map(detection_result, poses)
        
        # 检测指点事件
        result.pointing_events = self._detect_pointing(detection_result, poses)
        
        # 分析视线交互
        result.gaze_events = self._analyze_gaze(detection_result, poses)
        
        # 计算凝聚度
        result.cohesion_score = self._compute_cohesion(result)
        
        # 计算每人统计
        result.person_attention_stats = self._compute_person_stats(result)
        
        return result
    
    def _compute_attention_map(
        self, 
        detection_result: Dict, 
        poses: List[Dict]
    ) -> Dict[str, List[AttentionTarget]]:
        """计算每个人的注意力目标"""
        attention_map = {}
        
        frames = detection_result["frames"]
        
        # 简化版：基于位置推断注意力
        # 假设每个人关注最近的物体或人
        
        for frame_data in frames:
            timestamp = frame_data["timestamp"]
            track_ids = frame_data["track_ids"]
            boxes = frame_data["boxes"]
            
            # 为每个追踪的人分配注意力目标
            for i, track_id in enumerate(track_ids):
                person_id = f"person_{track_id}"
                
                if person_id not in attention_map:
                    attention_map[person_id] = []
                
                # 简化：假设关注画面中心或最近的物体
                # 实际应用中应结合姿态估计的头部朝向
                
        return attention_map
    
    def _detect_pointing(
        self, 
        detection_result: Dict,
        poses: List[Dict]
    ) -> List[PointingEvent]:
        """检测指点材料事件"""
        pointing_events = []
        
        # 简化版：检测手腕位置是否接近桌面物体
        # 实际应用中需要：
        # 1. 检测桌面区域
        # 2. 检测书本/纸张等物体
        # 3. 判断手部是否在物体上方且持续一定时间
        
        for pose_data in poses:
            keypoints = pose_data.get("keypoints", [])
            
            for i, kps in enumerate(keypoints):
                # kps: [17, 3] - COCO 关键点
                # 9, 10: 左右手腕
                if len(kps) >= 11:
                    left_wrist = kps[9]  # x, y, conf
                    right_wrist = kps[10]
                    
                    # 检测手腕是否在某个区域（简化）
                    # 实际需要结合物体检测
        
        return pointing_events
    
    def _analyze_gaze(
        self,
        detection_result: Dict,
        poses: List[Dict]
    ) -> List[GazeEvent]:
        """分析视线交互"""
        gaze_events = []
        
        # 简化版：基于头部朝向推断视线
        # 实际应用中需要：
        # 1. 提取头部朝向向量（从姿态关键点）
        # 2. 计算与其他人的相对位置
        # 3. 判断是否朝向某人
        
        for pose_data in poses:
            keypoints = pose_data.get("keypoints", [])
            
            for i, kps in enumerate(keypoints):
                if len(kps) >= 7:
                    # 使用鼻子和肩膀推断朝向
                    nose = kps[0]  # x, y, conf
                    left_shoulder = kps[5]
                    right_shoulder = kps[6]
                    
                    # 计算朝向向量
                    if nose[2] > self.config.pose_keypoint_threshold:
                        # 简化：使用肩膀中点到鼻子的向量
                        shoulder_mid = [
                            (left_shoulder[0] + right_shoulder[0]) / 2,
                            (left_shoulder[1] + right_shoulder[1]) / 2
                        ]
                        direction = [
                            nose[0] - shoulder_mid[0],
                            nose[1] - shoulder_mid[1]
                        ]
                        # 用 direction 判断朝向
        
        return gaze_events
    
    def _compute_cohesion(self, result: VideoAnalysisResult) -> float:
        """计算小组凝聚度"""
        # 简化版：基于视线交互的数量和持续时间
        # 凝聚度高 = 成员之间有更多相互关注
        
        total_gaze_duration = sum(e.duration for e in result.gaze_events)
        
        # 归一化到 0-1
        # 假设理想情况下每分钟有 30 秒的相互关注
        if total_gaze_duration > 0:
            cohesion = min(1.0, total_gaze_duration / 30.0)
        else:
            cohesion = 0.5  # 默认中等
        
        return cohesion
    
    def _compute_person_stats(self, result: VideoAnalysisResult) -> Dict[str, dict]:
        """计算每人的注意力统计"""
        stats = {}
        
        all_persons = set(result.attention_map.keys())
        
        for person_id in all_persons:
            attention_targets = result.attention_map.get(person_id, [])
            
            attention_to_others = 0.0
            attention_to_materials = 0.0
            
            for target in attention_targets:
                if target.target_type == "person":
                    attention_to_others += target.duration
                else:
                    attention_to_materials += target.duration
            
            # 计算指点频率
            pointing_count = sum(
                1 for e in result.pointing_events 
                if e.speaker_id == person_id
            )
            
            stats[person_id] = {
                "attention_to_others": attention_to_others,
                "attention_to_materials": attention_to_materials,
                "pointing_frequency": pointing_count,
            }
        
        return stats
