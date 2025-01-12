"""Enhanced player tracking module"""
from .. import DEFAULT_CONFIG, logger
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import json
import os
from datetime import datetime

@dataclass
class PlayerTrack:
    """Track information for a single player"""
    player_id: int
    jersey_number: int
    jersey_color: Optional[str] = None
    first_seen: int = 0  # frame number
    last_seen: int = 0   # frame number
    total_frames: int = 0
    positions: List[Tuple[float, float, float, float]] = field(default_factory=list)  # List of (x1, y1, x2, y2)
    confidences: List[float] = field(default_factory=list)
    is_active: bool = True
    velocity: List[Tuple[float, float]] = field(default_factory=list)  # (dx, dy) between frames
    
    def add_position(self, bbox: Tuple[float, float, float, float], frame_number: int, confidence: float):
        """Add new position to track"""
        if not self.positions:
            self.first_seen = frame_number
        
        self.positions.append(bbox)
        self.confidences.append(confidence)
        
        # Calculate velocity if we have previous positions
        if len(self.positions) > 1:
            prev_x = (self.positions[-2][0] + self.positions[-2][2]) / 2
            prev_y = (self.positions[-2][1] + self.positions[-2][3]) / 2
            curr_x = (bbox[0] + bbox[2]) / 2
            curr_y = (bbox[1] + bbox[3]) / 2
            self.velocity.append((curr_x - prev_x, curr_y - prev_y))
        else:
            self.velocity.append((0, 0))
            
        self.last_seen = frame_number
        self.total_frames += 1
        
    def get_current_position(self) -> Optional[Tuple[float, float, float, float]]:
        """Get the most recent position"""
        return self.positions[-1] if self.positions else None
        
    def get_average_speed(self) -> float:
        """Calculate average movement speed in pixels per frame"""
        if not self.velocity:
            return 0.0
        speeds = [np.sqrt(dx*dx + dy*dy) for dx, dy in self.velocity]
        return sum(speeds) / len(speeds)
        
    def predict_next_position(self) -> Optional[Tuple[float, float, float, float]]:
        """Predict next position based on velocity"""
        if len(self.positions) < 2:
            return None
            
        last_pos = self.positions[-1]
        if not self.velocity:
            return last_pos
            
        avg_velocity = np.mean(self.velocity[-5:], axis=0) if len(self.velocity) >= 5 else self.velocity[-1]
        width = last_pos[2] - last_pos[0]
        height = last_pos[3] - last_pos[1]
        
        return (
            last_pos[0] + avg_velocity[0],
            last_pos[1] + avg_velocity[1],
            last_pos[0] + avg_velocity[0] + width,
            last_pos[1] + avg_velocity[1] + height
        )

class EnhancedPlayerTracker:
    def __init__(self,
                 max_track_age: int = DEFAULT_CONFIG['tracker']['max_track_age'],
                 min_detection_conf: float = DEFAULT_CONFIG['tracker']['min_detection_conf'],
                 iou_threshold: float = DEFAULT_CONFIG['tracker']['iou_threshold']):
        """Initialize tracker with configuration"""
        self.max_track_age = max_track_age
        self.min_detection_conf = min_detection_conf
        self.iou_threshold = iou_threshold
        
        self.tracks: Dict[int, PlayerTrack] = {}
        self.next_track_id = 0
        self.current_frame = 0
        
        # Team analysis
        self.team_positions: Dict[str, List[List[Tuple[float, float]]]] = {
            'home': [],
            'away': []
        }
        
        # Track analysis metrics
        self.metrics = {
            'total_tracks': 0,
            'active_tracks': 0,
            'total_detections': 0,
            'missed_detections': 0,
            'switches': 0,
            'processing_fps': 0
        }

    def update(self, frame: np.ndarray, detections: List[Dict], frame_number: int) -> Dict[int, PlayerTrack]:
        """Update tracks with new detections"""
        self.current_frame = frame_number
        active_tracks = self._get_active_tracks()
        
        # Match detections to existing tracks
        if active_tracks and detections:
            matches, unmatched_tracks, unmatched_detections = self._match_detections(
                active_tracks, detections
            )
        else:
            matches = []
            unmatched_tracks = list(active_tracks)
            unmatched_detections = list(range(len(detections)))
            
        # Update matched tracks
        for track_id, detection_idx in matches:
            self._update_track(track_id, detections[detection_idx])
            
        # Create new tracks
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            if detection['confidence'] >= self.min_detection_conf:
                self._create_track(detection)
                
        # Update metrics
        self.metrics['total_detections'] += len(detections)
        self.metrics['active_tracks'] = len(self._get_active_tracks())
                
        return {track_id: track for track_id, track in self.tracks.items() 
                if track.is_active}

    def _get_active_tracks(self) -> List[int]:
        """Get list of active track IDs"""
        active_tracks = []
        for track_id, track in list(self.tracks.items()):
            if self.current_frame - track.last_seen > self.max_track_age:
                track.is_active = False
            elif track.is_active:
                active_tracks.append(track_id)
        return active_tracks

    def _match_detections(self, track_ids: List[int], detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to tracks using IoU"""
        if not track_ids or not detections:
            return [], track_ids, list(range(len(detections)))
            
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            track_bbox = track.get_current_position()
            if track_bbox is None:
                continue
                
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track_bbox, detection['bbox'])
                
        # Find matches using Hungarian algorithm
        matches = []
        unmatched_tracks = set(range(len(track_ids)))
        unmatched_detections = set(range(len(detections)))
        
        # Greedy matching for simplicity
        while True:
            # Find highest IoU
            if len(unmatched_tracks) == 0 or len(unmatched_detections) == 0:
                break
                
            max_iou = 0
            best_match = None
            
            for i in unmatched_tracks:
                for j in unmatched_detections:
                    if iou_matrix[i, j] > max_iou and iou_matrix[i, j] >= self.iou_threshold:
                        max_iou = iou_matrix[i, j]
                        best_match = (i, j)
            
            if best_match is None:
                break
                
            i, j = best_match
            matches.append((track_ids[i], j))
            unmatched_tracks.remove(i)
            unmatched_detections.remove(j)
            
        unmatched_tracks = [track_ids[i] for i in unmatched_tracks]
        unmatched_detections = list(unmatched_detections)
        
        return matches, unmatched_tracks, unmatched_detections

    def _calculate_iou(self, bbox1: Tuple[float, float, float, float],
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = bbox1_area + bbox2_area - intersection
        
        return intersection / union if union > 0 else 0

    def _create_track(self, detection: Dict) -> int:
        """Create new track from detection"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        track = PlayerTrack(
            player_id=track_id,
            jersey_number=detection.get('jersey_number', -1),
            jersey_color=detection.get('jersey_color', None)
        )
        
        track.add_position(
            detection['bbox'],
            self.current_frame,
            detection['confidence']
        )
        
        self.tracks[track_id] = track
        self.metrics['total_tracks'] += 1
        
        return track_id

    def _update_track(self, track_id: int, detection: Dict):
        """Update existing track with new detection"""
        if track_id in self.tracks:
            self.tracks[track_id].add_position(
                detection['bbox'],
                self.current_frame,
                detection['confidence']
            )

    def get_player_stats(self) -> Dict:
        """Get comprehensive statistics for all tracked players"""
        stats = {}
        for track_id, track in self.tracks.items():
            if track.total_frames > 0:  # Only include tracks with data
                stats[track_id] = {
                    'jersey_number': track.jersey_number,
                    'jersey_color': track.jersey_color,
                    'first_seen': track.first_seen,
                    'last_seen': track.last_seen,
                    'total_frames': track.total_frames,
                    'is_active': track.is_active,
                    'average_confidence': np.mean(track.confidences),
                    'average_speed': track.get_average_speed()
                }
        return stats

    def save_tracking_data(self, output_path: str):
        """Save tracking results to file"""
        data = {
            'tracks': {
                track_id: {
                    'jersey_number': track.jersey_number,
                    'jersey_color': track.jersey_color,
                    'positions': [[float(x) for x in pos] for pos in track.positions],
                    'confidences': [float(c) for c in track.confidences],
                    'first_seen': track.first_seen,
                    'last_seen': track.last_seen,
                    'total_frames': track.total_frames,
                    'is_active': track.is_active
                }
                for track_id, track in self.tracks.items()
            },
            'metrics': self.metrics
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    def get_active_players(self) -> Set[int]:
        """Get set of currently active jersey numbers"""
        return {track.jersey_number for track in self.tracks.values() 
                if track.is_active and track.jersey_number > 0}

    def draw_tracks(self, frame: np.ndarray):
        """Draw tracking visualization on frame"""
        for track in self.tracks.values():
            if track.is_active and track.positions:
                bbox = track.get_current_position()
                if bbox is None:
                    continue
                    
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw jersey number and info
                label = f"#{track.jersey_number}" if track.jersey_number > 0 else f"ID:{track.player_id}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw motion trail
                if len(track.positions) > 1:
                    points = [(int((p[0] + p[2])/2), int((p[1] + p[3])/2)) 
                             for p in track.positions[-10:]]
                    for i in range(len(points)-1):
                        cv2.line(frame, points[i], points[i+1], (0, 255, 0), 1)