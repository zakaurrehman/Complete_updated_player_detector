"""Video processing module for hockey video analysis with color tracking"""
from src import DEFAULT_CONFIG, logger
from src.detector import PlayerDetector
import cv2
import numpy as np
import os
from typing import Generator, Tuple, Optional, List, Callable, Set, Dict
import time
from dataclasses import dataclass

@dataclass
class VideoChunk:
    """Class to store video chunk information"""
    start_frame: int
    end_frame: int
    start_time: float  # in seconds
    end_time: float    # in seconds
    fps: float

class VideoProcessor:
    def __init__(self, video_path: str):
        """Initialize video processor"""
        self.video_path = video_path
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video loaded: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        
        # Initialize player detector
        self.detector = PlayerDetector()
        
        # Initialize time frame tracking
        self.timeframe_duration = 30  # seconds
        
        # Processing metrics
        self.metrics = {
            'processed_frames': 0,
            'detected_players': set(),
            'start_time': None,
            'processing_time': 0,
            'timeframes_analyzed': 0
        }

    def get_timeframe_str(self, frame_number: int, offset_second: int = 0) -> str:
        """Get formatted timeframe string"""
        total_seconds = frame_number / self.fps
        current_timeframe = int(total_seconds // self.timeframe_duration)
        
        start_seconds = current_timeframe * self.timeframe_duration + offset_second
        end_seconds = start_seconds + self.timeframe_duration - offset_second
        
        start_minutes = int(start_seconds // 60)
        start_secs = int(start_seconds % 60)
        end_minutes = int(end_seconds // 60)
        end_secs = int(end_seconds % 60)
        
        return f"{start_minutes:02d}:{start_secs:02d} - {end_minutes:02d}:{end_secs:02d}"

    def get_current_timeframe(self, frame_number: int) -> int:
        """Get current timeframe index"""
        return int((frame_number / self.fps) // self.timeframe_duration)

    def format_timeframe_output(self, timeframe_idx: int, player_detections: Dict[str, Dict[int, int]]) -> str:
        """Format output for a timeframe with color grouping"""
        timeframe_str = self.get_timeframe_str(
            timeframe_idx * self.timeframe_duration * self.fps,
            offset_second=1 if timeframe_idx > 0 else 0
        )
        
        output_lines = [f"Time {timeframe_str} - Active Players:"]
        min_detections = 3  # Minimum detections required to consider a player active
        
        # Process each color
        active_players_by_color = {}
        for color, players in player_detections.items():
            active_players = [num for num, count in players.items() if count >= min_detections]
            if active_players:
                active_players_by_color[color] = sorted(active_players)
        
        if active_players_by_color:
            # Add each color's players to output
            for color in sorted(active_players_by_color.keys()):
                players = active_players_by_color[color]
                color_str = f"{color.title()} jersey:"
                players_str = ", ".join(f"#{num}" for num in players)
                output_lines.append(f"{color_str.ljust(15)} {players_str}")
        else:
            output_lines.append("No active players detected")
            
        return "\n".join(output_lines)

    def process_video(self, output_dir: str, progress_callback: Optional[Callable[[float], None]] = None) -> str:
        """Process video with timeframe-based color tracking"""
        os.makedirs(output_dir, exist_ok=True)
        output_stats_path = os.path.join(output_dir, "player_stats.txt")
        stats_output = []
        
        print("Starting video analysis...")
        self.metrics['start_time'] = time.time()
        
        try:
            processed_frames = 0
            current_timeframe = -1
            frame_skip = int(self.fps / 5)  # Process 5 frames per second
            
            # Track players by color within timeframes
            # Structure: timeframe -> color -> {player_number -> detection_count}
            timeframe_detections: Dict[int, Dict[str, Dict[int, int]]] = {}
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                try:
                    # Only process every nth frame
                    if processed_frames % frame_skip == 0:
                        frame_timeframe = self.get_current_timeframe(processed_frames)
                        
                        # Initialize new timeframe tracking
                        if frame_timeframe not in timeframe_detections:
                            timeframe_detections[frame_timeframe] = {}
                        
                        # Process frame and update detections
                        detections = self.detector.detect_players(frame, processed_frames)
                        for det in detections:
                            color = det['jersey_color']
                            number = det['jersey_number']
                            
                            if color not in timeframe_detections[frame_timeframe]:
                                timeframe_detections[frame_timeframe][color] = {}
                            
                            if number not in timeframe_detections[frame_timeframe][color]:
                                timeframe_detections[frame_timeframe][color][number] = 0
                            timeframe_detections[frame_timeframe][color][number] += 1
                        
                        # Check if we've entered a new timeframe
                        if frame_timeframe > current_timeframe:
                            # Output previous timeframe results if valid
                            if current_timeframe >= 0:
                                output = self.format_timeframe_output(
                                    current_timeframe,
                                    timeframe_detections[current_timeframe]
                                )
                                print(output)
                                stats_output.append(output)
                                self.metrics['timeframes_analyzed'] += 1
                            
                            current_timeframe = frame_timeframe
                    
                    # Update progress
                    if processed_frames % int(self.fps) == 0:
                        progress = (processed_frames / self.total_frames) * 100
                        if progress_callback:
                            progress_callback(progress)
                    
                    processed_frames += 1
                    self.metrics['processed_frames'] = processed_frames
                    
                except Exception as e:
                    logger.error(f"Error processing frame {processed_frames}: {str(e)}")
                    processed_frames += 1
                    continue
            
            # Output final timeframe
            if current_timeframe >= 0:
                output = self.format_timeframe_output(
                    current_timeframe,
                    timeframe_detections[current_timeframe]
                )
                print(output)
                stats_output.append(output)
                self.metrics['timeframes_analyzed'] += 1
            
            # Save results
            with open(output_stats_path, 'w') as f:
                f.write('\n'.join(stats_output))
            
            self.metrics['processing_time'] = time.time() - self.metrics['start_time']
            self._log_processing_stats()
            
            return output_stats_path
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return ""
        finally:
            self.cleanup()

    def _log_processing_stats(self):
        """Log processing statistics"""
        stats = {
            'Total time': f"{self.metrics['processing_time']:.2f}s",
            'Processed frames': self.metrics['processed_frames'],
            'Average FPS': f"{self.metrics['processed_frames'] / self.metrics['processing_time']:.2f}",
            'Timeframes analyzed': self.metrics['timeframes_analyzed']
        }
        
        logger.info("\nProcessing Statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

    def cleanup(self):
        """Release resources"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()