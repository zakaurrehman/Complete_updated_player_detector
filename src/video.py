"""Video processing module for hockey video analysis"""
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
        self.current_timeframe_players: Dict[int, Set[int]] = {}  # Key: timeframe_index, Value: set of player numbers
        self.timeframe_duration = 30  # seconds
        
        # Processing metrics
        self.metrics = {
            'processed_frames': 0,
            'processed_chunks': 0,
            'detected_players': set(),
            'start_time': None,
            'processing_time': 0
        }

    def get_current_timeframe(self, frame_number: int) -> int:
        """Get current timeframe index"""
        return int((frame_number / self.fps) // self.timeframe_duration)

    def process_frame(self, frame: np.ndarray, frame_number: int) -> None:
        """Process a single frame and track players in timeframes"""
        # Get detections from detector
        detections = self.detector.detect_players(frame, frame_number)
        
        # Get current timeframe
        timeframe_idx = self.get_current_timeframe(frame_number)
        
        # Initialize timeframe if needed
        if timeframe_idx not in self.current_timeframe_players:
            self.current_timeframe_players[timeframe_idx] = set()
            
        # Add detected players to current timeframe
        for detection in detections:
            self.current_timeframe_players[timeframe_idx].add(detection['jersey_number'])

    def format_timeframe_output(self, timeframe_idx: int) -> str:
        """Format output for a timeframe"""
        timeframe_str = self.get_timeframe_str(timeframe_idx * self.timeframe_duration * self.fps)
        
        if timeframe_idx in self.current_timeframe_players and self.current_timeframe_players[timeframe_idx]:
            players = sorted(self.current_timeframe_players[timeframe_idx])
            players_str = ", ".join(f"#{num}" for num in players)
            return f"Time {timeframe_str} - Active Players: {players_str}"
        else:
            return f"Time {timeframe_str} - No active players detected"

    def process_video(self, output_dir: str, progress_callback: Optional[Callable[[float], None]] = None) -> str:
        """Process entire video and track players in timeframes"""
        os.makedirs(output_dir, exist_ok=True)
        output_stats_path = os.path.join(output_dir, "player_stats.txt")
        stats_output = []
        
        print("Starting video analysis...")
        self.metrics['start_time'] = time.time()
        
        try:
            processed_frames = 0
            current_timeframe = -1
            frame_skip = int(self.fps / 5)  # Process 5 frames per second
            timeframe_players: Dict[int, Dict[int, int]] = {}  # Timeframe -> {Player -> Frame count}
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                try:
                    # Only process every nth frame
                    if processed_frames % frame_skip == 0:
                        # Get current timeframe
                        frame_timeframe = self.get_current_timeframe(processed_frames)
                        
                        # Initialize new timeframe if needed
                        if frame_timeframe not in timeframe_players:
                            timeframe_players[frame_timeframe] = {}
                            
                        # Process frame and update player counts in current timeframe
                        detections = self.detector.detect_players(frame, processed_frames)
                        for detection in detections:
                            player_num = detection['jersey_number']
                            if player_num not in timeframe_players[frame_timeframe]:
                                timeframe_players[frame_timeframe][player_num] = 0
                            timeframe_players[frame_timeframe][player_num] += 1
                        
                        # Check if we've entered a new timeframe
                        if frame_timeframe > current_timeframe:
                            # Output previous timeframe results if valid
                            if current_timeframe >= 0:
                                # Get players that appeared enough times in the timeframe
                                min_appearances = 3  # Minimum number of detections needed
                                active_players = {
                                    player for player, count in timeframe_players[current_timeframe].items()
                                    if count >= min_appearances
                                }
                                
                                # Format timeframe string with 1-second offset for non-first frames
                                time_str = self.get_timeframe_str(
                                    current_timeframe * self.timeframe_duration * self.fps,
                                    offset_second=1 if current_timeframe > 0 else 0
                                )
                                
                                # Format output
                                if active_players:
                                    players_str = ", ".join(f"#{num}" for num in sorted(active_players))
                                    output = f"Time {time_str} - Active Players: {players_str}"
                                else:
                                    output = f"Time {time_str} - No active players detected"
                                
                                print(output)
                                stats_output.append(output)
                            
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
                min_appearances = 3
                active_players = {
                    player for player, count in timeframe_players[current_timeframe].items()
                    if count >= min_appearances
                }
                
                time_str = self.get_timeframe_str(
                    current_timeframe * self.timeframe_duration * self.fps,
                    offset_second=1 if current_timeframe > 0 else 0
                )
                
                if active_players:
                    players_str = ", ".join(f"#{num}" for num in sorted(active_players))
                    output = f"Time {time_str} - Active Players: {players_str}"
                else:
                    output = f"Time {time_str} - No active players detected"
                
                print(output)
                stats_output.append(output)
            
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

    def get_timeframe_str(self, frame_number: int, offset_second: int = 0) -> str:
        """Get formatted timeframe string with optional offset"""
        total_seconds = frame_number / self.fps
        current_timeframe = int(total_seconds // self.timeframe_duration)
        
        start_seconds = current_timeframe * self.timeframe_duration + offset_second
        end_seconds = start_seconds + self.timeframe_duration - offset_second
        
        start_minutes = int(start_seconds // 60)
        start_secs = int(start_seconds % 60)
        end_minutes = int(end_seconds // 60)
        end_secs = int(end_seconds % 60)
        
        return f"{start_minutes:02d}:{start_secs:02d} - {end_minutes:02d}:{end_secs:02d}"

    def _log_processing_stats(self):
        """Log processing statistics"""
        stats = {
            'Total time': f"{self.metrics['processing_time']:.2f}s",
            'Processed frames': self.metrics['processed_frames'],
            'Average FPS': f"{self.metrics['processed_frames'] / self.metrics['processing_time']:.2f}",
            'Total timeframes': len(self.current_timeframe_players)
        }
        
        logger.info("\nProcessing Statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

    def cleanup(self):
        """Release resources"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()