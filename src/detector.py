"""Improved player and jersey number detection module with high accuracy focus"""
from src import DEFAULT_CONFIG, logger, device
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import time
import easyocr

class PlayerDetector:
    def __init__(self, 
                 model_path: str = "yolov8x.pt",
                 conf_threshold: float = None,
                 iou_threshold: float = None):
        """Initialize detector with balanced accuracy and sensitivity"""
        # Use configuration from package defaults if not specified
        self.conf_threshold = conf_threshold or 0.25  # Lower confidence threshold for better detection
        self.iou_threshold = iou_threshold or 0.45    # Adjusted IOU threshold
        self.model_path = model_path
        
        # Initialize tracking dictionaries
        self.player_tracking = {}
        self.jersey_detections = {}
        self.min_detections = 2  # Reduced minimum detections for initial discovery
        
        # Performance metrics
        self.metrics = {
            'total_frames': 0,
            'total_detections': 0,
            'jersey_numbers_found': set()
        }

        # Initialize models
        self.device = device
        logger.info(f"Initializing PlayerDetector with confidence threshold: {self.conf_threshold}")
        logger.info(f"Using device: {self.device}")
        
        self.model = self.load_model()
        self.reader = easyocr.Reader(['en'], gpu=True if str(self.device)=='cuda' else False)

    def load_model(self):
        """Load and configure YOLO model"""
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            logger.info(f"Model loaded successfully: {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_jersey_region(self, jersey_region: np.ndarray) -> List[np.ndarray]:
        """Apply multiple preprocessing techniques for better number detection"""
        try:
            processed_versions = []
            
            # Basic grayscale conversion
            gray = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2GRAY)
            
            # Version 1: Standard preprocessing
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced1 = clahe.apply(denoised)
            _, threshold1 = cv2.threshold(enhanced1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_versions.append(threshold1)
            
            # Version 2: High contrast enhancement
            enhanced2 = cv2.convertScaleAbs(gray, alpha=1.8, beta=20)
            _, threshold2 = cv2.threshold(enhanced2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_versions.append(threshold2)
            
            # Version 3: Adaptive thresholding
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            processed_versions.append(adaptive)
            
            return processed_versions
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return []

    def detect_jersey_number(self, jersey_region: np.ndarray) -> Optional[int]:
        """Detect jersey numbers with improved sensitivity"""
        try:
            # More permissive size validation
            if jersey_region.shape[0] < 20 or jersey_region.shape[1] < 20:
                return None

            # Get preprocessed versions
            processed_versions = self.preprocess_jersey_region(jersey_region)
            if not processed_versions:
                return None

            all_candidates = []
            
            # OCR configurations with adjusted parameters
            ocr_configs = [
                {
                    'min_size': 20,
                    'text_threshold': 0.5,
                    'low_text': 0.3,
                    'width_ths': 0.5,
                    'height_ths': 0.5
                },
                {
                    'min_size': 15,
                    'text_threshold': 0.45,
                    'low_text': 0.25,
                    'width_ths': 0.4,
                    'height_ths': 0.4
                }
            ]

            for img in processed_versions:
                for config in ocr_configs:
                    results = self.reader.readtext(
                        img,
                        allowlist='0123456789',
                        paragraph=False,
                        batch_size=1,
                        **config
                    )

                    for bbox, text, conf in results:
                        try:
                            if conf > 0.45:  # Lower confidence threshold
                                text = text.strip()
                                if text.isdigit():
                                    number = int(text)
                                    if 1 <= number <= 99:  # Valid jersey range
                                        box_width = bbox[1][0] - bbox[0][0]
                                        box_height = bbox[2][1] - bbox[1][1]
                                        relative_size = (box_width * box_height) / (img.shape[0] * img.shape[1])
                                        aspect_ratio = box_height / box_width
                                        
                                        # More permissive size and aspect ratio validation
                                        if (0.1 <= relative_size <= 0.9 and
                                            0.8 <= aspect_ratio <= 3.0):
                                            all_candidates.append((number, conf))
                                            logger.debug(f"Found candidate number {number} with confidence {conf:.2f}")
                        except ValueError:
                            continue

            if all_candidates:
                # Group candidates
                number_counts = {}
                number_confidences = {}
                
                for num, conf in all_candidates:
                    if num not in number_counts:
                        number_counts[num] = 0
                        number_confidences[num] = 0
                    number_counts[num] += 1
                    number_confidences[num] += conf

                # Find best candidate
                best_number = None
                best_score = 0
                
                for num, count in number_counts.items():
                    avg_conf = number_confidences[num] / count
                    consistency_score = count * avg_conf
                    
                    # Adjusted thresholds
                    min_score = 0.7 if num < 10 else 0.6
                    
                    if consistency_score > best_score and avg_conf > min_score:
                        best_score = consistency_score
                        best_number = num

                if best_number is not None:
                    # Update tracking
                    if best_number not in self.jersey_detections:
                        self.jersey_detections[best_number] = 1
                    else:
                        self.jersey_detections[best_number] += 1

                    if self.jersey_detections[best_number] >= self.min_detections:
                        logger.debug(f"Confirmed jersey number {best_number}")
                        return best_number

            return None

        except Exception as e:
            logger.error(f"Jersey number detection error: {str(e)}")
            return None

    def detect_players(self, frame: np.ndarray, frame_number: int) -> List[Dict]:
        """Detect players with improved sensitivity"""
        try:
            self.metrics['total_frames'] += 1
            frame_players = set()
            detections = []
            
            # Run YOLO detection with adjusted parameters
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],  # Person class
                verbose=False
            )
            
            if results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    width = x2 - x1
                    height = y2 - y1
                    
                    # More permissive size filtering
                    min_width = frame.shape[1] // 30  # Less restrictive minimum width
                    max_width = frame.shape[1] // 2   # More permissive maximum width
                    min_height = frame.shape[0] // 20
                    max_height = frame.shape[0] // 1.5  # More permissive maximum height
                    
                    if (min_width <= width <= max_width and
                        min_height <= height <= max_height and
                        1.2 <= height/width <= 4.0):  # More permissive aspect ratio
                        
                        # Extract larger jersey regions
                        jersey_regions = []
                        
                        # Multiple region configurations
                        configs = [
                            (0.1, 0.4),   # Upper torso (larger region)
                            (0.2, 0.5),   # Mid torso
                            (0.15, 0.45),  # Center region
                            (0.25, 0.55)   # Lower torso
                        ]
                        
                        for top_ratio, bottom_ratio in configs:
                            jersey_top = y1 + int(height * top_ratio)
                            jersey_bottom = y1 + int(height * bottom_ratio)
                            
                            # Add margins to width
                            margin = int(width * 0.1)
                            jersey_left = max(0, x1 - margin)
                            jersey_right = min(frame.shape[1], x2 + margin)
                            
                            if 0 <= jersey_top < jersey_bottom <= frame.shape[0]:
                                region = frame[jersey_top:jersey_bottom, jersey_left:jersey_right]
                                if region.size > 0 and region.shape[0] >= 20 and region.shape[1] >= 20:
                                    jersey_regions.append(region)
                        
                        # Try to detect number in each region
                        for region in jersey_regions:
                            jersey_number = self.detect_jersey_number(region)
                            if jersey_number is not None:
                                if jersey_number not in frame_players:
                                    frame_players.add(jersey_number)
                                    detection = {
                                        'bbox': (x1, y1, x2, y2),
                                        'confidence': float(conf),
                                        'frame_number': frame_number,
                                        'jersey_number': jersey_number
                                    }
                                    detections.append(detection)
                                    logger.debug(f"Detected player #{jersey_number} with confidence {conf:.2f}")
                                break
                
                if frame_players:
                    self.update_player_tracking(frame_players, frame_number)
                    logger.info(f"Frame {frame_number}: Found players {sorted(list(frame_players))}")

            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return []

    def update_player_tracking(self, current_players: set, frame_number: int):
        """Track player appearances with improved consistency"""
        for number in current_players:
            if number not in self.player_tracking:
                self.player_tracking[number] = {
                    'first_seen': frame_number,
                    'last_seen': frame_number,
                    'total_frames': 1,
                    'consecutive_frames': 1
                }
            else:
                track = self.player_tracking[number]
                track['last_seen'] = frame_number
                track['total_frames'] += 1
                
                # Allow small gaps in detection
                if frame_number - track['last_seen'] <= 2:
                    track['consecutive_frames'] += 1
                else:
                    track['consecutive_frames'] = 1

    def get_player_stats(self, frame_number: int, fps: float = DEFAULT_CONFIG['video']['default_fps']) -> str:
        """Get formatted player statistics with 30-second intervals"""
        # Calculate timestamp
        total_seconds = frame_number / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        
        # Only output at 30-second intervals
        if seconds % 30 != 0:
            return ""
            
        timestamp = f"{minutes:02d}:{seconds:02d}"
        
        # Get active players with reasonable threshold
        active_players = [num for num, data in self.player_tracking.items() 
                         if data['consecutive_frames'] >= 2]
        
        # Format player numbers
        formatted_players = [f"#{num}" for num in sorted(active_players)]
        
        # Output format
        if formatted_players:
            return f"Time {timestamp} - Active Players: {', '.join(formatted_players)}"
        else:
            return f"Time {timestamp} - No active players detected"

    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        return {
            'total_frames': self.metrics['total_frames'],
            'total_detections': self.metrics['total_detections'],
            'unique_numbers': sorted(list(self.metrics['jersey_numbers_found'])),
            'detection_counts': dict(sorted(
                self.jersey_detections.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        }

    def cleanup(self):
        """Clean up resources"""
        self.number_cache.clear()
        self.cache_expiry.clear()
        self.jersey_detections.clear()
        self.player_tracking.clear()
        torch.cuda.empty_cache()