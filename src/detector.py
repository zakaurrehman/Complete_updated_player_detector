"""Improved player and jersey number detection module with color detection"""
from src import DEFAULT_CONFIG, logger, device
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional, Set
import time
import easyocr

class PlayerDetector:
    def __init__(self, 
                 model_path: str = "yolov8x.pt",
                 conf_threshold: float = None,
                 iou_threshold: float = None):
        """Initialize detector with improved settings"""
        # Use configuration from package defaults if not specified
        self.conf_threshold = conf_threshold or 0.3  # Detection confidence threshold
        self.iou_threshold = iou_threshold or 0.45   # IOU threshold
        self.model_path = model_path
        
        # Initialize tracking dictionaries
        self.player_tracking = {}
        self.jersey_detections = {}
        self.min_detections = 2  # Minimum detections needed to confirm
        
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
        
        # Cache for optimization
        self.number_cache = {}
        self.color_cache = {}

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

    def detect_jersey_color(self, jersey_region: np.ndarray) -> str:
        """Enhanced jersey color detection with broader ranges"""
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
            
            # Calculate mean HSV values
            mean_hsv = cv2.mean(hsv)[:3]
            
            # Calculate histograms for each channel
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Get dominant values
            dominant_h = np.argmax(h_hist)
            dominant_s = np.argmax(s_hist)
            dominant_v = np.argmax(v_hist)

            # Define broader color ranges in HSV
            # Format: (hue_range, min_saturation, min_value)
            color_definitions = {
                'red': [
                    ((0, 10), 50, 50),    # Lower red range
                    ((160, 180), 50, 50)  # Upper red range
                ],
                'blue': [((100, 140), 50, 50)],
                'white': [((0, 180), 0, 200)],
                'black': [((0, 180), 0, 0, 50)],
                'yellow': [((20, 40), 100, 100)],
                'green': [((40, 80), 50, 50)],
                'orange': [((10, 20), 100, 100)]
            }

            # Check for white/black first using value and saturation
            if dominant_v > 200 and dominant_s < 30:
                return 'white'
            if dominant_v < 50 and dominant_s < 50:
                return 'black'

            # Check other colors
            for color, ranges in color_definitions.items():
                for color_range in ranges:
                    h_range = color_range[0]
                    min_s = color_range[1]
                    min_v = color_range[2]

                    # Check if dominant hue falls within range
                    if h_range[0] <= dominant_h <= h_range[1]:
                        if dominant_s >= min_s and dominant_v >= min_v:
                            # Additional validation using mean values
                            if self._validate_color(mean_hsv, color):
                                return color

            # Calculate color score for each defined color
            color_scores = {}
            for color, ranges in color_definitions.items():
                max_score = 0
                for color_range in ranges:
                    h_range = color_range[0]
                    min_s = color_range[1]
                    min_v = color_range[2]

                    # Calculate score based on how well the color matches the criteria
                    h_score = self._calculate_hue_score(dominant_h, h_range)
                    s_score = 1.0 if dominant_s >= min_s else dominant_s / min_s
                    v_score = 1.0 if dominant_v >= min_v else dominant_v / min_v

                    score = h_score * s_score * v_score
                    max_score = max(max_score, score)
                
                color_scores[color] = max_score

            # Get the color with highest score if it's above threshold
            best_color = max(color_scores.items(), key=lambda x: x[1])
            if best_color[1] > 0.5:  # Minimum confidence threshold
                return best_color[0]

            return 'unknown'

        except Exception as e:
            logger.error(f"Color detection error: {str(e)}")
            return 'unknown'

    def _validate_color(self, mean_hsv: Tuple[float, float, float], color: str) -> bool:
        """Additional validation for color detection"""
        h, s, v = mean_hsv
        
        if color == 'red':
            return (h < 10 or h > 160) and s > 50 and v > 50
        elif color == 'blue':
            return 100 < h < 140 and s > 50 and v > 50
        elif color == 'white':
            return s < 30 and v > 200
        elif color == 'black':
            return v < 50
        elif color == 'yellow':
            return 20 < h < 40 and s > 100 and v > 100
        elif color == 'green':
            return 40 < h < 80 and s > 50 and v > 50
        elif color == 'orange':
            return 10 < h < 20 and s > 100 and v > 100
        
        return False

    def _calculate_hue_score(self, hue: float, hue_range: Tuple[float, float]) -> float:
        """Calculate how well a hue matches a given range"""
        if hue_range[0] <= hue <= hue_range[1]:
            return 1.0
        
        # Handle red hue wrap-around
        if hue_range[0] > hue_range[1]:  # Red color case
            if hue >= hue_range[0] or hue <= hue_range[1]:
                return 1.0
                
        # Calculate distance to range
        dist_to_range = min(
            abs(hue - hue_range[0]),
            abs(hue - hue_range[1])
        )
        
        # Convert distance to score (0-1)
        max_dist = 90  # Maximum possible distance in hue space
        score = max(0, 1 - (dist_to_range / max_dist))
        return score

    def preprocess_jersey_region(self, jersey_region: np.ndarray) -> List[np.ndarray]:
        """Enhanced preprocessing for jersey regions"""
        try:
            processed_versions = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2GRAY)
            
            # Version 1: Standard preprocessing
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced1 = clahe.apply(denoised)
            _, threshold1 = cv2.threshold(enhanced1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_versions.append(threshold1)
            
            # Version 2: High contrast
            enhanced2 = cv2.convertScaleAbs(gray, alpha=1.8, beta=20)
            _, threshold2 = cv2.threshold(enhanced2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_versions.append(threshold2)
            
            return processed_versions
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return []

    def detect_jersey_number(self, jersey_region: np.ndarray) -> Optional[int]:
        """Detect jersey numbers with adjusted thresholds"""
        try:
            # More permissive size validation
            if jersey_region.shape[0] < 15 or jersey_region.shape[1] < 15:  # Reduced minimum size
                return None

            # Get preprocessed versions
            processed_versions = self.preprocess_jersey_region(jersey_region)
            if not processed_versions:
                return None

            all_candidates = []
            
            # More permissive OCR configurations
            ocr_configs = [
                {
                    'min_size': 15,
                    'text_threshold': 0.4,  # Lower threshold
                    'low_text': 0.25,
                    'width_ths': 0.4,
                    'height_ths': 0.4
                },
                {
                    'min_size': 12,         # Even smaller minimum size
                    'text_threshold': 0.35,  # Lower threshold
                    'low_text': 0.2,
                    'width_ths': 0.3,
                    'height_ths': 0.3
                }
            ]

            for img in processed_versions:
                for config in ocr_configs:
                    try:
                        results = self.reader.readtext(
                            img,
                            allowlist='0123456789',
                            paragraph=False,
                            batch_size=1,
                            **config
                        )

                        for bbox, text, conf in results:
                            try:
                                if conf > 0.4:  # Lower confidence threshold
                                    text = text.strip()
                                    if text.isdigit():
                                        number = int(text)
                                        if 1 <= number <= 99:  # Valid jersey range
                                            box_width = bbox[1][0] - bbox[0][0]
                                            box_height = bbox[2][1] - bbox[1][1]
                                            relative_size = (box_width * box_height) / (img.shape[0] * img.shape[1])
                                            aspect_ratio = box_height / box_width
                                            
                                            # More permissive size and ratio constraints
                                            if (0.05 <= relative_size <= 0.95 and  # More permissive size range
                                                0.7 <= aspect_ratio <= 3.5):       # More permissive aspect ratio
                                                all_candidates.append((number, conf))
                                                logger.debug(f"Found candidate number {number} with confidence {conf:.2f}")
                            except ValueError:
                                continue
                    except Exception as e:
                        logger.warning(f"OCR error with config {config}: {str(e)}")
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

                # Find best candidate with more permissive thresholds
                best_number = None
                best_score = 0
                
                for num, count in number_counts.items():
                    avg_conf = number_confidences[num] / count
                    consistency_score = count * avg_conf
                    
                    # Lower minimum scores
                    min_score = 0.5 if num < 10 else 0.4  # Lower thresholds for both single and double digits
                    
                    if consistency_score > best_score and avg_conf > min_score:
                        best_score = consistency_score
                        best_number = num

                if best_number is not None:
                    if best_number not in self.jersey_detections:
                        self.jersey_detections[best_number] = 1
                    else:
                        self.jersey_detections[best_number] += 1

                    # Reduced minimum detections requirement
                    if self.jersey_detections[best_number] >= 2:  # Reduced from 3 to 2
                        return best_number

            return None

        except Exception as e:
            logger.error(f"Jersey number detection error: {str(e)}")
            return None

    def detect_players(self, frame: np.ndarray, frame_number: int) -> List[Dict]:
        """Detect players with adjusted thresholds"""
        try:
            self.metrics['total_frames'] += 1
            frame_players = []
            detections = []
            
            # Run YOLO detection with more permissive threshold
            results = self.model(
                frame,
                conf=0.25,  # Lower confidence threshold
                iou=0.35,   # Lower IOU threshold
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
                    min_width = frame.shape[1] // 35   # More permissive minimum width
                    max_width = frame.shape[1] // 2.5  # More permissive maximum width
                    min_height = frame.shape[0] // 25  # More permissive minimum height
                    max_height = frame.shape[0] // 1.5 # More permissive maximum height
                    
                    if (min_width <= width <= max_width and
                        min_height <= height <= max_height and
                        1.0 <= height/width <= 4.5):  # More permissive aspect ratio
                        
                        # Try multiple jersey regions
                        jersey_regions = []
                        region_configs = [
                            (0.15, 0.45),  # Upper chest
                            (0.20, 0.50),  # Mid chest
                            (0.25, 0.55),  # Lower chest
                        ]
                        
                        for top_ratio, bottom_ratio in region_configs:
                            jersey_top = y1 + int(height * top_ratio)
                            jersey_bottom = y1 + int(height * bottom_ratio)
                            # Add horizontal margin
                            margin = int(width * 0.1)
                            jersey_left = max(0, x1 - margin)
                            jersey_right = min(frame.shape[1], x2 + margin)
                            
                            if 0 <= jersey_top < jersey_bottom <= frame.shape[0]:
                                region = frame[jersey_top:jersey_bottom, jersey_left:jersey_right]
                                if region.size > 0:
                                    jersey_regions.append(region)
                        
                        # Try to detect number in each region
                        for region in jersey_regions:
                            jersey_number = self.detect_jersey_number(region)
                            if jersey_number is not None:
                                jersey_color = self.detect_jersey_color(region)
                                logger.debug(f"Found player #{jersey_number} with color {jersey_color}")
                                
                                player_info = {
                                    'number': jersey_number,
                                    'color': jersey_color,
                                    'confidence': float(conf)
                                }
                                frame_players.append(player_info)
                                
                                detection = {
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': float(conf),
                                    'frame_number': frame_number,
                                    'jersey_number': jersey_number,
                                    'jersey_color': jersey_color
                                }
                                detections.append(detection)
                                break  # Found a number, no need to check other regions
                
                if frame_players:
                    self.update_player_tracking(frame_players, frame_number)
                    logger.info(f"Frame {frame_number}: Found {len(frame_players)} players")

            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return []
        
    def update_player_tracking(self, current_players: List[Dict], frame_number: int):
        """Track player appearances with color information"""
        for player in current_players:
            number = player['number']
            color = player['color']
            key = (number, color)
            
            if key not in self.player_tracking:
                self.player_tracking[key] = {
                    'first_seen': frame_number,
                    'last_seen': frame_number,
                    'total_frames': 1,
                    'consecutive_frames': 1
                }
            else:
                track = self.player_tracking[key]
                track['last_seen'] = frame_number
                track['total_frames'] += 1
                if frame_number - track['last_seen'] <= 2:
                    track['consecutive_frames'] += 1
                else:
                    track['consecutive_frames'] = 1

    def get_player_stats(self, frame_number: int, fps: float = DEFAULT_CONFIG['video']['default_fps']) -> str:
        """Get formatted player statistics with color grouping"""
        # Calculate timestamp
        total_seconds = frame_number / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"

        # Get active players grouped by color
        active_players_by_color = {}
        for (number, color), data in self.player_tracking.items():
            if data['consecutive_frames'] >= 2:
                if color not in active_players_by_color:
                    active_players_by_color[color] = []
                active_players_by_color[color].append(number)

        # Format output
        if active_players_by_color:
            output_lines = [f"Time {timestamp} - Active Players:"]
            for color in sorted(active_players_by_color.keys()):
                players = sorted(active_players_by_color[color])
                color_str = f"{color.title()} jersey:"
                players_str = ", ".join(f"#{num}" for num in players)
                output_lines.append(f"{color_str.ljust(15)} {players_str}")
            return "\n".join(output_lines)
        else:
            return f"Time {timestamp} - No active players detected"

    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        stats = {
            'total_frames': self.metrics['total_frames'],
            'total_detections': self.metrics['total_detections'],
            'unique_numbers': sorted(list(self.metrics['jersey_numbers_found'])),
            'jersey_colors': sorted(set(color for _, color in self.player_tracking.keys())),
            'detection_counts': dict(sorted(
                self.jersey_detections.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        }
        return stats

    def cleanup(self):
        """Clean up resources"""
        self.number_cache.clear()
        self.color_cache.clear()
        self.jersey_detections.clear()
        self.player_tracking.clear()
        torch.cuda.empty_cache()