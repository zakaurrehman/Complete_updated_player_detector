"""Hockey Player Tracking Package"""
import os
import logging
import cv2
import numpy as np
import torch

# Package information
__version__ = '1.0.0'

# Setup logging
def setup_logging():
    """Setup package-wide logging configuration"""
    logger = logging.getLogger('hockey_tracker')
    logger.setLevel(logging.INFO)
    
    # Create console handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    logger.info(f"GPU initialized: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
else:
    device = torch.device('cpu')
    logger.warning("CUDA not available, using CPU")

# Define default configuration
DEFAULT_CONFIG = {
    'video': {
        'chunk_duration': 300,  # 5 minutes in seconds
        'min_clip_duration': 5,  # minimum highlight clip duration in seconds
        'output_codec': 'mp4v',
        'default_fps': 30
    },
    'detector': {
        'confidence_threshold': 0.25,
        'nms_threshold': 0.45,
        'device': device
    },
    'tracker': {
        'max_track_age': 30,
        'min_detection_conf': 0.25,
        'iou_threshold': 0.3
    }
}

# Utility functions
def check_file_exists(filepath):
    """Check if a file exists and is accessible"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if not os.path.isfile(filepath):
        raise ValueError(f"Not a file: {filepath}")
    if not os.access(filepath, os.R_OK):
        raise PermissionError(f"File not readable: {filepath}")
    return True

def create_output_dir(dirpath):
    """Create output directory if it doesn't exist"""
    try:
        os.makedirs(dirpath, exist_ok=True)
        return dirpath
    except Exception as e:
        logger.error(f"Error creating directory {dirpath}: {str(e)}")
        raise

# Initialize output directories
OUTPUT_DIRS = {
    'output': 'output',
    'chunks': 'output/chunks',
    'highlights': 'output/highlights',
    'debug': 'output/debug'
}