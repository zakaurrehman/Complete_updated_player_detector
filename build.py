"""Build script for creating executable"""
import PyInstaller.__main__
import sys
import os

def create_executable():
    # Get the absolute path to the yolov8 model file
    model_path = os.path.abspath("yolov8x.pt")
    
    # Define PyInstaller arguments
    args = [
        'gui.py',  # Your main script
        '--name=HockeyPlayerAnalysis',  # Name of the executable
        '--onedir',  # Create a one-folder bundle
        '--noconsole',  # Don't show console window
        '--add-data=src;src',  # Include src package
        f'--add-data={model_path};.',  # Include YOLO model
        '--hidden-import=easyocr',
        '--hidden-import=torch',
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=ultralytics',
        '--hidden-import=src',
    ]
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)

if __name__ == "__main__":
    create_executable()