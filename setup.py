"""Setup script for creating distribution"""
from setuptools import setup, find_packages

setup(
    name="HockeyPlayerAnalysis",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'ultralytics==8.0.196',
        'opencv-python==4.8.1.78',
        'numpy>=1.24.3',
        'torch>=2.0.0',
        'easyocr>=1.7.1',
    ],
    entry_points={
        'console_scripts': [
            'hockey-analysis=gui:main',
        ],
    },
)