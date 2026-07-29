from setuptools import setup, find_packages

setup(
    name="turtlebot_tracker",
    version="1.0.0",
    description="StarLine Hackathon 2026 - 3D Turtlebot2 Detection and Tracking in LiDAR Point Clouds",
    author="Hackathon Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "open3d>=0.17.0",
        "POT>=0.9.0",
        "scikit-learn>=1.2.0",
        "rosbags>=0.9.15",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)