import os

# API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-kiuFsJpM40jgti6bl_CNLw")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.rabbithole.cred.club/v1")

# Video Processing Configuration
FRAME_INTERVAL = 2  # Extract frame every X seconds
SCENE_THRESHOLD = 30  # Threshold for scene change detection (0-100)
MIN_SCENE_DURATION = 3  # Minimum scene duration in seconds