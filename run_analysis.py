import os
import sys
import json
import requests
from video_processor import VideoProcessor

try:
    from config import OPENAI_API_KEY, OPENAI_API_BASE
except ImportError:
    OPENAI_API_KEY = "sk-kiuFsJpM40jgti6bl_CNLw"
    OPENAI_API_BASE = "https://api.rabbithole.cred.club/v1"

def main():
    # Download sample video if no path provided
    if len(sys.argv) < 2:
        print("No video path provided, downloading a sample video...")
        video_path = "sample_video.mp4"
        try:
            # Download a sample video if needed
            if not os.path.exists(video_path):
                sample_url = "https://filesamples.com/samples/video/mp4/sample_640x360.mp4"
                response = requests.get(sample_url)
                with open(video_path, 'wb') as f:
                    f.write(response.content)
                print(f"Sample video downloaded to {video_path}")
        except Exception as e:
            print(f"Error downloading sample video: {e}")
            return
    else:
        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
    
    # Set environment variables for the API
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
    
    print(f"Using API key: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}")
    print(f"Using API base: {OPENAI_API_BASE}")
    
    # Process video
    processor = VideoProcessor()  # Use the default constructor
    analysis = processor.process_video(video_path)
    
    # Save results
    output_file = os.path.splitext(os.path.basename(video_path))[0] + "_analysis.json"
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis complete! Results saved to {output_file}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total scenes detected: {len(analysis)}")
    for i, scene in enumerate(analysis):
        print(f"Scene {i+1}: {scene['duration']}s - {scene['metadata'].get('scene_context', 'N/A')}")

if __name__ == "__main__":
    main()