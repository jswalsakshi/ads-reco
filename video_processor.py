import cv2
import numpy as np
import os
import json
import traceback
import threading
import time
from datetime import datetime
import queue
import textwrap
import requests
import base64
from io import BytesIO
from PIL import Image

class RealTimeVideoProcessor:
    def __init__(self, api_key):
        """Initialize the real-time video processor with GPT-4o API key"""
        self.api_key = api_key
        self.analysis_queue = queue.Queue()
        self.current_analysis = {}
        self.last_analysis_time = 0
        self.analysis_interval = 2.0  # Analyze every 2 seconds
        self.stop_analysis = False
        self.json_font_scale = 0.65  # Font scale for JSON display
        self.json_panel_width = 600  # Wider panel for better readability
        
    def _analyze_frame_with_gpt4o(self, frame):
        """
        Analyze a frame using GPT-4o via API
        
        Args:
            frame: The video frame to analyze
            
        Returns:
            Dictionary with analysis results from GPT-4o
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert image to base64
            pil_image = Image.fromarray(rgb_frame)
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Prepare API request
            url = "https://api.rabbithole.cred.club/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Craft prompt for GPT-4o
            prompt = """Analyze this image in detail and provide the following information:
            1. Scene Description: Describe what's happening in this scene in detail.
            2. Objects: List the main objects visible in this image as a JSON array.
            3. People and Actions: Describe any people in this image and what they are doing.
            4. Emotional Tone: What is the mood or emotional tone of this scene?
            
            Format your response as JSON with these fields.
            """
            
            # Create payload
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                        ]
                    }
                ],
                "response_format": {"type": "json_object"}
            }
            
            # Make API request
            response = requests.post(url, headers=headers, json=payload)
            response_data = response.json()
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                # Parse the response content into JSON
                content = response_data['choices'][0]['message']['content']
                analysis_results = json.loads(content)
                
                # Ensure Objects is always a list
                if 'Objects' in analysis_results and isinstance(analysis_results['Objects'], str):
                    analysis_results['Objects'] = [obj.strip() for obj in analysis_results['Objects'].split(',')]
                
                return analysis_results
            else:
                # Handle API error
                return {
                    "error": "Failed to get analysis from GPT-4o",
                    "api_response": response_data
                }
                
        except Exception as e:
            print(f"Error analyzing with GPT-4o: {e}")
            traceback.print_exc()
            return {
                "error": f"Exception during analysis: {str(e)}"
            }
            
    def _analyze_frame(self, frame, frame_number=None, video_time=None):
        """
        Analyze a single frame
        
        Args:
            frame: The video frame to analyze
            frame_number: Optional frame number
            video_time: Optional video timestamp
            
        Returns:
            Dictionary with detailed analysis results
        """
        # Basic metadata to include in all analyses
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        analysis = {
            "timestamp": timestamp,
            "frame_number": frame_number,
            "video_time": video_time
        }
        
        # Get frame basic stats
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # HSV color analysis for basic scene understanding
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_h = np.mean(hsv[:,:,0])
        avg_s = np.mean(hsv[:,:,1])
        avg_v = np.mean(hsv[:,:,2])
        
        # Add basic image statistics
        analysis["image_stats"] = {
            "brightness": f"{brightness:.1f}",
            "motion_blur": f"{blur:.1f}",
            "color_saturation": f"{avg_s:.1f}",
            "color_value": f"{avg_v:.1f}"
        }
        
        # Get GPT-4o analysis if API key is provided
        if self.api_key:
            # Get detailed analysis from GPT-4o
            gpt4o_analysis = self._analyze_frame_with_gpt4o(frame)
            analysis["gpt4o_analysis"] = gpt4o_analysis
            
            # Extract potential activities from GPT-4o description
            activities = []
            activity_keywords = ["walking", "running", "sitting", "standing", "eating", 
                                "talking", "playing", "working", "sleeping", "driving"]
            
            if "Scene Description" in gpt4o_analysis:
                description_lower = gpt4o_analysis["Scene Description"].lower()
                for activity in activity_keywords:
                    if activity in description_lower:
                        activities.append(activity)
            
            if activities:
                analysis["detected_activities"] = activities
            else:
                analysis["detected_activities"] = ["unknown"]
                
            # Extract emotional tone if available
            if "Emotional Tone" in gpt4o_analysis:
                analysis["emotional_tone"] = gpt4o_analysis["Emotional Tone"]
        else:
            # Mock analysis if no API key is provided
            if brightness > 150:
                scene_type = "bright outdoor scene"
                if avg_h > 20 and avg_h < 30:
                    description = "A sunny day with golden light. The scene appears to be outdoors."
                elif avg_h > 90 and avg_h < 150:
                    description = "A scene with greenery, possibly in a park or garden."
                else:
                    description = "A well-lit outdoor scene with clear visibility."
            elif brightness < 50:
                scene_type = "dark scene"
                description = "A dark scene with low visibility. Likely indoors or at night."
            else:
                scene_type = "normal lighting"
                if avg_s > 100:
                    description = "A colorful scene with moderate lighting. Various objects are visible."
                else:
                    description = "A scene with balanced lighting. The environment appears neutral."
                
            if blur < 10:
                activity = "static shot"
                motion_desc = "Little to no movement detected. The scene is relatively still."
            else:
                activity = "movement detected"
                motion_desc = "Significant movement detected in the frame, suggesting activity."
                
            # Mock objects based on color analysis
            mock_objects = []
            if avg_h > 20 and avg_h < 30 and brightness > 150:
                mock_objects.append("sky")
            if avg_h > 90 and avg_h < 150:
                mock_objects.append("vegetation")
            if avg_s < 50 and avg_v > 150:
                mock_objects.append("buildings")
            if avg_s > 100 and 60 < avg_h < 180:
                mock_objects.append("water")
                
            if not mock_objects:
                mock_objects = ["unknown object"]
                
            analysis.update({
                "detailed_description": description + " " + motion_desc,
                "scene_context": scene_type,
                "key_objects": mock_objects,
                "emotional_tone": "neutral",
                "detected_activity": activity,
                "color_data": {
                    "hue": f"{avg_h:.1f}",
                    "saturation": f"{avg_s:.1f}",
                    "value": f"{avg_v:.1f}"
                }
            })
            
        return analysis
        
    def _analysis_worker(self, video_path):
        """
        Worker thread to analyze video frames
        
        Args:
            video_path: Path to the video file
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.analysis_queue.put({"error": f"Could not open video file {video_path}"})
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_time = 1.0 / fps
        
        frame_count = 0
        
        while not self.stop_analysis:
            ret, frame = cap.read()
            
            if not ret:
                # Loop back to beginning for continuous analysis
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue
                
            frame_count += 1
            current_time = frame_count * frame_time
            
            # Format video time as HH:MM:SS.ms
            video_time_str = self._format_time(current_time)
            
            # Only analyze frames at the specified interval
            if current_time - self.last_analysis_time >= self.analysis_interval:
                print(f"Analyzing frame {frame_count} at video time {video_time_str}...")
                
                # Analyze the current frame with frame number and video time
                analysis = self._analyze_frame(
                    frame, 
                    frame_number=frame_count, 
                    video_time=video_time_str
                )
                
                # Add frame position as percentage through video
                if total_frames > 0:
                    analysis["video_progress"] = f"{(frame_count / total_frames) * 100:.1f}%"
                
                self.analysis_queue.put(analysis)
                self.last_analysis_time = current_time
                
            # Control analysis rate to not overload the system
            time.sleep(0.01)
            
        cap.release()
    
    def _format_time(self, seconds):
        """Format seconds as HH:MM:SS.ms"""
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{int((s % 1) * 1000):03d}"

    def _overlay_ad(self, frame, ad_path):
        """
        Overlay an ad image on a video frame
        
        Args:
            frame: The video frame to overlay the ad on
            ad_path: Path to the ad image file
            
        Returns:
            Frame with the ad overlaid
        """
        try:
            # Check if ad image exists
            if not os.path.exists(ad_path):
                return frame
                
            # Read the ad image
            ad_img = cv2.imread(ad_path, cv2.IMREAD_UNCHANGED)
            
            if ad_img is None:
                return frame
                
            # Resize ad if needed
            ad_height, ad_width = ad_img.shape[:2]
            _, frame_width = frame.shape[:2]
            
            # Position ad in the top-right corner
            y_offset = 10
            x_offset = frame_width - ad_width - 10
            
            # No need to create ROI since we're directly overwriting
            
            # Simple overlay
            frame[y_offset:y_offset+ad_height, x_offset:x_offset+ad_width] = ad_img
            
            return frame
        except Exception as e:
            print(f"Error overlaying ad: {e}")
            return frame
    
    def _draw_json_panel(self, analysis, width, height):
        """
        Create a visually appealing JSON panel with the analysis data
        
        Args:
            analysis: The analysis dictionary
            width: Width of the panel
            height: Height of the panel
            
        Returns:
            Image with the JSON data rendered
        """
        # Create a panel with nice background
        panel = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add a header bar
        cv2.rectangle(panel, (0, 0), (width, 50), (70, 70, 70), -1)
        cv2.putText(panel, "REAL-TIME VIDEO ANALYSIS", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
        
        # Current timestamp on header
        timestamp = analysis.get("timestamp", "")
        time_width = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][0]
        cv2.putText(panel, timestamp, (width - time_width - 20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1)
        
        # Display top section with key information when using GPT-4o
        if "gpt4o_analysis" in analysis:
            y_pos = 70
            
            # Show video time and frame number
            video_time = analysis.get("video_time", "")
            frame_num = analysis.get("frame_number", "")
            progress = analysis.get("video_progress", "")
            
            cv2.putText(panel, f"Video Time: {video_time}", (20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y_pos += 25
            
            cv2.putText(panel, f"Frame: {frame_num} ({progress})", (20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y_pos += 35
            
            # Display the GPT-4o scene description
            gpt4o_data = analysis["gpt4o_analysis"]
            
            # Scene description header
            cv2.rectangle(panel, (10, y_pos-20), (width-10, y_pos), (200, 200, 220), -1)
            cv2.putText(panel, "SCENE DESCRIPTION", (15, y_pos-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 100), 1)
            y_pos += 10
            
            # Display scene description text with wrapping
            scene_desc = gpt4o_data.get("Scene Description", "")
            if scene_desc:
                wrapped_text = textwrap.wrap(scene_desc, width=65)
                for i, line in enumerate(wrapped_text[:5]):  # Show first 5 lines only
                    cv2.putText(panel, line, (15, y_pos + i*22), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                y_pos += min(len(wrapped_text), 5) * 22 + 20
            
            # Objects section
            cv2.rectangle(panel, (10, y_pos-20), (width-10, y_pos), (200, 220, 200), -1)
            cv2.putText(panel, "OBJECTS", (15, y_pos-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)
            y_pos += 10
            
            # Display objects - handle both list and string formats
            objects_data = gpt4o_data.get("Objects", [])
            if isinstance(objects_data, str):
                objects_text = objects_data
            else:
                objects_text = ", ".join(str(obj) for obj in objects_data)
                
            if objects_text:
                wrapped_text = textwrap.wrap(objects_text, width=65)
                for i, line in enumerate(wrapped_text[:3]):  # Show first 3 lines only
                    cv2.putText(panel, line, (15, y_pos + i*22), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                y_pos += min(len(wrapped_text), 3) * 22 + 20
            
            # People and actions section
            cv2.rectangle(panel, (10, y_pos-20), (width-10, y_pos), (220, 200, 200), -1)
            cv2.putText(panel, "PEOPLE & ACTIONS", (15, y_pos-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 0, 0), 1)
            y_pos += 10
            
            # Display people and actions
            people_text = gpt4o_data.get("People and Actions", "")
            if people_text:
                wrapped_text = textwrap.wrap(people_text, width=65)
                for i, line in enumerate(wrapped_text[:3]):  # Show first 3 lines only
                    cv2.putText(panel, line, (15, y_pos + i*22), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                y_pos += min(len(wrapped_text), 3) * 22 + 20
            
            # Add image stats at the bottom
            stats_top = height - 120
            cv2.rectangle(panel, (10, stats_top), (width//2-10, height-10), (240, 240, 255), -1)
            cv2.rectangle(panel, (10, stats_top), (width//2-10, height-10), (70, 70, 70), 1)
            
            cv2.putText(panel, "IMAGE STATS", (20, stats_top + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 70), 1)
            
            if "image_stats" in analysis:
                stats = analysis["image_stats"]
                y_stat = stats_top + 50
                for key, value in stats.items():
                    cv2.putText(panel, f"{key}: {value}", (20, y_stat), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                    y_stat += 20
            
            # Add emotional tone in the bottom right
            emotion_top = height - 120
            cv2.rectangle(panel, (width//2+10, emotion_top), (width-10, height-10), (255, 240, 240), -1)
            cv2.rectangle(panel, (width//2+10, emotion_top), (width-10, height-10), (70, 70, 70), 1)
            
            cv2.putText(panel, "EMOTIONAL TONE", (width//2+20, emotion_top + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 70), 1)
            
            # Display emotional tone from GPT-4o
            emotion_text = gpt4o_data.get("Emotional Tone", "")
            if emotion_text:
                wrapped_text = textwrap.wrap(emotion_text, width=30)
                for i, line in enumerate(wrapped_text[:3]):
                    cv2.putText(panel, line, (width//2+20, emotion_top + 50 + i*20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            
        else:
            # For non-GPT4o analysis, show the standard JSON display
            formatted_json = json.dumps(analysis, indent=2)
            lines = formatted_json.split('\n')
            
            # Starting position for text
            y_pos = 80
            left_margin = 20
            line_height = int(30 * self.json_font_scale)
            
            for line in lines:
                # Calculate indentation
                indent_level = len(line) - len(line.lstrip())
                indent = indent_level * 10  # 10 pixels per indent level
                
                # Get the actual content
                content = line.strip()
                
                # Skip empty lines
                if not content:
                    y_pos += line_height // 2
                    continue
                    
                # Colorize different parts of the JSON
                if content.endswith('{') or content.endswith('[') or content.endswith('},') or content.endswith('],'):
                    # Brackets and braces
                    color = (0, 0, 180)  # Dark blue
                elif ':' in content:
                    # Keys
                    key_part = content.split(':', 1)[0].strip('"",')
                    value_part = content.split(':', 1)[1].strip()
                    
                    # Draw the key part
                    cv2.putText(panel, key_part + ":", (left_margin + indent, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, self.json_font_scale, (180, 0, 0), 1)
                    
                    # Calculate position for value part
                    key_width = cv2.getTextSize(key_part + ":", cv2.FONT_HERSHEY_SIMPLEX, self.json_font_scale, 1)[0][0]
                    
                    # Handle long values with line wrapping
                    if len(value_part) > 40:
                        wrapped_value = textwrap.wrap(value_part, width=40)
                        cv2.putText(panel, wrapped_value[0], (left_margin + indent + key_width + 10, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, self.json_font_scale, (0, 100, 0), 1)
                        
                        # Add additional lines for long values
                        for i in range(1, len(wrapped_value)):
                            y_pos += line_height
                            cv2.putText(panel, wrapped_value[i], (left_margin + indent + key_width + 10, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, self.json_font_scale, (0, 100, 0), 1)
                    else:
                        cv2.putText(panel, value_part, (left_margin + indent + key_width + 10, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, self.json_font_scale, (0, 100, 0), 1)
                    
                    y_pos += line_height
                    continue
                else:
                    # Other elements
                    color = (0, 0, 0)  # Black
                    
                cv2.putText(panel, content, (left_margin + indent, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, self.json_font_scale, color, 1)
                y_pos += line_height
                
            # Add a summary box at the bottom 
            if "detailed_description" in analysis:
                # Draw a box for the summary
                summary_top = height - 120
                cv2.rectangle(panel, (10, summary_top), (width-10, height-10), (230, 230, 250), -1)
                cv2.rectangle(panel, (10, summary_top), (width-10, height-10), (70, 70, 70), 1)
                
                # Summary title
                cv2.putText(panel, "SCENE SUMMARY", (20, summary_top + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 1)
                
                # Scene description
                description = analysis.get("detailed_description", "")
                wrapped_desc = textwrap.wrap(description, width=65)
                
                y_text = summary_top + 55
                for line in wrapped_desc[:2]:  # Limit to 2 lines
                    cv2.putText(panel, line, (20, y_text), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    y_text += 25
        
        return panel
    
    def process_video_realtime(self, video_path, use_ads=False, ad_pool=None):
        """
        Process and display video in real-time with analysis sidebar
        
        Args:
            video_path: Path to the video file
            use_ads: Whether to overlay ads based on analysis
            ad_pool: Dictionary of available ads
        """
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Start analysis thread
        self.stop_analysis = False
        analysis_thread = threading.Thread(target=self._analysis_worker, args=(video_path,))
        analysis_thread.daemon = True
        analysis_thread.start()
        
        # Create window for display
        window_name = "Video with Real-time Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Set up the display size (video + analysis panel)
        analysis_width = self.json_panel_width
        combined_width = width + analysis_width
        
        # Resize the window to fit both video and analysis
        cv2.resizeWindow(window_name, combined_width, height)
        
        # Active ads tracking
        active_ads = {}
        
        # For frame timing
        frame_time = 1.0 / fps
        
        while True:
            # Read a frame
            ret, frame = cap.read()
            
            if not ret:
                print("End of video or error reading frame.")
                break
                
            current_time = time.time()
            
            # Check if we have new analysis results
            try:
                while not self.analysis_queue.empty():
                    self.current_analysis = self.analysis_queue.get_nowait()
                    
                    # Save to JSON file
                    with open("current_analysis.json", "w") as f:
                        json.dump(self.current_analysis, f, indent=2)
                        
                    # Print comprehensive analysis to console
                    print("\n" + "="*80)
                    print(f"NEW FRAME ANALYSIS - FRAME {self.current_analysis.get('frame_number')} @ {self.current_analysis.get('video_time')}")
                    print("="*80)
                    
                    # Print GPT-4o's detailed description if available
                    if "gpt4o_analysis" in self.current_analysis:
                        gpt4o_data = self.current_analysis["gpt4o_analysis"]
                        print(f"\nSCENE DESCRIPTION:")
                        print(f"{gpt4o_data.get('Scene Description', '')}")
                        
                        print(f"\nOBJECTS:")
                        if isinstance(gpt4o_data.get('Objects'), list):
                            print(", ".join(gpt4o_data['Objects']))
                        else:
                            print(f"{gpt4o_data.get('Objects', '')}")
                        
                        print(f"\nPEOPLE & ACTIONS:")
                        print(f"{gpt4o_data.get('People and Actions', '')}")
                        
                        print(f"\nEMOTIONAL TONE:")
                        print(f"{gpt4o_data.get('Emotional Tone', '')}")
                    else:
                        # For non-GPT4o analysis
                        if "detailed_description" in self.current_analysis:
                            print(f"\nDescription: {self.current_analysis['detailed_description']}")
                    
                    print("="*80 + "\n")
                        
                    # Check for ad triggers when new analysis arrives
                    if use_ads and ad_pool:
                        # Create a combined text from all GPT-4o analyses
                        combined_text = ""
                        if "gpt4o_analysis" in self.current_analysis:
                            gpt4o_data = self.current_analysis["gpt4o_analysis"]
                            # Combine all GPT-4o analysis fields into one text
                            combined_text = " ".join(str(value).lower() for value in gpt4o_data.values())
                        else:
                            # Fallback for non-GPT4o analysis
                            detected_objects = self.current_analysis.get("key_objects", [])
                            scene_context = self.current_analysis.get("scene_context", "").lower()
                            description = self.current_analysis.get("detailed_description", "").lower()
                            combined_text = description + " " + scene_context + " " + " ".join(detected_objects).lower()
                        
                        print("Processing GPT-4o analysis data")
                        
                        for ad_id, ad_info in ad_pool.items():
                            # Check against all relevant fields
                            keywords = ad_info.get("keywords", [])
                            for keyword in keywords:
                                # Check if keyword appears in the combined text
                                if keyword in combined_text and ad_id not in active_ads:
                                    active_ads[ad_id] = {
                                        "start_time": current_time,
                                        "duration": 5.0,
                                        "ad_path": f"ad_{ad_id}.png",
                                        "triggered_by": keyword
                                    }
                                    print(f"Triggered ad #{ad_id} ({ad_info['metadata']['brand']}) based on keyword: {keyword}")
                                    break
            except queue.Empty:
                pass
                
            # Create analysis panel if we have data
            if self.current_analysis:
                analysis_panel = self._draw_json_panel(
                    self.current_analysis, 
                    self.json_panel_width, 
                    height
                )
            else:
                # Create a blank analysis panel
                analysis_panel = np.ones((height, self.json_panel_width, 3), dtype=np.uint8) * 240
                cv2.putText(analysis_panel, "Waiting for analysis...", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    
            # Show active ads
            for ad_id, ad_info in list(active_ads.items()):
                ad_elapsed = current_time - ad_info["start_time"]
                
                if ad_elapsed <= ad_info["duration"]:
                    # Overlay the ad
                    frame = self._overlay_ad(frame, ad_info["ad_path"])
                else:
                    # Remove expired ad
                    del active_ads[ad_id]
                    print(f"Removed ad #{ad_id}")
            
            # Combine frame and analysis panel
            combined_frame = np.hstack((frame, analysis_panel))
            
            # Show the combined frame
            cv2.imshow(window_name, combined_frame)
            
            # Calculate delay to maintain proper playback speed
            processing_time = time.time() - current_time
            delay = max(1, int((frame_time - processing_time) * 1000))
            
            # Break if 'q' is pressed
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
                
        self.stop_analysis = True
        cap.release()
        cv2.destroyAllWindows()
        
        if analysis_thread.is_alive():
            analysis_thread.join(timeout=1.0)
        
        print("Video processing completed")

# Example ad pool with more detailed triggers
SAMPLE_AD_POOL = {
    1: {
        "id": 1,
        "keywords": ["beach", "water", "ocean", "lake", "swimming", "outdoor", "sunny", "vacation"],
        "metadata": {
            "brand": "SunnyDays Resort",
            "categories": ["travel", "vacation"]
        }
    },
    2: {
        "id": 2,
        "keywords": ["food", "eating", "restaurant", "dinner", "lunch", "meal", "cooking"],
        "metadata": {
            "brand": "FoodieDelight",
            "categories": ["food", "dining"]
        }
    },
    3: {
        "id": 3,
        "keywords": ["running", "sports", "exercise", "fitness", "active", "workout", "playing"],
        "metadata": {
            "brand": "FitLife Athletics",
            "categories": ["sports", "fitness"]
        }
    },
    4: {
        "id": 4,
        "keywords": ["car", "driving", "vehicle", "road", "transportation", "travel"],
        "metadata": {
            "brand": "SpeedyAuto",
            "categories": ["automotive", "transportation"]
        }
    },
    5: {
        "id": 5,
        "keywords": ["technology", "computer", "phone", "device", "digital", "electronic"],
        "metadata": {
            "brand": "TechWorld",
            "categories": ["technology", "electronics"]
        }
    }
}

# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = RealTimeVideoProcessor(api_key="sk-kiuFsJpM40jgti6bl_CNLw")
    
    # Sample video path - replace with actual path
    sample_video_path = "sample_video.mp4"
    
    # Create sample ad images
    print("Creating sample ad images...")
    for ad_id, ad_info in SAMPLE_AD_POOL.items():
        # Create a simple ad for testing
        ad_img = np.ones((150, 300, 3), dtype=np.uint8) * 255  # White background
        
        # Get ad metadata
        ad_brand = ad_info["metadata"]["brand"]
        ad_category = ad_info["metadata"]["categories"][0] if ad_info["metadata"]["categories"] else "general"
        
        # Add text to the ad
        cv2.putText(ad_img, f"{ad_brand}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(ad_img, f"{ad_category}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
        cv2.rectangle(ad_img, (0, 0), (299, 149), (0, 0, 255), 3)
        
        # Save the ad image
        ad_path = f"ad_{ad_id}.png"
        cv2.imwrite(ad_path, ad_img)
        print(f"Created ad image: {ad_path}")
    
    print("\nStarting real-time video processing with LLaVA analysis...")
    print("Press 'q' to exit.")
    
    if os.path.exists(sample_video_path):
        # Process the video with real-time analysis
        processor.process_video_realtime(sample_video_path, use_ads=True, ad_pool=SAMPLE_AD_POOL)
    else:
        print(f"Error: Video file {sample_video_path} not found.")
        print("Please provide a valid video file path.")