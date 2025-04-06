import json
import numpy as np
import os
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from config import *

# Try to import lightfm, but provide a fallback if it's not available
try:
    from lightfm import LightFM
    LIGHTFM_AVAILABLE = True
    print("Successfully imported LightFM")
except ImportError:
    print("Warning: lightfm package not available. Using content-based recommendations only.")
    LIGHTFM_AVAILABLE = False

class AdEngine:
    def __init__(self):
        self.ad_pool = self._load_ads()  # {ad_id: {"image": base64, "metadata": {...}}}
        self.user_profiles = self._load_user_profiles()  # Mock user data
        self.ad_exposure = {}  # Track ad exposure per user to prevent fatigue
        self.lightfm_model = None
        self.use_collaborative = LIGHTFM_AVAILABLE
        self.engagement_history = {}  # Store engagement metrics for A/B testing
        self.micro_moment_cache = {}  # Cache for detected micro-moments
        
        # Training data if available
        self._init_models()

    def _load_ads(self):
        """Load ad inventory - extended with richer metadata"""
        try:
            with open("ad_inventory.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Return mock data if file not found
            return {
                101: {
                    "metadata": {
                        "categories": ["sports", "energy"], 
                        "target_age": [18, 35],
                        "target_gender": "all",
                        "keywords": ["active", "fitness", "energy", "performance"],
                        "brand": "SportsFuel",
                        "duration": 5,  # seconds - shorter for micro-moments
                        "format": "video"
                    }
                },
                102: {
                    "metadata": {
                        "categories": ["luxury", "automotive"], 
                        "target_age": [25, 50],
                        "target_gender": "all",
                        "keywords": ["premium", "luxury", "driving", "performance"],
                        "brand": "LuxMotors",
                        "duration": 4,  # seconds - shorter for micro-moments
                        "format": "video"
                    }
                },
                103: {
                    "metadata": {
                        "categories": ["travel", "beach", "vacation"], 
                        "target_age": [20, 60],
                        "target_gender": "all",
                        "keywords": ["beach", "tropical", "relaxation", "resort"],
                        "brand": "SunEscape",
                        "duration": 3,  # seconds - shorter for micro-moments
                        "format": "video"
                    }
                },
                104: {
                    "metadata": {
                        "categories": ["outdoor", "apparel"], 
                        "target_age": [18, 45],
                        "target_gender": "all",
                        "keywords": ["adventure", "hiking", "nature", "gear"],
                        "brand": "WildTrail",
                        "duration": 3,  # seconds - shorter for micro-moments
                        "format": "video"
                    }
                },
                105: {
                    "metadata": {
                        "categories": ["wellness", "health"], 
                        "target_age": [25, 65],
                        "target_gender": "all",
                        "keywords": ["wellness", "mindfulness", "health", "balance"],
                        "brand": "ZenLife",
                        "duration": 5,  # seconds - shorter for micro-moments
                        "format": "video"
                    }
                }
            }

    def _load_user_profiles(self):
        """Load user profiles - with viewing history and preferences"""
        try:
            with open("user_profiles.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Return mock data if file not found
            return {
                "user1": {
                    "demographics": {
                        "age": 28,
                        "gender": "female",
                        "location": "urban"
                    },
                    "interests": ["travel", "fitness", "technology"],
                    "view_history": [103, 101, 104],
                    "engagement_rate": 0.35  # Click-through rate
                },
                "user2": {
                    "demographics": {
                        "age": 42,
                        "gender": "male",
                        "location": "suburban"
                    },
                    "interests": ["automotive", "sports", "finance"],
                    "view_history": [102, 101],
                    "engagement_rate": 0.28
                }
            }

    def _init_models(self):
        """Initialize recommendation models"""
        try:
            # Try to load interaction data and train model
            with open("interaction_history.json", "r") as f:
                interactions = json.load(f)
                if self.use_collaborative:
                    print("Training collaborative filtering model...")
                    self.train_lightfm(interactions)
        except FileNotFoundError:
            print("No interaction history found. Starting with content-based only.")
    
    def train_lightfm(self, interactions):
        """Train hybrid recommender"""
        if not LIGHTFM_AVAILABLE:
            print("LightFM not available. Skipping collaborative filtering training.")
            return False
            
        # Convert to sparse matrix
        interactions_matrix = self._create_interaction_matrix(interactions)
        self.lightfm_model = LightFM(loss='warp')
        self.lightfm_model.fit(interactions_matrix, epochs=20)
        return True
    
    def _create_interaction_matrix(self, interactions):
        """Helper for LightFM - mocked implementation"""
        # In a real system, you would create a proper sparse matrix
        # For now, we'll return a mock matrix if LightFM is available
        if LIGHTFM_AVAILABLE:
            return np.ones((len(self.user_profiles), len(self.ad_pool)))
        return None
    
    def match_ads(self, scene_metadata, user_id=None):
        """Rank ads based on scene context + user profile"""
        # Content-based similarity
        scene_features = self._extract_features(scene_metadata)
        ad_features = np.array([self._extract_features(ad["metadata"]) 
                               for ad in self.ad_pool.values()])
        
        # Calculate similarity scores
        if len(ad_features) > 0:
            content_scores = cosine_similarity([scene_features], ad_features)[0]
        else:
            # Handle empty ad pool
            return []
        
        # Initialize with content scores
        final_scores = content_scores.copy()
        
        # Apply collaborative filtering if available
        if user_id and self.lightfm_model and LIGHTFM_AVAILABLE:
            try:
                user_idx = list(self.user_profiles.keys()).index(user_id)
                item_indices = list(range(len(self.ad_pool)))
                user_scores = self.lightfm_model.predict(user_idx, item_indices)
                final_scores = 0.7*content_scores + 0.3*user_scores
            except Exception as e:
                print(f"Error applying collaborative filtering: {e}")
                # Fall back to content-based scores
        
        # Return sorted ads
        ranked_ads = sorted(zip(self.ad_pool.keys(), final_scores), 
                          key=lambda x: x[1], reverse=True)
        
        return ranked_ads[:3]  # Top 3 ads
    
    def _extract_features(self, metadata):
        """Convert metadata to feature vector (simple implementation)"""
        # This is a simplified feature extractor that works with scene metadata
        # In a real implementation, you'd use more sophisticated feature engineering
        
        feature_vector = np.zeros(15)  # Base feature vector
        
        # Process categories
        if "categories" in metadata:
            for category in metadata["categories"]:
                if category in ["sports", "energy"]:
                    feature_vector[0] = 1
                if category in ["luxury", "automotive"]:
                    feature_vector[1] = 1
                if category in ["beach", "travel", "vacation"]:
                    feature_vector[2] = 1
                if category in ["outdoor", "apparel"]:
                    feature_vector[3] = 1
                if category in ["wellness", "health", "fitness"]:
                    feature_vector[4] = 1
        
        # Process scene context
        if "scene_context" in metadata:
            context = metadata["scene_context"].lower()
            if any(word in context for word in ["beach", "ocean", "sea", "water"]):
                feature_vector[5] = 1
            if any(word in context for word in ["city", "urban"]):
                feature_vector[6] = 1
            if any(word in context for word in ["mountain", "outdoor", "nature"]):
                feature_vector[7] = 1
        
        # Process emotional tone
        if "emotional_tone" in metadata:
            tone = metadata["emotional_tone"].lower()
            if any(word in tone for word in ["peaceful", "calm", "serene"]):
                feature_vector[8] = 1
            if any(word in tone for word in ["exciting", "energetic"]):
                feature_vector[9] = 1
            if any(word in tone for word in ["luxury", "premium", "elegant"]):
                feature_vector[10] = 1
        
        # Add some random noise for variety
        feature_vector[11:15] = np.random.rand(4) * 0.2
        
        return feature_vector
    
    def find_optimal_placement(self, scene_data, video_duration):
        """
        Determine optimal ad placement timings within a video
        
        Args:
            scene_data: List of scene analysis information
            video_duration: Total video duration in seconds
            
        Returns:
            List of timestamps (seconds) for optimal ad placement
        """
        # Avoid placing ads in very short videos
        if video_duration < 30:
            return []
            
        placements = []
        
        # Get scene boundaries and find natural breaks
        scene_boundaries = [(scene["time_start"], scene["time_end"]) for scene in scene_data]
        
        # If only one scene detected, create artificial breakpoints
        if len(scene_boundaries) <= 1:
            # Create a placement every 1/4 of the video
            return [video_duration * 0.25, video_duration * 0.5, video_duration * 0.75]
        
        # Find scenes with lower engagement potential (e.g., scene transitions)
        for i, (scene_start, scene_end) in enumerate(scene_boundaries):
            # Skip first and last 10% of video
            if scene_start < video_duration * 0.1 or scene_end > video_duration * 0.9:
                continue
                
            # Look for scene transitions
            if i > 0:
                prev_end = scene_boundaries[i-1][1]
                transition_length = scene_start - prev_end
                
                # If there's a natural break between scenes
                if transition_length > 0.5:  # More than 0.5 second gap
                    placements.append(prev_end)
            
            # For longer scenes, consider mid-scene placement if low action
            scene_duration = scene_end - scene_start
            if scene_duration > 30:  # For scenes longer than 30 seconds
                scene_metadata = scene_data[i]["metadata"]["response"]
                
                # Check if scene has low action/movement
                is_low_action = False
                if "emotional_tone" in scene_metadata:
                    tone = scene_metadata["emotional_tone"].lower()
                    if any(word in tone for word in ["peaceful", "calm", "serene", "static"]):
                        is_low_action = True
                
                if is_low_action:
                    mid_point = scene_start + (scene_duration / 2)
                    placements.append(mid_point)
        
        # Ensure minimum spacing between ads (at least 20 seconds)
        filtered_placements = []
        min_spacing = 20  # seconds
        
        for placement in sorted(placements):
            if not filtered_placements or placement - filtered_placements[-1] >= min_spacing:
                filtered_placements.append(placement)
        
        return filtered_placements

    def detect_micro_moments(self, scene_data):
        """
        Detect micro-moments within scenes that are optimal for short-form ads
        
        Args:
            scene_data: List of scene analysis information
            
        Returns:
            List of micro-moments with timestamps and context
        """
        # Check if we have cached results
        scene_ids = tuple(scene["scene_id"] for scene in scene_data)
        if scene_ids in self.micro_moment_cache:
            print("Using cached micro-moment detection results")
            return self.micro_moment_cache[scene_ids]
        
        micro_moments = []
        
        for scene in scene_data:
            scene_id = scene["scene_id"]
            scene_start = scene["time_start"]
            scene_end = scene["time_end"]
            duration = scene["duration"]
            
            # Get scene metadata - handle different possible formats
            if "metadata" in scene and "response" in scene["metadata"]:
                metadata = scene["metadata"]["response"]
            elif "metadata" in scene:
                metadata = scene["metadata"]
            else:
                # If no proper metadata, skip this scene
                print(f"Warning: Scene {scene_id} has no usable metadata")
                continue
            
            # Skip very short scenes (less than 5 seconds)
            if duration < 3:
                continue
                
            # 1. Emotional peaks or transitions
            if "emotional_tone" in metadata:
                tone = metadata["emotional_tone"].lower()
                
                # Emotional peaks are good moments for ads that match the emotion
                if any(emotion in tone for emotion in ["exciting", "joyful", "surprising", "heartwarming"]):
                    # Place micro-moment shortly after the emotional peak (1/3 into the scene)
                    moment_time = scene_start + (duration / 3)
                    micro_moments.append({
                        "time": moment_time,
                        "type": "emotional_peak",
                        "emotion": tone,
                        "scene_id": scene_id,
                        "duration_recommendation": 3  # Short ads for emotional peaks
                    })
                
                # Calm moments are good for longer, informative ads
                elif any(emotion in tone for emotion in ["calm", "peaceful", "relaxed"]):
                    # Place micro-moment in the middle of a calm scene
                    moment_time = scene_start + (duration / 2)
                    micro_moments.append({
                        "time": moment_time,
                        "type": "calm_moment",
                        "emotion": tone,
                        "scene_id": scene_id,
                        "duration_recommendation": 5  # Slightly longer ads for calm moments
                    })
            
            # 2. Action completions (good moments for transitions)
            if "activity" in metadata:
                activity = metadata["activity"].lower()
                
                # After an action completes is a natural transition point
                if any(action in activity for action in ["completing", "finishing", "ending", "achieved"]):
                    moment_time = scene_end - 1  # 1 second before scene end
                    micro_moments.append({
                        "time": moment_time,
                        "type": "action_completion",
                        "activity": activity,
                        "scene_id": scene_id,
                        "duration_recommendation": 4  # Medium length for transitions
                    })
            
            # 3. Object-focused moments (good for product overlays)
            if "key_objects" in metadata and len(metadata["key_objects"]) > 0:
                # Product/object focused moment in the middle of the scene
                moment_time = scene_start + (duration / 2)
                micro_moments.append({
                    "time": moment_time,
                    "type": "object_focus",
                    "objects": metadata["key_objects"],
                    "scene_id": scene_id,
                    "duration_recommendation": 3  # Short overlay ads
                })
        
        # Filter for minimum spacing between micro-moments (at least 10 seconds)
        filtered_moments = []
        min_spacing = 10  # seconds
        
        # If no moments detected, create artificial ones
        if not micro_moments and len(scene_data) > 0:
            # Create artificial moments at regular intervals
            video_end = max([scene["time_end"] for scene in scene_data])
            interval = min(15, video_end / 3)  # Every 15 seconds or 1/3 of video
            
            for t in np.arange(interval, video_end - 5, interval):
                micro_moments.append({
                    "time": t,
                    "type": "regular_interval",
                    "scene_id": 1,  # Default scene
                    "duration_recommendation": 3  # Short default duration
                })
            print(f"No natural micro-moments detected. Created {len(micro_moments)} artificial ones.")
        
        for moment in sorted(micro_moments, key=lambda x: x["time"]):
            if not filtered_moments or moment["time"] - filtered_moments[-1]["time"] >= min_spacing:
                filtered_moments.append(moment)
        
        # Cache results
        self.micro_moment_cache[scene_ids] = filtered_moments
        
        return filtered_moments

    def match_micro_moment_ads(self, micro_moment, user_id=None):
        """
        Match ultra-short ads to specific micro-moments
        
        Args:
            micro_moment: Dict containing micro-moment data
            user_id: Optional user ID for personalization
            
        Returns:
            List of matched ads with scores
        """
        # Create a specialized scene context based on the micro-moment type
        micro_context = {
            "scene_context": "",
            "emotional_tone": micro_moment.get("emotion", "neutral"),
            "activity": micro_moment.get("activity", ""),
            "key_objects": micro_moment.get("objects", []),
        }
        
        # Adjust context based on micro-moment type
        if micro_moment["type"] == "emotional_peak":
            micro_context["scene_context"] = f"Emotional moment: {micro_moment.get('emotion', 'exciting')}"
            # Boost emotional resonance for these moments
            
        elif micro_moment["type"] == "calm_moment":
            micro_context["scene_context"] = f"Calm moment suitable for informative content"
            # Calm moments work well with more detailed info
            
        elif micro_moment["type"] == "action_completion":
            micro_context["scene_context"] = f"Transition after activity: {micro_moment.get('activity', 'completing')}"
            # Action completions work well with next-step suggestions
            
        elif micro_moment["type"] == "object_focus":
            objects = micro_moment.get("objects", ["product"])
            if isinstance(objects, list) and len(objects) > 0:
                objects_str = ", ".join(objects)
            else:
                objects_str = "products"
            micro_context["scene_context"] = f"Focus on objects: {objects_str}"
            # Object focus works well with product highlights
            
        elif micro_moment["type"] == "regular_interval":
            micro_context["scene_context"] = "Regular interval placement"
            micro_context["emotional_tone"] = "neutral"
        
        # Get standard ad matches
        all_matches = self.match_ads(micro_context, user_id)
        
        # If no matches, return empty list
        if not all_matches:
            return []
            
        # Filter for ads that match the recommended duration
        recommended_duration = micro_moment.get("duration_recommendation", 5)
        filtered_matches = []
        
        for ad_id, score in all_matches:
            # Convert string ad_id to int if needed
            if isinstance(ad_id, str) and ad_id.isdigit():
                ad_id = int(ad_id)
                
            # Get ad duration with fallback
            try:
                ad_duration = self.ad_pool[ad_id]["metadata"].get("duration", 15)
            except (KeyError, TypeError):
                # If ad_id not found in pool, use default duration
                ad_duration = 15
                
            # Adjust score based on how well the ad duration matches the micro-moment
            duration_match = 1.0 - (abs(ad_duration - recommended_duration) / 10)
            if duration_match < 0:
                duration_match = 0.1  # Minimum score
                
            adjusted_score = score * duration_match
            filtered_matches.append((ad_id, adjusted_score))
        
        # Re-sort after duration adjustment
        filtered_matches.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_matches[:2]  # Return top 2 matches for the micro-moment

    def generate_micro_moment_ad_plan(self, video_analysis, user_id=None):
        """
        Generate a complete plan for micro-moment ad insertions
        
        Args:
            video_analysis: Complete scene analysis
            user_id: Optional user ID for personalization
            
        Returns:
            Ad plan with timestamps and ad details
        """
        # Detect all micro-moments
        micro_moments = self.detect_micro_moments(video_analysis)
        
        print(f"Detected {len(micro_moments)} micro-moments in the video")
        
        # Create ad plan
        ad_plan = []
        
        for moment in micro_moments:
            # Get ad matches for this micro-moment
            ad_matches = self.match_micro_moment_ads(moment, user_id)
            
            if ad_matches:
                # Use top match
                ad_id, score = ad_matches[0]
                
                # Ensure ad_id is an integer
                if isinstance(ad_id, str) and ad_id.isdigit():
                    ad_id = int(ad_id)
                    
                # Get ad metadata safely
                try:
                    ad_metadata = self.ad_pool[ad_id]["metadata"]
                except (KeyError, TypeError):
                    print(f"Warning: Ad #{ad_id} not found in ad pool. Using placeholder.")
                    ad_metadata = {
                        "brand": "Default",
                        "categories": ["general"],
                        "duration": 5
                    }
                
                # Add to plan
                ad_plan.append({
                    "timestamp": moment["time"],
                    "ad_id": ad_id,
                    "score": float(score),
                    "micro_moment_type": moment["type"],
                    "recommended_duration": moment.get("duration_recommendation", 5),
                    "ad_metadata": ad_metadata
                })
        
        print(f"Generated ad plan with {len(ad_plan)} insertions")
        return ad_plan

    def record_interaction(self, user_id, ad_id, interaction_strength):
        """
        Record user-ad interaction to improve future recommendations
        
        Args:
            user_id: User identifier
            ad_id: Ad identifier 
            interaction_strength: Strength of interaction (0-1)
                - 1.0: Click/full engagement
                - 0.5: Viewed but no click
                - 0.1: Dismissed/skipped
        """
        try:
            # Load existing interaction data
            try:
                with open("interaction_history.json", "r") as f:
                    interactions = json.load(f)
            except FileNotFoundError:
                interactions = {}
            
            # Add or update interaction
            if user_id not in interactions:
                interactions[user_id] = {}
            
            # Convert ad_id to string for JSON
            ad_id_str = str(ad_id)
            interactions[user_id][ad_id_str] = interaction_strength
            
            # Save updated data
            with open("interaction_history.json", "w") as f:
                json.dump(interactions, f, indent=2)
            
            print(f"Recorded interaction: User {user_id}, Ad {ad_id}, Strength {interaction_strength}")
            
            # Periodically retrain model (e.g., after every 10 new interactions)
            total_interactions = sum(len(user_data) for user_data in interactions.values())
            if self.use_collaborative and total_interactions % 10 == 0:
                print(f"Accumulated {total_interactions} interactions. Retraining model...")
                self.train_lightfm(interactions)
                
        except Exception as e:
            print(f"Error recording interaction: {e}")
            
# Example usage and testing
if __name__ == "__main__":
    # Create an example interaction history if it doesn't exist
    if not os.path.exists("interaction_history.json"):
        example_interactions = {
            "user1": {
                "101": 0.8,
                "103": 1.0,
                "104": 0.5
            },
            "user2": {
                "102": 0.9,
                "101": 0.3
            }
        }
        with open("interaction_history.json", "w") as f:
            json.dump(example_interactions, f, indent=2)
        print("Created example interaction history file")
        
    # Create the engine
    engine = AdEngine()
    
    # Load the video analysis if available
    try:
        with open("video_analysis.json", "r") as f:
            video_analysis = json.load(f)
        
        print("\n===== AD OPTIMIZER ANALYSIS =====\n")
        print(f"Video contains {len(video_analysis)} scenes")
        
        # Find optimal ad placement
        video_duration = max([scene["time_end"] for scene in video_analysis])
        placement_times = engine.find_optimal_placement(video_analysis, video_duration)
        
        print(f"\nOptimal ad placement times: {', '.join([f'{t:.1f}s' for t in placement_times])}")
        
        print("\n===== MICRO-MOMENT ANALYSIS =====")
        # Detect micro-moments
        micro_moments = engine.detect_micro_moments(video_analysis)
        print(f"Detected {len(micro_moments)} micro-moments for ultra-short ads:")
        for i, moment in enumerate(micro_moments):
            print(f"  {i+1}. Type: {moment['type']} at {moment['time']:.1f}s - " + 
                  f"Recommended duration: {moment.get('duration_recommendation', 5)}s")
        
        # Generate ad plan for micro-moments
        print("\n===== MICRO-MOMENT AD PLAN =====")
        ad_plan = engine.generate_micro_moment_ad_plan(video_analysis)
        print(f"Generated ad plan with {len(ad_plan)} insertions:")
        for i, insertion in enumerate(ad_plan):
            ad_metadata = insertion["ad_metadata"]
            print(f"  {i+1}. At {insertion['timestamp']:.1f}s: {ad_metadata['brand']} " +
                  f"({ad_metadata['categories'][0]}) - {insertion['micro_moment_type']} - " +
                  f"Score: {insertion['score']:.2f}")
            
    except FileNotFoundError:
        print("Error: video_analysis.json not found. Run video processing first.")