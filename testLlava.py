# At the beginning of your file
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

# Debug flag
DEBUG_MODE = True

# Check for required dependencies and set flags
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
USE_LLAVA = False
USE_CLASSIFIER = False
USE_PRETRAINED = False

# Try importing torch first
try:
    import torch
    TORCH_AVAILABLE = True
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"Error importing PyTorch: {e}")
    print("Please install PyTorch: pip install torch torchvision")

# Try importing transformers only if torch is available
if TORCH_AVAILABLE:
    try:
        from transformers import AutoProcessor, AutoModelForVisionText2Text
        TRANSFORMERS_AVAILABLE = True
        
        # Load LLaVA model with better error handling
        try:
            print("Loading LLaVA model - this may take a few minutes...")
            model_name = "llava-hf/llava-1.5-7b-hf"
            
            llava_processor = AutoProcessor.from_pretrained(model_name)
            print("LLaVA processor loaded successfully")
            
            llava_model = AutoModelForVisionText2Text.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("LLaVA model loaded successfully")
            
            def analyze_with_llava(image, prompt="Describe this image in detail."):
                """Generate comprehensive scene analysis using LLaVA model"""
                try:
                    if DEBUG_MODE:
                        print(f"Analyzing with prompt: '{prompt[:30]}...'")
                    
                    # Process image and text inputs
                    inputs = llava_processor(
                        text=prompt,
                        images=image,
                        return_tensors="pt"
                    ).to(llava_model.device)
                    
                    # Generate output text
                    output = llava_model.generate(
                        **inputs,
                        max_new_tokens=100,  # Reduced for faster analysis
                        do_sample=False
                    )
                    
                    # Decode the output
                    result = llava_processor.decode(output[0], skip_special_tokens=True)
                    
                    if DEBUG_MODE:
                        print(f"LLaVA result preview: '{result[:50]}...'")
                    return result
                except Exception as e:
                    error_msg = f"Error in analyze_with_llava: {e}"
                    print(error_msg)
                    if DEBUG_MODE:
                        traceback.print_exc()
                    return f"Analysis error: {str(e)}"
            
            USE_LLAVA = True
            print("✅ LLaVA loaded successfully and ready for analysis")
            
            # Also try to load classifier if LLaVA worked
            try:
                from transformers import pipeline
                classifier = pipeline("image-classification")
                USE_CLASSIFIER = True
                print("Loaded classifier for basic image classification")
            except Exception as e:
                print(f"Could not load classifier: {e}")
                USE_CLASSIFIER = False
                
        except Exception as e:
            print(f"❌ Error loading LLaVA model: {e}")
            if DEBUG_MODE:
                traceback.print_exc()
    
    except ImportError as e:
        print(f"Error importing transformers: {e}")
        print("Please install transformers: pip install transformers")
# Rest of your code...