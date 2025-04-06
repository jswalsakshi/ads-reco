import os
import requests
import json

# Set environment variables within the script
# (In a real project, you would set these outside the script)
os.environ["LITELLM_API_KEY"] = "sk-kiuFsJpM40jgti6bl_CNLw"  # Replace with your actual API key
os.environ["OPENAI_API_BASE"] = "https://api.rabbithole.cred.club/v1"

# Print current environment variables to verify they're set
print(f"API Key set: {'LITELLM_API_KEY' in os.environ}")
print(f"API Base set: {'OPENAI_API_BASE' in os.environ}")

# Get the values (masking the key for security)
api_key = os.environ.get("LITELLM_API_KEY", "")
api_base = os.environ.get("OPENAI_API_BASE", "")

if api_key and api_base:
    # Mask key for display
    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "****"
    print(f"API Key: {masked_key}")
    print(f"API Base: {api_base}")

    # Simple test request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "claude-3-7-sonnet",  # Using one of the available models
        "messages": [
            {"role": "user", "content": "Hello, are you working correctly?"}
        ]
    }
    
    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            print("\n✅ Configuration successful! API response received.")
            response_data = response.json()
            # Print a snippet of the response
            content = response_data["choices"][0]["message"]["content"]
            print(f"\nResponse snippet: {content[:100]}...")
        else:
            print(f"\n❌ API request failed with status code {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"\n❌ Exception occurred: {str(e)}")
else:
    print("❌ Environment variables not set correctly")