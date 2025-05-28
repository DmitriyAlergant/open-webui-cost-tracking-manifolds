#!/usr/bin/env python3
import os
import sys
import json
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv
import time

def load_environment(env_file=None):
    """Load environment variables from .env file"""
    if env_file:
        env_path = Path(env_file)
        if not env_path.exists():
            print(f"❌ Environment file {env_file} not found")
            sys.exit(1)
    else:
        env_path = Path.cwd() / '.env'
    
    load_dotenv(dotenv_path=env_path, override=True)
    
    api_url = os.getenv('OPENWEBUI_API_URL', 'http://localhost:3000')
    api_key = os.getenv('OPENWEBUI_API_KEY')
    
    if not api_key:
        print("❌ OPENWEBUI_API_KEY not found in environment")
        sys.exit(1)
    
    return api_url, api_key

def get_available_models(api_url, api_key):
    """Fetch list of available models from OpenWebUI API"""
    url = f"{api_url}/api/models"
    headers = {'Authorization': f'Bearer {api_key}'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        models_data = response.json()
        
        # Debug: print the response structure
        print(f"🔍 Models API response type: {type(models_data)}")
        if isinstance(models_data, dict):
            print(f"🔍 Response keys: {list(models_data.keys())}")
        
        # Extract model IDs from the response
        models = []
        if isinstance(models_data, dict) and 'data' in models_data:
            models = [model['id'] for model in models_data['data']]
        elif isinstance(models_data, list):
            models = [model['id'] if isinstance(model, dict) else model for model in models_data]
        elif isinstance(models_data, dict) and 'models' in models_data:
            # Alternative structure
            models = [model['id'] if isinstance(model, dict) else model for model in models_data['models']]
        else:
            print(f"❌ Unexpected response format: {type(models_data)}")
            print(f"🔍 Response content: {json.dumps(models_data, indent=2)[:500]}...")
            return []
        
        print(f"✅ Found {len(models)} available models")
        return models
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch models: {e}")
        return []

def test_model_chat_completion(api_url, api_key, model_id):
    """Test a single model with a chat completion request"""
    url = f"{api_url}/api/chat/completions"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": "How many colors are in a rainbow? Respond simply. This is a model health check."
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the response content - try multiple possible structures
        content = None
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                content = choice['message']['content']
            elif 'text' in choice:
                content = choice['text']
        elif 'message' in result and 'content' in result['message']:
            content = result['message']['content']
        elif 'response' in result:
            content = result['response']
        elif 'text' in result:
            content = result['text']
        
        if content:
            print(f"✅ {model_id} is working")
            return True
        else:
            print(f"❌ {model_id}: No response content received")
            print(f"🔍 Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            return False
    
    except requests.exceptions.Timeout:
        print(f"⏰ {model_id}: Request timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {model_id}: Request failed - {e}")
        return False
    except json.JSONDecodeError:
        print(f"❌ {model_id}: Invalid JSON response")
        return False
    except Exception as e:
        print(f"❌ {model_id}: Unexpected error - {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test all available OpenWebUI models with chat completions')
    parser.add_argument('-e', '--env', help='Path to .env file', default=None)
    
    args = parser.parse_args()
    
    # Load environment variables
    api_url, api_key = load_environment(args.env)
    
    print(f"🚀 Testing models on {api_url}")
    print("=" * 60)
    
    # Get available models
    models = get_available_models(api_url, api_key)
    
    if not models:
        print("❌ No models found to test")
        sys.exit(1)
    
    # Test each model
    successful_tests = 0
    failed_tests = 0
    
    for i, model_id in enumerate(models, 1):
        print(f"[{i}/{len(models)}] Testing {model_id}...")
        
        success = test_model_chat_completion(api_url, api_key, model_id)
        
        if success:
            successful_tests += 1
        else:
            failed_tests += 1
        
        # Small delay between requests to be respectful
        if i < len(models):
            time.sleep(1)
    
    # Summary
    print("=" * 60)
    print(f"🎯 Test Summary:")
    print(f"✅ Successful: {successful_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📊 Total: {len(models)}")
    
    if failed_tests == 0:
        print("🎉 All models tested successfully!")
    elif successful_tests > 0:
        print("⚠️  Some models failed testing")
    else:
        print("💥 All model tests failed")

if __name__ == "__main__":
    main() 