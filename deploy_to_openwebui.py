import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment variables
API_URL = os.getenv('OPENWEBUI_API_URL', 'http://localhost:3000')
API_KEY = os.getenv('OPENWEBUI_API_KEY')

# Define required metadata attributes
REQUIRED_METADATA = ['name', 'description', 'title', 'author', 'version']

def extract_metadata(content):
    lines = content.split('\n')
    metadata = {}
    if lines[0].strip() == '"""':
        for line in lines[1:]:
            if line.strip() == '"""':
                break
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
    return metadata

def validate_metadata(metadata):
    missing = [attr for attr in REQUIRED_METADATA if attr not in metadata]
    if missing:
        raise ValueError(f"Missing required metadata: {', '.join(missing)}")

def deploy_function(filename, token):
    with open(filename, 'r') as file:
        content = file.read()
    
    metadata = extract_metadata(content)
    validate_metadata(metadata)
    function_id = Path(filename).stem

    url = f"{API_URL}/api/v1/functions/id/{function_id}/update"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    data = {
        "id": function_id,
        #"name": metadata.get('name', function_id),
        "meta": {
            "description": metadata.get('description', ''),
            "manifest": {
                "title": metadata.get('title', ''),
                "author": metadata.get('author', ''),
                "author_url": metadata.get('author_url', ''),
                "funding_url": metadata.get('funding_url', ''),
                "version": metadata.get('version', '')
            }
        },
        "content": content
    }

    print(json.dumps(data, indent=2))

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python deploy_script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    
    try:
        token = API_KEY
        result = deploy_function(filename, token)
        print(f"Function deployed successfully: {result}")
    except ValueError as e:
        print(f"Metadata validation error: {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error deploying function: {e}")
        sys.exit(1)
