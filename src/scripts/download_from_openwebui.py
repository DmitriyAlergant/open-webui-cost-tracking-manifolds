#!/usr/bin/env python3
"""
Download functions from OpenWebUI API and save them locally.
Usage: python src/scripts/download_from_openwebui.py [function_id ...]

If no function IDs provided, downloads all functions listed in the script.
"""
import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path=env_path, override=True)

API_URL = os.getenv('OPENWEBUI_API_URL', 'http://localhost:3000')
API_KEY = os.getenv('OPENWEBUI_API_KEY')

# Default functions to download (from upload_gb_llm_functions.sh)
DEFAULT_FUNCTIONS = [
    "module_usage_tracking",
    "module_litellm_pipe",
    "module_fallback_router",
    "usage_reporting_bot",
    "module_usage_tracking_pricing_data",
    "aimediator",
    "proxyapi",
    "yandexgpt",
    "gemini",
    "deepseek",
    "openai",
    "anthropic",
]

def download_function(function_id: str, token: str) -> dict | None:
    """Download a single function from OpenWebUI API."""
    url = f"{API_URL}/api/v1/functions/id/{function_id}"
    headers = {'Authorization': f'Bearer {token}'}

    response = requests.get(url, headers=headers)

    if response.status_code == 404:
        print(f"NOT FOUND: {function_id}")
        return None
    elif response.status_code == 401:
        print(f"UNAUTHORIZED: {function_id}")
        return None

    response.raise_for_status()
    return response.json()

def save_function(function_data: dict, output_dir: Path) -> Path:
    """Save function content to a file."""
    function_id = function_data['id']
    content = function_data.get('content', '')

    output_file = output_dir / f"{function_id}.py"
    output_file.write_text(content)
    return output_file

def main():
    if not API_KEY:
        print("ERROR: OPENWEBUI_API_KEY not set")
        sys.exit(1)

    # Get function IDs from args or use defaults
    function_ids = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_FUNCTIONS

    # Output directory
    output_dir = Path("downloaded_functions")
    output_dir.mkdir(exist_ok=True)

    print(f"Downloading from {API_URL}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    downloaded = 0
    for func_id in function_ids:
        try:
            data = download_function(func_id, API_KEY)
            if data and data.get('content'):
                output_file = save_function(data, output_dir)
                print(f"OK: {func_id} -> {output_file}")
                downloaded += 1
            elif data:
                print(f"EMPTY: {func_id} (no content)")
        except requests.exceptions.RequestException as e:
            print(f"ERROR: {func_id} - {e}")

    print("-" * 50)
    print(f"Downloaded {downloaded}/{len(function_ids)} functions")

if __name__ == "__main__":
    main()
