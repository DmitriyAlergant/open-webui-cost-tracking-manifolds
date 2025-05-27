#!/usr/bin/env python3
"""
OpenWebUI Model Update Script

This script automatically creates and updates models in OpenWebUI based on the
AVAILABLE_MODELS defined in function modules (openai, anthropic, etc.).

Features:
- Reads AVAILABLE_MODELS from specified function modules
- Maps model names to appropriate provider logos
- Sets system prompts from environment variables
- Creates or updates models in OpenWebUI via API
- Handles both create and update operations automatically
- Supports custom environment files via -e flag
- Supports logo overrides via environment variables
- Supports submitting model order configuration via --model-order-file flag
- Validates model IDs against pricing data module via --pricing-module flag

Environment Variables Required:
- OPENWEBUI_API_URL: OpenWebUI instance URL (default: http://localhost:3000)
- OPENWEBUI_API_KEY: OpenWebUI API token
- SYSTEM_PROMPT: Default system prompt for all models (optional)

Logo Override Environment Variables (optional):
- LOGO_OVERRIDES: JSON object mapping modules/models to logo paths
  Example: LOGO_OVERRIDES='{"openrouter": "./images/logo_backup.png", "openrouter.gpt-4": "./custom_gpt4.png"}'
- LOGO_OVERRIDE_<MODULE>: Individual logo override for a specific module
  Example: LOGO_OVERRIDE_OPENROUTER="./images/logo_backup.png"

Model Order File Format (JSON):
{
  "DEFAULT_MODELS": "module.model-name",
  "MODEL_ORDER_LIST": [
    "module.model1",
    "module.model2",
    ...
  ]
}

Pricing Validation:
When --pricing-module is specified, the script validates that all model IDs from
the imported modules exist in the pricing_data dictionary of the specified module.
If any model IDs are missing, the script stops execution and reports the missing models.

Usage:
    python update_models_in_openwebui.py openai
    python update_models_in_openwebui.py openai anthropic
    python update_models_in_openwebui.py -e .env.production openai
    python update_models_in_openwebui.py --env-file /path/to/custom.env openai
    python update_models_in_openwebui.py --model-order-file ./model_order.json openai
    python update_models_in_openwebui.py --pricing-module ./src/functions/pricing-data-module/module_usage_tracking_pricing_data.py openai
"""

import os
import sys
import json
import requests
import base64
import importlib.util
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

def load_environment(env_file: Optional[str] = None) -> None:
    """Load environment variables from specified file or default .env"""
    if env_file:
        env_path = env_file
        if not os.path.exists(env_path):
            print(f"Error: Environment file {env_path} not found")
            sys.exit(1)
    else:
        env_path = os.path.join(os.getcwd(), '.env')
    
    load_dotenv(dotenv_path=env_path, override=True)
    if env_file:
        print(f"📄 Loaded environment from: {env_path}")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update models in OpenWebUI based on function modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python update_models_in_openwebui.py openai
    python update_models_in_openwebui.py anthropic
    python update_models_in_openwebui.py -e .env.production openai
    python update_models_in_openwebui.py --model-order-file ./model_order_list.json openai
    python update_models_in_openwebui.py --pricing-module ./src/functions/pricing-data-module/module_usage_tracking_pricing_data.py openai
        """
    )
    
    parser.add_argument(
        '-e', '--env-file',
        type=str,
        help='Path to environment file (default: .env in current directory)'
    )
    
    parser.add_argument(
        '--model-order-file',
        type=str,
        help='Path to JSON file containing model order configuration to submit to OpenWebUI'
    )
    
    parser.add_argument(
        '--pricing-module',
        type=str,
        help='Path to pricing data module to validate model IDs against'
    )
    
    parser.add_argument(
        'modules',
        nargs='+',
        help='Module names to process (e.g., openai, anthropic)'
    )
    
    return parser.parse_args()

# Parse arguments and load environment
args = parse_arguments()
load_environment(args.env_file)

# Configuration
API_URL = os.getenv('OPENWEBUI_API_URL', 'http://localhost:3000')
API_KEY = os.getenv('OPENWEBUI_API_KEY')
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', '')

# Logo mapping based on model ID patterns
LOGO_MAPPING = {
    'openai': 'logo_openai.png',
    'anthropic': 'logo_anthropic.png', 
    'gemini': 'logo_gemini.png',
    'deepseek': 'logo_deepseek.png',
    'grok': 'logo_grok.png',
    'meta': 'logo_meta.png'
}

def parse_logo_overrides() -> Dict[str, str]:
    """Parse logo overrides from environment variables.
    
    Supports multiple formats:
    - LOGO_OVERRIDES='{"module_name": "path/to/logo.png", "module.model_id": "path/to/logo.png"}'
    - Individual overrides: LOGO_OVERRIDE_PROXYAPI="./images/logo_backup.png"
    
    Returns:
        Dictionary mapping module/model names to logo file paths
    """
    overrides = {}
    
    # Parse JSON format from LOGO_OVERRIDES
    logo_overrides_json = os.getenv('LOGO_OVERRIDES')
    if logo_overrides_json:
        try:
            parsed_overrides = json.loads(logo_overrides_json)
            if isinstance(parsed_overrides, dict):
                overrides.update(parsed_overrides)
                print(f"📷 Loaded logo overrides from LOGO_OVERRIDES: {list(parsed_overrides.keys())}")
            else:
                print("Warning: LOGO_OVERRIDES should be a JSON object")
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in LOGO_OVERRIDES: {e}")
    
    # Parse individual LOGO_OVERRIDE_* environment variables
    for key, value in os.environ.items():
        if key.startswith('LOGO_OVERRIDE_') and value:
            # Convert LOGO_OVERRIDE_PROXYAPI to proxyapi
            override_key = key[len('LOGO_OVERRIDE_'):].lower()
            overrides[override_key] = value
            print(f"📷 Loaded logo override for {override_key}: {value}")
    
    return overrides

# Load logo overrides
LOGO_OVERRIDES = parse_logo_overrides()

def load_logo_as_base64(logo_filename: str) -> str:
    """Load a logo file and convert it to base64 data URL."""
    # Handle both relative and absolute paths
    if os.path.isabs(logo_filename):
        logo_path = Path(logo_filename)
    else:
        # Try relative to images directory first, then relative to current directory
        logo_path = Path('images') / logo_filename
        if not logo_path.exists():
            logo_path = Path(logo_filename)
    
    if not logo_path.exists():
        print(f"Warning: Logo file {logo_path} not found")
        return "/static/favicon.png"  # Fallback to default
    
    with open(logo_path, 'rb') as f:
        logo_data = f.read()
    
    # Encode as base64 data URL
    base64_data = base64.b64encode(logo_data).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"

def determine_logo_provider(model_id: str) -> str:
    """Determine the logo provider based on model ID patterns."""
    model_id_lower = model_id.lower()
    
    # OpenAI patterns
    if any(pattern in model_id_lower for pattern in ['gpt', 'o1', 'o3', 'o4', 'chatgpt']):
        return 'openai'
    
    # Check for models starting with 'o' followed by digit (OpenAI pattern)
    if len(model_id_lower) >= 2 and model_id_lower[0] == 'o' and model_id_lower[1].isdigit():
        return 'openai'
    
    # Anthropic patterns
    if 'claude' in model_id_lower:
        return 'anthropic'
    
    # Google patterns
    if 'gemini' in model_id_lower:
        return 'gemini'
    
    # DeepSeek patterns
    if 'deepseek' in model_id_lower:
        return 'deepseek'
    
    # Grok patterns
    if 'grok' in model_id_lower:
        return 'grok'
    
    # Meta patterns
    if any(pattern in model_id_lower for pattern in ['llama', 'meta']):
        return 'meta'
    
    # Default fallback
    return 'openai'

def get_logo_data_url(model_id: str, module_name: str = None) -> str:
    """Get the base64 data URL for the appropriate logo.
    
    Args:
        model_id: The model ID to determine logo for
        module_name: The module name for override lookup
    
    Returns:
        Base64 data URL for the logo
    """
    # Check for overrides in priority order:
    # 1. Full model ID (module.model_id)
    # 2. Module name
    # 3. Model ID alone
    # 4. Default provider mapping
    
    if module_name:
        full_model_id = f"{module_name}.{model_id}"
        
        # Check for full model ID override
        if full_model_id in LOGO_OVERRIDES:
            return load_logo_as_base64(LOGO_OVERRIDES[full_model_id])
        
        # Check for module-level override
        if module_name in LOGO_OVERRIDES:
            return load_logo_as_base64(LOGO_OVERRIDES[module_name])
    
    # Check for model ID override
    if model_id in LOGO_OVERRIDES:
        return load_logo_as_base64(LOGO_OVERRIDES[model_id])
    
    # Fall back to default provider mapping
    provider = determine_logo_provider(model_id)
    logo_filename = LOGO_MAPPING.get(provider, 'logo_openai.png')
    return load_logo_as_base64(logo_filename)

def load_function_module(module_path: str, module_name: str) -> Optional[List[Dict[str, Any]]]:
    """Load a function module and extract its AVAILABLE_MODELS."""
    module_path = Path(module_path)
    
    if not module_path.exists():
        print(f"Error: Module file {module_path} not found")
        return None
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load module spec for {module_name}")
        return None
        
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error loading module {module_name}: {e}")
        return None
    
    # Extract AVAILABLE_MODELS
    if not hasattr(module, 'AVAILABLE_MODELS'):
        print(f"Error: Module {module_name} does not have AVAILABLE_MODELS")
        return None
    
    return module.AVAILABLE_MODELS

def load_pricing_data_module(pricing_module_path: str) -> Optional[Dict[str, Any]]:
    """Load a pricing data module and extract its pricing_data."""
    pricing_module_path = Path(pricing_module_path)
    
    if not pricing_module_path.exists():
        print(f"Error: Pricing module file {pricing_module_path} not found")
        return None
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("pricing_data_module", pricing_module_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load module spec for pricing module {pricing_module_path}")
        return None
        
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error loading pricing module {pricing_module_path}: {e}")
        return None
    
    # Extract pricing_data
    if not hasattr(module, 'pricing_data'):
        print(f"Error: Pricing module {pricing_module_path} does not have pricing_data")
        return None
    
    return module.pricing_data

def validate_models_against_pricing_data(available_models: List[Dict[str, Any]], pricing_data: Dict[str, Any], module_name: str) -> bool:
    """Validate that all model IDs from available_models exist in pricing_data.
    
    Args:
        available_models: List of model definitions from a function module
        pricing_data: Dictionary containing pricing information for models
        module_name: Name of the module being validated (for error messages)
    
    Returns:
        True if all models are found in pricing data, False otherwise
    """
    missing_models = []
    
    for model in available_models:
        model_id = model.get('id')
        if not model_id:
            print(f"Warning: Model in {module_name} module has no 'id' field: {model}")
            continue
            
        pricing_model_id = module_name + "." + model_id
        if pricing_model_id in pricing_data:
            print(f"✅ Model {pricing_model_id} found in pricing data")
        else:
            print(f"❌ Model {pricing_model_id} not found in pricing data")
            missing_models.append(model_id)
    
    if missing_models:
        print(f"❌ Error: The following model IDs from module '{module_name}' are missing from pricing data:")
        for model_id in missing_models:
            print(f"   - {model_id}")
        print(f"   Please add pricing information for these models to the pricing data module.")
        return False
    
    print(f"✅ All {len(available_models)} models from module '{module_name}' found in pricing data")
    return True

def check_model_exists(model_id: str, token: str) -> Optional[Dict[str, Any]]:
    """Check if a model exists in OpenWebUI and return its data."""
    url = f"{API_URL}/api/v1/models/base"
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Try to parse JSON, handle if it's not valid
        try:
            models = response.json()
        except json.JSONDecodeError as json_err:
            print(f"Error: Response from {url} is not valid JSON for model_id '{model_id}'.")
            print(f"  Status Code: {response.status_code}")
            print(f"  Response Text: {response.text[:500]}...") # Print first 500 chars
            print(f"  JSONDecodeError: {json_err}")
            return None

        # The /api/v1/models/base endpoint returns a direct list, not wrapped in 'data'
        if not isinstance(models, list):
            print(f"Error: Expected response to be a list from {url} for model_id '{model_id}', but got {type(models)}.")
            print(f"  Response JSON: {models}")
            return None

        for model in models:
            if isinstance(model, dict) and model.get('id') == model_id:
                return model
        
        return None  # Model not found in the list
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred in check_model_exists for {model_id} at {url}: {http_err}")
        if http_err.response is not None:
            print(f"  Response Status: {http_err.response.status_code}")
            print(f"  Response Text: {http_err.response.text[:500]}...")
        return None
    except requests.exceptions.RequestException as e: # Catches network errors, DNS failures, etc.
        print(f"Network/request error in check_model_exists for {model_id} at {url}: {e}")
        return None

def create_model(model_data: Dict[str, Any], token: str) -> bool:
    """Create a new model in OpenWebUI."""
    url = f"{API_URL}/api/v1/models/create"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, json=model_data)
        response.raise_for_status()
        print(f"✓ Created model: {model_data['name']} (ID: {model_data['id']})")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Error creating model {model_data['id']}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response: {e.response.text}")
        return False

def update_model(model_id: str, model_data: Dict[str, Any], token: str) -> bool:
    """Update an existing model in OpenWebUI."""
    url = f"{API_URL}/api/v1/models/model/update?id={model_id}"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, json=model_data)
        response.raise_for_status()
        print(f"✓ Updated model: {model_data['name']} (ID: {model_id})")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Error updating model {model_id}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response: {e.response.text}")
        return False

def merge_preserve_existing(new_data: Dict[str, Any], existing_data: Dict[str, Any], preserve_keys: set = None) -> Dict[str, Any]:

    if preserve_keys is None:
        preserve_keys = set()
    
    result = new_data.copy()
    
    for key, existing_value in existing_data.items():
        if key in preserve_keys:
            # Always preserve these keys
            result[key] = existing_value
        elif key not in new_data:
            # Key doesn't exist in new data, preserve from existing
            result[key] = existing_value
        elif isinstance(existing_value, dict) and isinstance(new_data.get(key), dict):
            # Both are dictionaries, merge recursively
            result[key] = merge_preserve_existing(new_data[key], existing_value, preserve_keys)
        # If key exists in new_data and is not a dict, new_data takes precedence
    
    return result

def build_model_data(model: Dict[str, Any], module_name: str, existing_model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the model data structure for OpenWebUI API."""
    # Construct the full model ID with module prefix
    full_model_id = f"{module_name}.{model['id']}"
    
    # Get logo data URL
    logo_url = get_logo_data_url(model['id'], module_name)
    
   # Build capabilities (standard set for all models)
    capabilities = {
        "vision": True,
        "file_upload": True,
        "web_search": True,
        "image_generation": False,
        "code_interpreter": True,
        "citations": True
    }
    
    # Build the new model data structure (what we want to set/override)
    new_model_data = {
        "id": full_model_id,
        "name": model.get('name', model['id']),
        "meta": {
            "profile_image_url": logo_url,
            "capabilities": capabilities,
        },
        "params": {},
        "object": "model",
        "owned_by": "system",
        "pipe": {"type": "pipe"},
    }
    
    # Add description if provided in the module
    if model.get("description"):
        new_model_data["meta"]["description"] = model["description"]
    
    # Add system prompt if provided
    if SYSTEM_PROMPT:
        new_model_data["params"]["system"] = SYSTEM_PROMPT
    
    # If updating, merge with existing data preserving what we're not overriding
    if existing_model:
        # Merge recursively, preserving existing values where we're not overriding
        model_data = merge_preserve_existing(new_model_data, existing_model)
    else:
        model_data = new_model_data
    
    return model_data

def submit_model_order_config(config_file: str, token: str) -> bool:
    """Submit model order configuration to OpenWebUI.
    
    Args:
        config_file: Path to JSON file containing model order configuration
        token: OpenWebUI API token
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(config_file):
        print(f"Error: Model order file {config_file} not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Validate required fields
        if not isinstance(config_data, dict):
            print(f"Error: Model order file must contain a JSON object")
            return False
        
        if "MODEL_ORDER_LIST" not in config_data:
            print(f"Error: Model order file must contain 'MODEL_ORDER_LIST' field")
            return False
        
        if not isinstance(config_data["MODEL_ORDER_LIST"], list):
            print(f"Error: 'MODEL_ORDER_LIST' must be an array")
            return False
        
        print(f"📋 Submitting model order configuration from {config_file}")
        print(f"   Default model: {config_data.get('DEFAULT_MODELS', 'Not specified')}")
        print(f"   Model count: {len(config_data['MODEL_ORDER_LIST'])}")
        
        # Log the first few and last few models for verification
        model_list = config_data['MODEL_ORDER_LIST']
        
        # Submit to OpenWebUI API
        url = f"{API_URL}/api/v1/configs/models"
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, json=config_data)
        
        print(f"\nAPI Response Status Code: {response.status_code}")
        
        response.raise_for_status()
        
        print(f"✅ Successfully submitted model order configuration")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in model order file: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Error submitting model order configuration: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response Status: {e.response.status_code}")
            print(f"  Response Headers: {dict(e.response.headers)}")
            print(f"  Response Text: {e.response.text}")
        return False
    except Exception as e:
        print(f"Error processing model order file: {e}")
        return False

def process_module(module_path: str, module_name: str, token: str, pricing_data: Optional[Dict[str, Any]] = None) -> int:
    """Process all models in a function module."""
    print(f"\n🔄 Processing module: {module_name}")
    
    # Load the module and its models
    available_models = load_function_module(module_path, module_name)
    if not available_models:
        return 0
    
    # Validate against pricing data if provided
    if pricing_data is not None:
        if not validate_models_against_pricing_data(available_models, pricing_data, module_name):
            print(f"❌ Stopping processing of module '{module_name}' due to missing pricing data")
            return 0
    
    success_count = 0
    
    for model in available_models:
        model_id = f"{module_name}.{model['id']}"
        
        # Check if model already exists
        existing_model = check_model_exists(model_id, token)
        
        # Build model data
        model_data = build_model_data(model, module_name, existing_model)
        
        # Create or update the model
        if existing_model:
            if update_model(model_id, model_data, token):
                success_count += 1
        else:
            if create_model(model_data, token):
                success_count += 1
    
    print(f"📊 Module {module_name}: {success_count}/{len(available_models)} models processed successfully")
    return success_count

def main():
    """Main function to process all specified modules."""
    if not API_KEY:
        print("Error: OPENWEBUI_API_KEY environment variable is required")
        sys.exit(1)
    
    print(f"🚀 Starting model updates for OpenWebUI at {API_URL}")
    
    # Load pricing data module if specified
    pricing_data = None
    if args.pricing_module:
        print(f"📊 Loading pricing data from: {args.pricing_module}")
        pricing_data = load_pricing_data_module(args.pricing_module)
        if pricing_data is None:
            print("❌ Failed to load pricing data module. Exiting.")
            sys.exit(1)
        print(f"✅ Loaded pricing data for {len(pricing_data)} models")
    
    # Submit model order configuration if provided
    if args.model_order_file:
        if not submit_model_order_config(args.model_order_file, API_KEY):
            print("Failed to submit model order configuration")
            sys.exit(1)
        print()  # Add spacing
    
    # Process modules
    total_success = 0
    failed_modules = []
    
    for module_path in args.modules:
        module_name = os.path.basename(module_path)
        if module_name.endswith('.py'):
            module_name = module_name[:-3]
        module_success = process_module(module_path, module_name, API_KEY, pricing_data)
        if pricing_data is not None and module_success == 0:
            # If pricing validation was enabled and no models were processed, 
            # it means validation failed
            failed_modules.append(module_name)
        else:
            total_success += module_success
    
    # Check if any modules failed due to pricing validation
    if failed_modules:
        print(f"\n❌ The following modules failed pricing validation and were not processed:")
        for module_name in failed_modules:
            print(f"   - {module_name}")
        print(f"\n💡 Please add missing model pricing data to: {args.pricing_module}")
        sys.exit(1)
    
    print(f"\n✅ Completed! {total_success} models processed successfully across {len(args.modules)} modules")
    
    if args.model_order_file:
        print("📋 Model order configuration was also submitted successfully")

if __name__ == "__main__":
    main() 