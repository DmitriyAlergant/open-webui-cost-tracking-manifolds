# OpenWebUI Model Update Script

This script automatically creates and updates models in OpenWebUI based on the `AVAILABLE_MODELS` defined in function modules (openai, anthropic, etc.).

## Features

- ✅ Reads `AVAILABLE_MODELS` from specified function modules
- ✅ Maps model names to appropriate provider logos automatically
- ✅ Sets system prompts from environment variables
- ✅ Creates or updates models in OpenWebUI via API
- ✅ Handles both create and update operations automatically
- ✅ Preserves existing model configurations when updating

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```bash
# Required: OpenWebUI Configuration
OPENWEBUI_API_URL=http://localhost:3000
OPENWEBUI_API_KEY=your_openwebui_api_token_here

# Optional: System prompt to be applied to all models
SYSTEM_PROMPT=The assistant always responds to the person in the language they use or request. If the person messages in Russian then respond in Russian, if the person messages in English then respond in English, and so on for any language.

MARKDOWN USAGE:
Use markdown for code formatting. Immediately after closing coding markdown, ask the person if they would like explanation or breakdown of the code. Avoid detailed explanation until or unless specifically requested.

RESPONSE STYLE:
Initially a shorter answer while respecting any stated length and comprehensiveness preferences. Address the specific query or task at hand, avoiding tangential information unless critical. Avoid writing lists when possible, but if the lists is needed focus on key info instead of trying to be comprehensive.
```

## Getting Your OpenWebUI API Token

1. Log into your OpenWebUI instance
2. Go to Admin Settings > Users
3. Click on your profile and generate an API token
4. Copy the token and use it as `OPENWEBUI_API_KEY`

## Logo Mapping

The script automatically determines the appropriate logo based on model ID patterns:

| Pattern | Logo |
|---------|------|
| `gpt`, `o1`, `o3`, `o4`, `chatgpt`, `o` + digit | OpenAI logo |
| `claude` | Anthropic logo |
| `gemini` | Google/Gemini logo |
| `deepseek` | DeepSeek logo |
| `grok` | Grok logo |
| `llama`, `meta` | Meta logo |

## Usage

Usage:
    python update_models_in_openwebui.py src/functions/openai.py
    python update_models_in_openwebui.py src/functions/openai.py src/functions/anthropic.py
    python update_models_in_openwebui.py -e .env.production src/functions/openai.py
    python update_models_in_openwebui.py --env-file /path/to/custom.env src/functions/openai.py
    python update_models_in_openwebui.py --model-order-file ./model_order.json src/functions/openai.py
    python update_models_in_openwebui.py --pricing-module ./src/functions/pricing-data-module/module_usage_tracking_pricing_data.py src/functions/openai.py


## What the Script Does

For each model in the specified modules, the script:

1. **Checks if the model exists** in OpenWebUI
3. **Determines the appropriate logo** based on model name patterns
4. **Sets model metadata**:
   - Name (from `AVAILABLE_MODELS`)
   - Description (if available in `AVAILABLE_MODELS`)
   - System prompt (from environment variable SYSTEM_PROMPT)
   - Profile image (automatically selected logo)
   - Preserves existing visibility and access control flags
5. **Creates or updates** the model via OpenWebUI API


## Example Output

```
🚀 Starting model updates for OpenWebUI at http://localhost:3000

🔄 Processing module: openai
✓ Created model: gpt-4o (ID: openai.gpt-4o)
✓ Updated model: claude-sonnet-4-20250514 (ID: openai.claude-sonnet-4-20250514)
✓ Created model: gemini-2.5-pro-preview (ID: openai.gemini-2.5-pro-preview-05-06)
📊 Module openai: 25/25 models processed successfully

✅ Completed! 25 models processed successfully across 1 modules
```
