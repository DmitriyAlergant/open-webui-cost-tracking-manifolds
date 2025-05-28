# Cost Tracking Manifold Functions for Open WebUI

## Overview

This repository contains another implementation of manifold functions for OpenWebUI integration with major providers (Anthropic, OpenAI, etc), with usage volumetrics and costs tracking.

To track costs for all providers, you need to DISABLE the OpenWebUI built-in OpenAI integration and only rely on these functions.

While Open WebUI Community has earlier developed a few usage costs tracking Filter functions (example: https://openwebui.com/f/bgeneto/cost_tracker or https://openwebui.com/f/infamystudio/gpt_usage_tracker) they all suffer from common flaws, such as not capturing the usage via APi. OpenWebUI Filter Functions 'outflow' method is only called from the OpenWebUI-specific /chat/completed endpoint, that is only triggered by the Web UI. It is not part of OpenAI-compatible API specs. This method (and therefore any Filter outflow methods) would not be called by any OpenAI-compatible 3rd party tools leveraging OpenWebUI's API.

This implementation fully relies on Pipes (Manifolds) to wrap around the responses streaming and capture usage in a reliable way, so both Web UI and API usage is captured in a similar way. For scalability, usage data is captured to the main database (SQLite or Postgre). For convenient costs retrieval we provide a chatbot interface presenting itself as another model ("Usage Reporting Bot")


## Features

- **Usage and Costs data is logged to DB** a new table usage_costs is created in the main SQLite or Postgres DB
- **Costs tracking works both for UI and API usage** tracking does not depend on /chat/completed calls by the Web UI
- **Emits status event messages** with token counts, estimated costs, request status
- **Built-in reporting on usage costs** by users (their own costs only) and administrators: **Talk with the 'Usage Reporting Bot'** pipe
- **Uses LiteLLM SDK for unified integration**
- **Captures and displays reasoning thoughts into thinking boxes** for most models that support it
- **Automated scripted updates of OpenWebUI base model definition:** names, descriptions, system prompts, profile icons

## Requirements

- OpenWebUI v0.6.9+ - Anthropic manifold now depends on requirements installation (LiteLLM SDK), available in OWUI 0.6.9+

## Installation (Manual)

- In Open WebUI, manually create functions with IDs matching the .py file names from this repo (src/functions). You may reference the screenshot below. <br/>Сopy&Paste the code of each function from the Python scripts in this repo.

- Prefer short IDs for provider manifold funtions (```openai```, ```anthropic```, ```google```, etc). These IDs will become a part of the Model ID (ex: openai.gpt-4o-mini) and will be visible throughout the system such as over API, in Usage_Costs table and costs reporting, etc.

- **All "module_*" functions must to be deployed**, and the ID must match the .py file name precisely (ex: ```module_litellm_pipe``` etc.). 
Other pipe functions in this repo are referencing these shared modules by their ID (hardcoded), so all modules must be deployed with an exact ID matching the python script file name.

- Provider-specific functions (openai, anthropic, gemini, openrouter, etc.) can be deployed at your discretion. You may only deploy those that you need.

- Double-check function IDs before saving! Any time you typed or changed the function name (when creating it for the first time), OpenWebUI automatically adjusts the ID. Make sure to fix the ID before saving the function. If you saved the function with a wrong ID, or if you are getting "function_module_... not found" error, recreate the module function with the correct ID.

- Do not forget to also deploy the ``src/functionsdata/module_usage_tracking_pricing_data.py`` as one more function with  "``module_usage_tracking_pricing_data``" ID.

- The "Usage Reporting Bot" function is optional, but it provides convenient access to users and administrators to view their accumulated usage costs (adds a pseudo "model" to the list). You may want to deploy it. 

- in Admin Settings -> Functions, enable all deployed functions and modules, and configure all Valves to provide API keys or other settings where applicable

- **Additional Recommendation**: configure and improve Model Definitions using Admin Settings -> Models section, such as

  - Access (Public or Private restricted to a Group)
  - Visibility (show or hide the model from themain dropdown list)
  - Upload model profile picture images for models. **Keep them small - few kb max (64x64)**<br/> OpenWebUI return the image data for all models on every API call to the web client.
  - Provide System Prompts for models
  - Provide Descriptions for models

## Installation (Scripted)

### 1. Clone the repo locally

### 2. (Optional) Activate virtual pyenv

### 3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### 4. Create `.env` file with connection detials

   ```python
   #Your OpenWebUI URL, e.g. http://localhost:3000
   OPENWEBUI_API_URL=http:// 

   #Get your API key in OpenWebUI User Settings -> Account -> API Keys
   OPENWEBUI_API_KEY=sk-....  
   ```

   **NOTE:** If API Key Authentication in your Open-WebUI instance is disabled or restricted by endpoints, you can use JWT Token instead (just use the token value as OPENWEBUI_API_KEY). You can obtain your current JWT token from the Browser's Developer Tools after logging into Open WebUI.  Application -> Cookies -> (your Open WebUI URL) -> "token" cookie value
   
### 5. Deploy pipe functions and modules to OpenWebUI:

   ```bash
   python deploy_to_openwebui.py src/functions/module*                  # shared modules (must be enabled)
   python deploy_to_openwebui.py src/functions/usage_reporting_bot.py   # Usage Reporting Bot ("model") 
   python deploy_to_openwebui.py src/functions/openai.py                # direct OpenAI pipe via LiteLLM SDK
   python deploy_to_openwebui.py src/functions/anthropic.py             # direct Anthropic pipe via LiteLLM SDK
   python deploy_to_openwebui.py src/functions/gemini.py                # direct Gemini pipe via LiteLLM SDK
   python deploy_to_openwebui.py src/functions/openrouter.py            # direct OpenRouter pipe via LiteLLM SDK
   ```

### 6. In OpenWebUI Enable all functions, configure function Valves, then configure models
   
   Provide API keys where applicable, and/or custom API URLs if required

   REview Models - disable unneeded, configure visibility, names, descriptions, system prompts, access, model profile images - take from this repo ./image/logo_*.png

### 7. Scripted update of model descriptions

This script automatically creates and updates models in OpenWebUI based on the `AVAILABLE_MODELS` defined in function modules (openai, anthropic, gemini, etc.).

**7.1. Recommended: add "SYSTEM_PROMPT" configuration to the .env file**

```bash
# Required: OpenWebUI Configuration
OPENWEBUI_API_URL=http://localhost:3000
OPENWEBUI_API_KEY=your_openwebui_api_token_here

# Optional: System prompt to be applied to all models, e.g.
SYSTEM_PROMPT=The assistant always responds to the person in the language they use or request. If the person messages in Russian then respond in Russian, if the person messages in English then respond in English, and so on for any language.
MARKDOWN USAGE: Use markdown for code formatting. Immediately after closing coding markdown, ask the person if they would like explanation or breakdown of the code. Avoid detailed explanation until or unless specifically requested.
RESPONSE STYLE: Initially a shorter answer while respecting any stated length and comprehensiveness preferences. Address the specific query or task at hand, avoiding tangential information unless critical. Avoid writing lists when possible, but if the lists is needed focus on key info instead of trying to be comprehensive.
```

**7.2. Run model update scripts**

```bash
# Update model settings for one provider
python update_models_in_openwebui.py  -env .env.local src/functions/openrouter.py

# Update model settings for multiple providers
python update_models_in_openwebui.py src/functions/openai.py src/functions/anthropic.py src/functions/gemini.py scc/functions/openrouter.py

# Validate if all models are available in the pricing data file
python update_models_in_openwebui.py --pricing-module ./src/functions/module_usage_tracking_pricing_data.py src/functions/openai.py

# Upload models ordering list
python update_models_in_openwebui.py -env .env.local --model-order-file ./model_order_list.json
```