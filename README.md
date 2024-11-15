# Cost Tracking Manifold Functions for Open WebUI

## Overview

This repository contains another implementation of manifold functions for OpenWebUI integration with major providers (Anthropic, OpenAI, etc), with usage volumetrics and costs tracking.

To track costs for all providers, you need to DISABLE the OpenWebUI built-in OpenAI integration and only rely on these functions.

While OpenWebUI Community has earlier developed a few usage costs tracking Filter functions (example: https://openwebui.com/f/bgeneto/cost_tracker or https://openwebui.com/f/infamystudio/gpt_usage_tracker) they all suffer from common flaws, such as not capturing the usage via APi. OpenWebUI Filter Functions 'outflow' method is only called from the OpenWebUI-specific /chat/completed endpoint, that is only triggered by the Web UI. It is not part of OpenAI-compatible API specs. This method (and therefore any Filter outflow methods) would not be called by any OpenAI-compatible 3rd party tools leveraging OpenWebUI's API.

This implementation fully relies on Pipes (Manifolds) to wrap around the responses streaming and capture usage in a reliable way, so both Web UI and API usage is captured in a similar way. For scalability, usage data is captured to the main database (SQLite or Postgre). For convenient costs retrieval we provide a chatbot interface presenting itself as another model ("Usage Reporting Bot")


## Features

- **Batch and Streaming completions** leverages official OpenAI and Anthropic SDKs
- **Usage data logged to DB** new table usage_costs in the main SQLite or Postgres DB
- **Usage tracking works both for UI and API usage** tracking DOES NOT depend on /chat/completed calls
- **Calculates usage costs** pricing data statically defined in a standalone function module, edit it as needed
- **Displays status message** with token counts, estimated costs, request status
- **Built-in reporting on usage costs** with 'Usage Reporting Bot' model pipe. Talk to it via chat

## Requirements

- Python 3.11+
- OpenWebUI v0.3.30+

## Installation

**1. Clone the repo**

**2. (Optional) Activate virtual pyenv**

**4. Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

**5. Create `.env` file with connection detials**

   ```python
   #Your OpenWebUI URL, e.g. http://localhost:3000
   OPENWEBUI_API_URL=http:// 

   #Get your API key in OpenWebUI User Settings -> Account -> API Keys
   OPENWEBUI_API_KEY=sk-....  
   ```
   
**6. Deploy functions to OpenWebUI:**

   ```bash
   python deploy_to_openwebui.py src/functions/*
   ```
**6. Separately deploy models pricing data "function" to OpenWebUI:**

   ```bash
   python deploy_to_openwebui.py src/functions/data/*
   ```

**7. Enable all functions and configure Valves**
   
   Provide API keys where applicable
   
   For OpenAI also provide comma-separated list of enabled models

**8. Expected Result**

![Deployed Functions Screenshot](images/deployed_functions_screenshot.png)