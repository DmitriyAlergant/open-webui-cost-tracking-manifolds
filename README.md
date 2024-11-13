# Cost Tracking Manifold Functions for Open WebUI

## Overview

This repository contains another implementation of manifold functions for OpenAI and Anthropic, with custom usage costs tracking to the main database (SQLite or Postgre), and bot interface for basic reporting.

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