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

1. **Clone the repo**

2. **Create all functions manually one-by-one into Open WebUI Workspace Functions interface**
   - Each function ID must match the .py file name
   - Functions "usage_tracking_util" and "usage_tracking_util_pricing_data" are shared dependencies and MUST be installed and enabled for either of the manifolds to work

3. **Enable all functions and configure Valves (where exist)**

## Development and Scripted Re-Deployment

1. (Optional) Activate virtual pyenv 

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create `.env` file with the following content:
   ```
   OPENWEBUI_API_URL=http://localhost:3000
   OPENWEBUI_API_KEY=sk-....
   ```

4. Deploy functions to OpenWebUI:
   ```
   python deploy_to_openwebui.py src/functions/*
   ```

5. Deploy Pricing Data function to OpenWebUI:
   ```
   python deploy_to_openwebui.py src/functions/data/*
   ```

6.