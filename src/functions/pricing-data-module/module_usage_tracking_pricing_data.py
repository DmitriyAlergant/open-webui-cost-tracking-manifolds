"""
title: Usage Costs Tracking Util - Models Pricing Data
author: Dmitriy Alergant
author_url: https://github.com/DmitriyAlergant-t1a/open-webui-cost-tracking-manifolds
version: 0.1.0
required_open_webui_version: 0.3.17
license: MIT
"""

pricing_data = {
    "google_openai_api.gemini-2.5-pro-preview-03-25": {
        "input_cost_per_token": 1.5,            # blend of short-context and long-context (>200K) price
        "output_cost_per_token": 12,            
        "input_display_cost_per_token": 1.5,
        "output_display_cost_per_token": 12,
        "token_units": 1000000,
        "cost_currency": "USD",
    },  
    "google_openai_api.gemini-2.5-flash-preview-04-17": {
        "input_cost_per_token": 0.15,
        "output_cost_per_token": 1.2,           # blend of thinking and non-thinking tokens price
        "input_display_cost_per_token": 0.15,
        "output_display_cost_per_token": 1.2,
        "token_units": 1000000,
        "cost_currency": "USD",
    },   
    "openai.o3": {
        "input_cost_per_token": 10,
        "output_cost_per_token": 40,
        "input_display_cost_per_token": 10,
        "output_display_cost_per_token": 40,
        "token_units": 1000000,
        "cost_currency": "USD",
    },   
    "openai.o4-mini": {
        "input_cost_per_token": 1.1,
        "output_cost_per_token": 4.4,
        "input_display_cost_per_token": 1.1,
        "output_display_cost_per_token": 4.4,
        "token_units": 1000000,
        "cost_currency": "USD",
    },   
    "openai.gpt-4.1": {
        "input_cost_per_token": 2,
        "output_cost_per_token": 8,
        "input_display_cost_per_token": 2,
        "output_display_cost_per_token": 8,
        "token_units": 1000000,
        "cost_currency": "USD",
    },   
    "openai.gpt-4.1-mini": {
        "input_cost_per_token": 0.4,
        "output_cost_per_token": 1.6,
        "input_display_cost_per_token": 0.4,
        "output_display_cost_per_token": 1.6,
        "token_units": 1000000,
        "cost_currency": "USD",
    },   
    "openai.gpt-4.1-nano": {
        "input_cost_per_token": 0.1,
        "output_cost_per_token": 0.4,
        "input_display_cost_per_token": 0.1,
        "output_display_cost_per_token": 0.4,
        "token_units": 1000000,
        "cost_currency": "USD",
    },   
    "openai.o1": {
        "input_cost_per_token": 0.015,
        "output_cost_per_token": 0.060,
        "input_display_cost_per_token": 0.015,
        "output_display_cost_per_token": 0.060,
        "token_units": 1000,
        "cost_currency": "USD",
    },    
    "openai.o1-preview": {
        "input_cost_per_token": 0.015,
        "output_cost_per_token": 0.060,
        "input_display_cost_per_token": 0.015,
        "output_display_cost_per_token": 0.060,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "openai.o3-mini": {    
        "input_cost_per_token": 0.0011,
        "output_cost_per_token": 0.0044,
        "input_display_cost_per_token": 0.0011,
        "output_display_cost_per_token": 0.0044,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "openai.o1-mini": {
        "input_cost_per_token": 0.0011,
        "output_cost_per_token": 0.0044,
        "input_display_cost_per_token": 0.0011,
        "output_display_cost_per_token": 0.0044,
        "token_units": 1000,
        "cost_currency": "USD",
    },   
    "openai.gpt-4o": {
        "input_cost_per_token": 0.0025,
        "output_cost_per_token": 0.0100,
        "input_display_cost_per_token": 0.0025,
        "output_display_cost_per_token": 0.0100,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "openai.gpt-4.5-preview": {
        "input_cost_per_token": 0.075,
        "output_cost_per_token": 0.150,
        "input_display_cost_per_token": 0.075,
        "output_display_cost_per_token": 0.150,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "openai.chatgpt-4o-latest": {
        "input_cost_per_token": 0.005,
        "output_cost_per_token": 0.0150,
        "input_display_cost_per_token": 0.005,
        "output_display_cost_per_token": 0.0150,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "openai.gpt-4o-2024-11-20": {
        "input_cost_per_token": 0.0025,
        "output_cost_per_token": 0.0100,
        "input_display_cost_per_token": 0.0025,
        "output_display_cost_per_token": 0.0100,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "openai.gpt-4o-2024-08-06": {
        "input_cost_per_token": 0.0025,
        "output_cost_per_token": 0.0100,
        "input_display_cost_per_token": 0.0025,
        "output_display_cost_per_token": 0.0100,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "openai.gpt-4o-2024-05-13": {
        "input_cost_per_token": 0.0050,
        "output_cost_per_token": 0.0150,
        "input_display_cost_per_token": 0.0050,
        "output_display_cost_per_token": 0.0150,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "openai.gpt-4o-mini": {
        "input_cost_per_token": 0.00015,
        "output_cost_per_token": 0.00060,
        "input_display_cost_per_token": 0.00015,
        "output_display_cost_per_token": 0.00060,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "openai.gpt-4-turbo": {
        "input_cost_per_token": 0.01,
        "output_cost_per_token": 0.03,
        "input_display_cost_per_token": 0.01,
        "output_display_cost_per_token": 0.03,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "openai.gpt-4": {
        "input_cost_per_token": 0.03,
        "output_cost_per_token": 0.06,
        "input_display_cost_per_token": 0.03,
        "output_display_cost_per_token": 0.06,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "anthropic.claude-3-opus": {
        "input_cost_per_token": 0.015,
        "output_cost_per_token": 0.075,
        "input_display_cost_per_token": 0.015,
        "output_display_cost_per_token": 0.075,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "anthropic.claude-3-sonnet": {
        "input_cost_per_token": 0.003,
        "output_cost_per_token": 0.015,
        "input_display_cost_per_token": 0.003,
        "output_display_cost_per_token": 0.015,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "anthropic.claude-3-5-sonnet": {
        "input_cost_per_token": 0.003,
        "output_cost_per_token": 0.015,
        "input_display_cost_per_token": 0.003,
        "output_display_cost_per_token": 0.015,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "anthropic.claude-3-7-sonnet": {
        "input_cost_per_token": 0.003,
        "output_cost_per_token": 0.015,
        "input_display_cost_per_token": 0.003,
        "output_display_cost_per_token": 0.015,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "anthropic.claude-3-haiku": {
        "input_cost_per_token": 0.00025,
        "output_cost_per_token": 0.00125,
        "input_display_cost_per_token": 0.00025,
        "output_display_cost_per_token": 0.00125,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "anthropic.claude-3-5-haiku": {
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.005,
        "input_display_cost_per_token": 0.001,
        "output_display_cost_per_token": 0.005,
        "token_units": 1000,
        "cost_currency": "USD",
    },
    "databricks.databricks-meta-llama-3-1-70b-instruct": {
        "input_cost_per_token": 1.00,
        "output_cost_per_token": 3.00,
        "input_display_cost_per_token": 1.00,
        "output_display_cost_per_token": 3.00,
        "token_units": 1000000,
        "cost_currency": "USD",
    },
    "databricks.databricks-meta-llama-3-1-405b-instruct": {
        "input_cost_per_token": 5.00,
        "output_cost_per_token": 15.00,
        "input_display_cost_per_token": 5.00,
        "output_display_cost_per_token": 15.00,
        "token_units": 1000000,
        "cost_currency": "USD",
    },
}

# For OpenWebUI to accept this as a Function Module, there has to be a Filter or Pipe or Action class
class Pipe:
    def __init__(self):
        self.type = "manifold"
        self.id = "usage-tracking-util-pricing-data"
        self.name = "Usage Tracking Util - Models Pricing data"
        
        pass

    def pipes(self) -> list[dict]:
        return []