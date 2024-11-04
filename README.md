# Cost Tracking Manifold Functions for Open WebUI

## Overview

This repository contains another implementation of manifold functions for OpenAI and Anthropic, with custom usage costs tracking to the main database (SQLite or Postgre), and bot interface for basic reporting.

## Features

- **Batch and Streaming completions** - Leverages official OpenAI and Anthropic SDKs
- **Usage data logged to DB** - new table usage_costs
- **Calculates usage costs** - pricing data statically defined in cost_tracking_util
- **Displays status message** - token counts, estimated costs, request status
- **Built-in reporting on usage costs** - talk to 'Usage Reporting Bot' model pipe

## Requirements

- Python 3.7+
- OpenWebUI v0.3.30+

## Installation

1. Download the source and import into Open WebUI functions interface.
2. cost_tracking_util function is a dependency, and must also be installed and enabled
