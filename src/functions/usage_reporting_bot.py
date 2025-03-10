"""
title: Usage Costs reporting bot
author: Dmitriy Alergant
author_url: https://github.com/DmitriyAlergant-t1a/open-webui-cost-tracking-manifolds
version: 0.1.0
required_open_webui_version: 0.3.17
license: MIT
"""

from pydantic import BaseModel, Field
import re

from datetime import datetime, timedelta
from typing import Optional

import requests

import sys


import pandas as pd

from openai import OpenAI, AsyncOpenAI

from decimal import Decimal
from datetime import datetime
from sqlalchemy import text
from open_webui.internal.db import get_db, engine
from open_webui.utils.misc import get_messages_content, get_last_user_message


class Config:
    DEBUG_PREFIX = "DEBUG:    " + __name__ + " -"
    DEBUG = True

class Pipe:
    class Valves(BaseModel):

        DEBUG: bool = Field(default=False, description="Display debugging messages")

        SUPERUSERS: str = Field(
            default="",
            description="Comma-separated list of user emails with elevated admin-like bot capabilities",
        )

        MAX_SQL_ROWS: int = Field(
            default=200,
            description="Maximum result rows to print for run sql command",
        )

        BALANCE_API_URL: str = Field(
            default="",
            description="Balance API endpoint URL",
        )

        BALANCE_API_KEY: str = Field(default="", description="API key for balance checking")

        SQL_ASSISTANT_MODEL: str = Field(
            default="anthropic.claude-3-5-sonnet-20241022",
            description="Model to use for SQL generation from natural language questions",
        )


    def __init__(self):
        self.type = "pipe"
        self.id = "usage-reporting-bot"
        self.name = "usage-reporting-bot"

        self.valves = self.Valves()

        self.debug_prefix = "DEBUG:    " + __name__ + " -"

        pass

    def get_provider_models(self):
        return [
            {"id": "admin.usage-reporting-bot", "name": "usage-reporting-bot"},
        ]
    
    def is_superuser (self, __user__: dict):
        if __user__["role"] == "admin":
            return True
        
        print (self.valves.SUPERUSERS.split(","))
        
        if __user__["email"] in [user.strip() for user in self.valves.SUPERUSERS.split(",")]:
            print ("Is superuser: True")
            return True
        
        print ("Is superuser: False")
        return False
        

    def strip_markdown_formatting(self, text: str) -> str:
        """
        Strip markdown formatting from text, particularly bold (**) markers.
        """
        # Remove bold markers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        # Remove any other markdown formatting if needed in the future
        return text.strip()

    async def pipe(self, body: dict, __user__: dict) -> str:
        command = get_last_user_message(body["messages"]).strip()
        # Strip any markdown formatting from the command
        command = self.strip_markdown_formatting(command)

        if self.valves.DEBUG:
            print(f"usage-reporting-bot ({__user__['email']}): {command}")

        # Replace /usage_costs with /usage_stats
        if command.startswith("/usage_costs"):
            command = "/usage_stats" + command[len("/usage_costs"):]

        if command == "/help":
            return self.print_help(__user__)
        else:
            return await self.handle_command(__user__, body, command)

    async def handle_command(self, __user__, body, command):
        if command == "/balance":
            if self.is_superuser(__user__):
                return self.get_balance()
            else:
                return "Sorry, this feature is only available to Admins"

        if match := re.match(r"/usage_stats\s+all(?:\s+(\d+)d)?", command):
            days = int(match.group(1)) if match.group(1) else 30

            if self.is_superuser(__user__):
                return self.generate_all_users_report(days)
            else:
                return "Sorry, this feature is only available to Admins"

        if match := re.match(r"/usage_stats\s+([^\s]+@[^\s]+)(?:\s+(\d+)d)?", command):
            specific_user = match.group(1)
            days = int(match.group(2)) if match.group(2) else 30

            is_admin = self.is_superuser(__user__)
            if is_admin or specific_user == __user__["email"]:
                return self.generate_single_user_report(days, specific_user, is_admin)
            else:
                return "Sorry, this feature is only available to Admins"

        if match := re.match(r"/usage_stats(?:\s+(\d+)d)?", command):
            days = int(match.group(1)) if match.group(1) else 30

            is_admin = self.is_superuser(__user__)
            return self.generate_single_user_report(days, __user__["email"], is_admin)

        if command.startswith("/run_sql "):
            if self.is_superuser(__user__):
                return self.run_sql_command(command[len("/run_sql "):])
            else:
                return "Sorry, this feature is only available to Admins"

        if command.startswith("/ask "):
            if self.is_superuser(__user__):
                return await self.handle_ask_command(__user__, body, command[5:])
            else:
                return "Sorry, this feature is only available to Admins"

        return "Invalid command\n\n" + self.print_help(__user__)

    def print_help(self, __user__):
        help_message = (
            "**Available Commands**\n"
            "* **/usage_stats 30d** or **/usage_costs 30d** my own usage stats for 30 days\n\n"
        )
        
        if self.is_superuser(__user__):
            help_message += (
                "**Available Commands (Admins Only)**\n"
                "* **/balance** Check current API balance\n"
                "* **/usage_stats all 45d** or **/usage_costs all 45d** stats by all users for 45 days\n"
                "* **/usage_stats user@email.com** or **/usage_costs user@email.com** stats for the indicated user (default is 30 days)\n"
                "* **/run_sql SELECT count(*) from usage_costs;** allows an admin to run arbitrary SQL SELECT from the database.\n  - For SQLite: use /run_sql PRAGMA table_info(usage_costs) to see available table columns\n  - For Postgres db: /run_sql SELECT * FROM information_schema.columns WHERE table_name = 'usage_costs'\n"
                "* **/ask** Ask questions about usage in natural language. SQL will be generated automatically.\n"
            )
        
        return help_message

    def get_usage_stats(
        self,
        user_email: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        """
        Retrieve total costs by user, summarized per user, model, currency, and date.

        :param user_email: Optional user email for filtering results
        :param start_date: Optional start date for filtering results
        :param end_date: Optional end date for filtering results
        :return: List of dictionaries containing summarized cost data
        """

        is_sqlite = "sqlite" in engine.url.drivername

        date_function = (
            "strftime('%Y-%m-%d', timestamp)"
            if is_sqlite
            else "to_char(timestamp, 'YYYY-MM-DD')"
        )

        query = f"""
            SELECT 
                user_email,
                model,
                cost_currency,
                {date_function} as date,
                SUM(total_cost) as total_cost,
                SUM(COALESCE(display_cost, total_cost)) as display_cost,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens
            FROM usage_costs
            {{where_clause}}
            GROUP BY user_email, model, cost_currency, {date_function}
            ORDER BY user_email, {date_function}, model, cost_currency
            """

        where_conditions = []
        params = {}

        if user_email:
            where_conditions.append("user_email = :user_email")
            params["user_email"] = user_email

        if start_date:
            where_conditions.append("timestamp >= :start_date")
            params["start_date"] = start_date

        if end_date:
            # Include the entire end_date by setting it to the start of the next day
            next_day = end_date + timedelta(days=1)
            where_conditions.append("timestamp < :end_date")
            params["end_date"] = next_day

        where_clause = (
            "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        )
        query = query.format(where_clause=where_clause)

        try:
            with get_db() as db:
                result = db.execute(text(query), params)
                rows = result.fetchall()

                summary = [
                    {
                        "user_email": row.user_email,
                        "model": row.model,
                        "currency": row.cost_currency,
                        "date": row.date,
                        "total_cost": float(row.total_cost),
                        "display_cost": float(row.display_cost),
                        "total_input_tokens": row.total_input_tokens,
                        "total_output_tokens": row.total_output_tokens,
                    }
                    for row in rows
                ]

                if Config.DEBUG:
                    print(
                        f"{Config.DEBUG_PREFIX} Retrieved total costs for {len(summary)} user-model-currency-date combinations"
                    )

                return summary

        except Exception as e:
            print(
                f"{Config.INFO_PREFIX} Database error in get_total_costs_by_user: {e}"
            )
            raise

    def process_usage_stats_command(self, command: str, __user__: dict):
        """
        Process the usage stats command and return a formatted report.

        :param command: The command string (e.g., "/usage_stats", "/usage_stats 30d", "/usage_stats all", "/usage_stats all 30d", "/usage_stats user@email.com", or "/usage_stats user@email.com 30d")
        :param __user__: Dictionary containing user information
        :return: A formatted string report of usage stats
        """

    def get_exchange_rates (self, currencies):
        exchange_rates = {}
        try:
            response = requests.get("https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json")
            if response.status_code == 200:
                data = response.json()
                usd_rates = data.get('usd', {})
                for currency in currencies:
                    if currency == 'USD':
                        exchange_rates[currency] = 1.0
                    else:
                        rate = usd_rates.get(currency.lower(), None)
                        exchange_rates[currency] = rate if rate else 1.0

        except Exception as e:
            print (f"Usage Reporting Bot: unable to fetch currency exchange rates, exception: {e}")

            # Default to 1.0 if unable to fetch or process exchange rates
            exchange_rates = {currency: 1.0 for currency in currencies}

        return exchange_rates

    def generate_all_users_report(self, days: int):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get usage stats for all users
        stats = self.get_usage_stats(start_date=start_date, end_date=end_date)

        if not stats:
            return f"No usage data found in the last {days} days."

        df = pd.DataFrame(stats)

        # Default currency to 'USD' if blank or null
        df['currency'] = df['currency'].fillna('USD').replace('', 'USD')

        # Get unique currencies in data and fetch exchange rates
        currencies = df['currency'].unique()

        if len(currencies) > 1 or 'USD' not in currencies:
            exchange_rates = self.get_exchange_rates(currencies)
        else:
            exchange_rates = {'USD': 1.0}

        # Convert costs to USD only for ranking purposes
        df['cost_for_ranking'] = df.apply(
            lambda row: row['total_cost'] / exchange_rates.get(row['currency'], 1.0),
            axis=1
        )

        # Prepare the report
        report = f"## Usage Report for All Users\n"
        report += f"### Period: {start_date.date()} to {end_date.date()}\n\n"

        # Total costs by currency
        report += "#### Total Usage Costs:\n"
        
        # Group by currency and sum the costs
        currency_totals = df.groupby("currency")[["total_cost", "display_cost"]].sum().round(2)
        
        for currency, row in currency_totals.iterrows():
            total = row["total_cost"]
            display_total = row["display_cost"]
            
            if currency == "USD":
                report += f"- **{display_total:,.2f} {currency}** (Display) | **{total:,.2f} {currency}** (True)\n"
            else:
                report += f"- **{display_total:,.2f} {currency}** (Display) | **{total:,.2f} {currency}** (True)\n"

        # Add total tokens information
        total_input_tokens = df["total_input_tokens"].sum()
        total_output_tokens = df["total_output_tokens"].sum()
        report += "\n#### Total Tokens Used:\n"
        report += f"- Input tokens:  **{total_input_tokens:,}**\n"
        report += f"- Output tokens: **{total_output_tokens:,}**\n"

        # Top 5 models by cost
        report += "\n#### Top 5 Models by Cost:\n"
        
        # Group by model and currency, then sum costs
        model_costs = df.groupby(['model', 'currency'])[['total_cost', 'display_cost', 'cost_for_ranking']].sum().reset_index()
        
        # Get top 5 models by USD-equivalent cost
        top_models = model_costs.groupby('model')['cost_for_ranking'].sum().nlargest(5).index
        top_model_data = model_costs[model_costs['model'].isin(top_models)]
        
        # Prepare headers for the table
        headers = ["Model"]
        for currency in currencies:
            if currency == 'USD':
                headers.append(f"$ (Display)")
                headers.append(f"$ (True)")
            else:
                headers.append(f"{currency} (Display)")
                headers.append(f"{currency} (True)")
        
        # Prepare rows for the table
        rows = []
        for model in top_models:
            row_data = [model]
            for currency in currencies:
                model_currency_data = top_model_data[(top_model_data['model'] == model) & (top_model_data['currency'] == currency)]
                if len(model_currency_data) > 0:
                    cost = model_currency_data['total_cost'].sum()
                    display_cost = model_currency_data['display_cost'].sum()
                    
                    if currency == 'USD':
                        row_data.append(f"${display_cost:,.2f}" if display_cost > 0 else "")
                        row_data.append(f"${cost:,.2f}" if cost > 0 else "")
                    else:
                        row_data.append(f"{display_cost:,.2f}" if display_cost > 0 else "")
                        row_data.append(f"{cost:,.2f}" if cost > 0 else "")
                else:
                    row_data.append("")
                    row_data.append("")
            rows.append(row_data)
        
        # Render the table
        col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
        
        table = "```\n"  # Start code block for fixed-width formatting
        table += " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths)) + "\n"
        table += "-|-".join("-" * width for width in col_widths) + "\n"
        for row in rows:
            table += " | ".join(f"{cell:<{width}}" for cell, width in zip(row, col_widths)) + "\n"
        table += "```\n"  # End code block
        
        report += table
        
        # Top 20 users by cost
        report += "\n#### Top 20 Users by Cost:\n"
        
        # Separate costs per currency for reporting
        for currency in currencies:
            df[f'{currency.lower()}_display_cost'] = df.apply(
                lambda row: row['display_cost'] if row['currency'] == currency else 0,
                axis=1
            )
            df[f'{currency.lower()}_total_cost'] = df.apply(
                lambda row: row['total_cost'] if row['currency'] == currency else 0,
                axis=1
            )

        # Get user totals and select top 20 users
        agg_dict = {}
        for currency in currencies:
            agg_dict[f'{currency.lower()}_display_cost'] = 'sum'
            agg_dict[f'{currency.lower()}_total_cost'] = 'sum'
        agg_dict['total_input_tokens'] = 'sum'
        agg_dict['total_output_tokens'] = 'sum'
        agg_dict['cost_for_ranking'] = 'sum'

        user_totals = df.groupby("user_email").agg(agg_dict).round(2)
        top_users = user_totals.nlargest(20, "cost_for_ranking")

        # Prepare headers for the table - only include currencies that have non-zero values
        headers = ["User"]
        for currency in currencies:
            if (top_users[f'{currency.lower()}_total_cost'] > 0).any():
                if currency == 'USD':
                    headers.extend([f"$ (Display)", f"$ (True)"])
                else:
                    headers.extend([f"{currency} (Display)", f"{currency} (True)"])
        headers.extend(["Input Tokens", "Output Tokens"])

        # Prepare rows for the table
        rows = []
        for user, row in top_users.iterrows():
            row_data = [user]
            for currency in currencies:
                if (top_users[f'{currency.lower()}_total_cost'] > 0).any():
                    display_cost = row[f'{currency.lower()}_display_cost']
                    total_cost = row[f'{currency.lower()}_total_cost']
                    if currency == 'USD':
                        row_data.extend([
                            f"${display_cost:,.2f}" if display_cost > 0 else "",
                            f"${total_cost:,.2f}" if total_cost > 0 else ""
                        ])
                    else:
                        row_data.extend([
                            f"{display_cost:,.2f}" if display_cost > 0 else "",
                            f"{total_cost:,.2f}" if total_cost > 0 else ""
                        ])
            row_data.extend([
                f"{row['total_input_tokens']:,}",
                f"{row['total_output_tokens']:,}"
            ])
            rows.append(row_data)

        # Render the table
        col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
        
        table = "```\n"  # Start code block for fixed-width formatting
        table += " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths)) + "\n"
        table += "-|-".join("-" * width for width in col_widths) + "\n"
        for row in rows:
            table += " | ".join(f"{cell:<{width}}" for cell, width in zip(row, col_widths)) + "\n"
        table += "```\n"  # End code block

        report += table

        return report

    def generate_single_user_report(self, days: int, user_email: str, is_admin: bool = False):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get usage stats
        stats = self.get_usage_stats(
            user_email=user_email, start_date=start_date, end_date=end_date
        )

        if not stats:
            return f"No usage data found for user {user_email} in the last {days} days."

        # Convert to DataFrame for easy manipulation
        df = pd.DataFrame(stats)

        # Default currency to 'USD' if blank or null
        df['currency'] = df['currency'].fillna('USD').replace('', 'USD')

        # Get unique currencies in data and fetch exchange rates
        currencies = df['currency'].unique()

        if len(currencies) > 1 or 'USD' not in currencies:
            exchange_rates = self.get_exchange_rates(currencies)
        else:
            exchange_rates = {'USD': 1.0}

        # Group by currency and sum the total cost and display cost
        currency_totals = df.groupby("currency")["total_cost"].sum().round(2)
        currency_display_totals = df.groupby("currency")["display_cost"].sum().round(2)

        # Prepare the report
        report = [
            f"## Usage Report for {user_email}",
            f"### Period: {start_date.date()} to {end_date.date()}",
            "",
            "#### Usage Costs:",
            ""
        ]

        # Sort currencies to ensure consistent order (USD first, then others alphabetically)
        sorted_currencies = sorted(currency_totals.index, key=lambda x: (x != 'USD', x))

        for currency in sorted_currencies:
            total = currency_totals[currency]
            display_total = currency_display_totals[currency]
            
            if is_admin:
                report.append(f"- **{display_total:,.2f} {currency}** (Display) | **USD {total:,.2f} {currency}** (True)")
            else:
                report.append(f"- **{display_total:,.2f} {currency}**")

        # Add total tokens information
        total_input_tokens = df["total_input_tokens"].sum()
        total_output_tokens = df["total_output_tokens"].sum()
        report.extend([
            "",
            "#### Total Tokens Used:",
            f"- Input tokens:  **{total_input_tokens:,}**",
            f"- Output tokens: **{total_output_tokens:,}**",
            ""
        ])

        # TOP 5 MODELS BY COST
        report.append("#### Top 5 Models by Cost:")
        report.append("")

        # Group data and select top 5 models by USD cost (currency-converted)
        model_costs = df.groupby(['model', 'currency'])[['total_cost', 'display_cost']].sum().reset_index()

        model_costs['cost_usd'] = model_costs.apply(
            lambda row: row['total_cost'] / exchange_rates.get(row['currency'], 1.0),
            axis=1
        )

        top_models = model_costs.groupby('model')['cost_usd'].sum().nlargest(5).index

        top_model_data = model_costs[model_costs['model'].isin(top_models)]

        # Prepare data for table rendering
        if is_admin:
            headers = ["Model"]
            for currency in sorted_currencies:  # Use sorted currencies for consistent order
                if currency == 'USD':
                    headers.append(f"$ (Display)")
                    headers.append(f"$ (True)")
                else:
                    headers.append(f"{currency} (Display)")
                    headers.append(f"{currency} (True)")
        else:
            headers = ["Model"]
            for currency in sorted_currencies:  # Use sorted currencies for consistent order
                if currency == 'USD':
                    headers.append("$")
                else:
                    headers.append(f"{currency}")

        rows = []
        for model in top_models:
            row_data = [model]
            for currency in sorted_currencies:  # Use sorted currencies for consistent order
                model_currency_data = top_model_data[(top_model_data['model'] == model) & (top_model_data['currency'] == currency)]
                if len(model_currency_data) > 0:
                    cost = model_currency_data['total_cost'].sum()
                    display_cost = model_currency_data['display_cost'].sum()
                    
                    if is_admin:
                        if currency == 'USD':
                            row_data.append(f"${display_cost:,.2f}" if display_cost > 0 else "")
                            row_data.append(f"${cost:,.2f}" if cost > 0 else "")
                        else:
                            row_data.append(f"{display_cost:,.2f}" if display_cost > 0 else "")
                            row_data.append(f"{cost:,.2f}" if cost > 0 else "")
                    else:
                        if currency == 'USD':
                            row_data.append(f"${display_cost:,.2f}" if display_cost > 0 else "")
                        else:
                            row_data.append(f"{display_cost:,.2f}" if display_cost > 0 else "")
                else:
                    if is_admin:
                        row_data.append("")
                        row_data.append("")
                    else:
                        row_data.append("")
            rows.append(row_data)

        # Render an ASCII table
        if rows:  # Only add table if we have data
            col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
            
            table_lines = [
                "```",
                " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths)),
                "-|-".join("-" * width for width in col_widths)
            ]
            
            for row in rows:
                table_lines.append(" | ".join(f"{cell:<{width}}" for cell, width in zip(row, col_widths)))
            
            table_lines.append("```")
            report.extend(table_lines)

        return "\n".join(report)
    
    def run_sql_command(self, sql_query):
        # Sanitize the query
        sql_query = sql_query.strip()
        print(f"usage_reporting_bot SQL QUERY: {sql_query}")

        if not re.match(r'^(SELECT|PRAGMA TABLE_)', sql_query, re.IGNORECASE):
            err_msg = "Error: Query must start with SELECT or PRAGMA TABLE_LIST() or PRAGMA TABLE_INFO(table)"
            print(f"usage_reporting_bot run_sql | {err_msg}")
            return f"{err_msg}"
        
        if not sql_query.endswith(';'):
            sql_query += ';'

        if sql_query.count(';') > 1:

            err_msg = "Error: Query must not contain multiple semicolons (;)"
            print(f"usage_reporting_bot run_sql |  {err_msg}")
            return f"{err_msg}"

        try:
            with get_db() as db:
                result = db.execute(text(sql_query))
                rows = result.fetchall()

                if not rows:
                    msg =  "Query executed successfully, but returned no results."
                    print(f"usage_reporting_bot run_sql |  {msg}")
                    return f"{msg}"

                # Get column names
                if hasattr(result, 'keys'):
                    headers = result.keys()
                else:
                    headers = rows[0]._fields

                max_print_rows = int(self.valves.MAX_SQL_ROWS)

                # Format data
                formatted_data = []
                total_rows = len(rows)
                for row in rows[:max_print_rows]:
                    formatted_row = []
                    for col in headers:
                        value = getattr(row, col, '')
                        # Add thousands separator for numeric values
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            if isinstance(value, int):
                                formatted_value = f"{value:,}"
                            else:
                                # For floats, keep decimal precision
                                formatted_value = f"{value:,.2f}" if value == int(value) else f"{value:,}"
                        else:
                            formatted_value = str(value)
                        formatted_row.append(formatted_value)
                    formatted_data.append(formatted_row)

                # Calculate column widths
                col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *formatted_data)]

                # Create a markdown table
                table = "```\n"  # Start code block for fixed-width formatting
                table += " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths)) + "\n"
                table += "-|-".join("-" * width for width in col_widths) + "\n"
                for row in formatted_data:
                    table += " | ".join(f"{cell:<{width}}" for cell, width in zip(row, col_widths)) + "\n"
                table += "```\n"  # End code block

                # Add truncation notice if necessary
                if total_rows > max_print_rows:
                    table += f"\n*Note: Results truncated. Showing {max_print_rows} rows out of {total_rows} total rows.*"

                print(f"usage_reporting_bot run_sql |  returned query results {len(rows)} rows")

                return f"Query results:\n\n{table}"

        except Exception as e:
            _, _, tb = sys.exc_info()
            line_number = tb.tb_lineno
            print(f"usage_reporting_bot run_sql | Error on line {line_number}: {e}")
            raise e

    def get_balance(self) -> str:
        if not self.valves.BALANCE_API_KEY:
            return "Error: API key not configured"

        if not self.valves.BALANCE_API_URL:
            return "Error: API URL not configured"

        try:
            if self.valves.DEBUG:
                print(f"{Config.DEBUG_PREFIX} Requesting balance from {self.valves.BALANCE_API_URL}")

            headers = {"Authorization": f"Bearer {self.valves.BALANCE_API_KEY}"}
            response = requests.get(self.valves.BALANCE_API_URL, headers=headers)
            response.raise_for_status()

            data = response.json()
            balance = data["balance"]

            if self.valves.DEBUG:
                print(f"{Config.DEBUG_PREFIX} API returned balance: {balance}")

            return f"Current API balance: {balance:.2f}"

        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                error_msg = f"Error retrieving balance: {e.response.status_code} - {e.response.text}"
            else:
                error_msg = f"Error retrieving balance: {str(e)}"
            if self.valves.DEBUG:
                print(f"{Config.DEBUG_PREFIX} {error_msg}")
            return error_msg

    def get_table_schema(self):
        """Get the usage_costs table schema based on database type"""
        is_sqlite = "sqlite" in engine.url.drivername
        
        if is_sqlite:
            query = "PRAGMA table_info(usage_costs);"
        else:
            query = """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'usage_costs'
                ORDER BY ordinal_position;
            """
            
        with get_db() as db:
            result = db.execute(text(query))
            rows = result.fetchall()
            
        if is_sqlite:
            schema = "\n".join([f"- {row.name}: {row.type}" for row in rows])
        else:
            schema = "\n".join([f"- {row.column_name}: {row.data_type}" for row in rows])
            
        return schema

    def get_user_api_key(self, user_id):
        """Get the user's API key from the database"""
        is_sqlite = "sqlite" in engine.url.drivername
        
        if is_sqlite:
            query = "SELECT api_key FROM user WHERE id = :user_id;"
        else:
            # For PostgreSQL, explicitly use the public schema
            query = "SELECT api_key FROM public.user WHERE id = :user_id;"
            
        if self.valves.DEBUG:
            print(f"{Config.DEBUG_PREFIX} Getting API key for user ID {user_id} using query: {query}")
            
        with get_db() as db:
            result = db.execute(text(query), {"user_id": user_id})
            row = result.fetchone()
            
            if self.valves.DEBUG and not row:
                print(f"{Config.DEBUG_PREFIX} No API key found for user ID {user_id}")
                
            return row.api_key if row else None

    async def handle_ask_command(self, __user__, body, question):
        """Handle natural language questions about usage data"""
        if self.valves.DEBUG:
            print(f"{Config.DEBUG_PREFIX} User {__user__['email']} asked: {question}")

        # Get user's API key
        api_key = self.get_user_api_key(__user__["id"])
        if not api_key:
            return ("Error: You must have an API key generated to use this feature.\n"
                   "Please go to Settings -> Account to generate an API key.")

        # Get database type and schema
        is_sqlite = "sqlite" in engine.url.drivername
        db_type = "SQLite" if is_sqlite else "PostgreSQL"
        schema = self.get_table_schema()

        # Construct the prompt
        prompt = (get_messages_content(body["messages"]) +      
                        f"""^^ THIS WAS PRIOR CONVERSATION CONTEXT^^
                        
                        NOW, you are a SQL query generator. Generate a SQL query for the following question:

                Question: {question}

                Database Type: {db_type}
                Table: usage_costs
                Schema:
                {schema}

                Note: the Task column is NULL for the regular chat requests; Task can be "title_generation", "tags_generation", "query_generation", "autocomplete_generation" made by the UI tool that accompany chats.
                Make a reasonable assumption about the users intention if they want information only from main chat completion requests (Task is NULL) or to include task usage. 
                For costs summarization, typically all tasks can be included. If a breakdown by model is requested, probably only main chat completions should be included. 
                For counting usage/requests, only main chat completions should be included. If unsure, consider building the report to separately highlight both numbers.

                The query must start with SELECT and end with a semicolon. Generate only the SQL query, nothing else. Do not use WITH or CTE clauses.""")

        try:
            # Create AsyncOpenAI client
            client = AsyncOpenAI(
                base_url="http://localhost:8080/api",
                api_key=api_key
            )

            # Get SQL query from LLM using async call
            completion = await client.chat.completions.create(
                model=self.valves.SQL_ASSISTANT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a SQL expert assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract and clean query from response
            sql_query = completion.choices[0].message.content.strip()
            sql_query = re.sub(r'^```sql\s*|\s*```$', '', sql_query, flags=re.MULTILINE)
            sql_query = sql_query.strip()

            if self.valves.DEBUG:
                print(f"{Config.DEBUG_PREFIX} Generated SQL for {__user__['email']}:\n{sql_query}")

            # Validate query
            if not re.match(r'^SELECT', sql_query, re.IGNORECASE):
                return "Error: executable query not obtained.\n" + sql_query 
            
            if not sql_query.rstrip().endswith(';'):
                sql_query += ';'

            if self.valves.DEBUG:
                print(f"{Config.DEBUG_PREFIX} Running SQL for {__user__['email']}...")

            # Execute the query
            result = self.run_sql_command(sql_query)

            # Format the response
            response = "Generated SQL Query:\n```sql\n"
            response += sql_query + "\n```\n\n"
            response += result

            if self.valves.DEBUG:
                print(f"{Config.DEBUG_PREFIX} Returning formatted response to {__user__['email']}")

            return response

        except Exception as e:
            _, _, tb = sys.exc_info()
            error_msg = f"Error on line {tb.tb_lineno}: {str(e)}"
            print(f"{Config.DEBUG_PREFIX} Error processing ask command for {__user__['email']}: {error_msg}")
            return f"Error processing question: {str(e)}"
