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

from open_webui.utils.misc import get_last_user_message

import pandas as pd

class Config:
    DEBUG_PREFIX = "DEBUG:    " + __name__ + " -"
    DEBUG = True


from decimal import Decimal
from datetime import datetime
from sqlalchemy import text
from open_webui.apps.webui.internal.db import get_db, engine


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
        

    def pipe(self, body: dict, __user__: dict) -> str:

        command = get_last_user_message(body["messages"])

        if self.valves.DEBUG:
            print(f"usage-reporting-bot ({__user__['email']}): {command}")

        if command == "/help":
            return self.print_help(__user__)

        else:
            return self.handle_command(__user__, command)

    def handle_command(self, __user__, command):

        if match := re.match(r"/usage_stats\s+all(?:\s+(\d+)d)?", command):
            days = int(match.group(1)) if match.group(1) else 30

            if self.is_superuser(__user__):
                return self.generate_all_users_report(days)
            else:
                return "Sorry, this feature is only available to Admins"

        if match := re.match(r"/usage_stats\s+([^\s]+@[^\s]+)(?:\s+(\d+)d)?", command):
            specific_user = match.group(1)
            days = int(match.group(2)) if match.group(2) else 30

            if self.is_superuser(__user__) or specific_user == __user__["email"]:
                return self.generate_single_user_report(days, specific_user)
            else:
                return "Sorry, this feature is only available to Admins"

        if match := re.match(r"/usage_stats(?:\s+(\d+)d)?", command):
            days = int(match.group(1)) if match.group(1) else 30

            return self.generate_single_user_report(days, __user__["email"])

        if command.startswith("/run_sql "):
            if self.is_superuser(__user__):
                return self.run_sql_command(command[len("/run_sql "):])  # Remove "/usage_sql " prefix
            else:
                return "Sorry, this feature is only available to Admins"

        return "Invalid command\n\n" + self.print_help(__user__)

    def print_help(self, __user__):
        help_message = (
            "**Available Commands**\n"
            "* **/usage_stats 30d** my own usage stats for 30 days\n\n"
        )
        
        if self.is_superuser(__user__):
            help_message += (
                "**Available Commands (Admins Only)**\n"
                "* **/usage_stats all 45d** stats by all users for 45 days\n"
                "* **/usage_stats user@email.com** stats for the indicated user (default is 30 days)\n"
                "* **/run_sql SELECT count(*) from usage_costs;** allows an admin to run arbitrary SQL SELECT from the database.\n  - For SQLite: use /run_sql PRAGMA table_info(usage_costs) to see available table columns\n  - For Postgres db: /run_sql SELECT * FROM information_schema.columns WHERE table_name = 'usage_costs'"
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
        """
        Generate a usage report for all users.

        :param days: Number of days to include in the report
        :return: A formatted string report of usage stats for all users
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get usage stats for all users
        stats = self.get_usage_stats(
            user_email=None, start_date=start_date, end_date=end_date
        )

        if not stats:
            return f"No usage data found for any users in the last {days} days."

        df = pd.DataFrame(stats)

        # Default currency to 'USD' if blank or null
        df['currency'] = df['currency'].fillna('USD').replace('', 'USD')

        # Get unique currencies in data
        currencies = df['currency'].unique()

        if len(currencies) > 1 or 'USD' not in currencies:
            exchange_rates = self.get_exchange_rates(currencies)
        else:
            exchange_rates = {'USD': 1.0}

        # Total usage costs by currency
        currency_totals = df.groupby("currency")["total_cost"].sum().round(2)

        # Convert all totals to USD for sorting
        usd_equivalent_totals = {
            currency: total / exchange_rates.get(currency, 1.0)
            for currency, total in currency_totals.items()
        }

        # Sort currencies by their USD equivalent value
        sorted_currencies = sorted(
            usd_equivalent_totals.keys(),
            key=lambda x: usd_equivalent_totals[x],
            reverse=True
        )

        # Prepare the report
        report = f"## Usage Report for All Users\n"
        report += f"### Period: {start_date.date()} to {end_date.date()}\n\n"
        report += "#### Total Usage Costs:\n"

        for currency in sorted_currencies:
            total = currency_totals[currency]
            if currency == "USD":
                report += f"- $ **{total:.2f}**\n"
            else:
                report += f"- **{total:.2f} {currency}**\n"

        # Top 20 users

        # Convert costs to USD for ranking
        df['cost_for_ranking'] = df.apply(
            lambda row: row['total_cost'] / exchange_rates.get(row['currency'], 1.0),
            axis=1
        )

        # Separate costs per currency for reporting
        for currency in currencies:
            df[f'{currency.lower()}_cost'] = df.apply(
                lambda row: row['total_cost'] if row['currency'] == currency else 0,
                axis=1
            )

        # Get user totals and select top 20 users

        agg_dict = {f'{currency.lower()}_cost': 'sum' for currency in currencies}
        agg_dict['cost_for_ranking'] = 'sum'

        user_totals = df.groupby("user_email").agg(agg_dict).round(2)

        top_users = user_totals.nlargest(20, "cost_for_ranking")

        # Prepare data for table rendering
        
        headers = ["User"] + [currency.upper() for currency in currencies if (top_users[f'{currency.lower()}_cost'] > 0).any()]
        rows = []
        for user, row in top_users.iterrows():
            row_data = [user]
            for currency in currencies:
                if (top_users[f'{currency.lower()}_cost'] > 0).any():
                    cost = row[f'{currency.lower()}_cost']
                    if currency == 'USD':
                        row_data.append(f"${cost:.2f}")
                    else:
                        row_data.append(f"{cost:.2f} {currency}")
            rows.append(row_data)

        # Render an ASCII table
        col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

        table = "```\n"  # Start code block for fixed-width formatting
        table += " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths)) + "\n"
        table += "-|-".join("-" * width for width in col_widths) + "\n"
        for row in rows:
            table += " | ".join(f"{cell:<{width}}" for cell, width in zip(row, col_widths)) + "\n"
        table += "```\n"  # End code block

        report += "\n#### Top 20 Users by Cost:\n"
        report += table

        return report

    def generate_single_user_report(self, days: int, user_email: str):
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

        # Group by currency and sum the total cost
        currency_totals = df.groupby("currency")["total_cost"].sum().round(2)

        # Prepare the report
        report = f"## Usage Report for {user_email}\n"
        report += f"### Period: {start_date.date()} to {end_date.date()}\n\n"
        report += "#### Total Usage Costs:\n"

        for currency, total in currency_totals.items():
            if currency == "USD":
                report += f"- $ **{total:.2f}**\n"
            else:
                report += f"- **{total:.2f} {currency}**\n"

        # Add total tokens information
        total_input_tokens = df["total_input_tokens"].sum()
        total_output_tokens = df["total_output_tokens"].sum()
        report += "\n#### Total Tokens Used:\n"
        report += f"- Input tokens:  **{total_input_tokens:,}**\n"
        report += f"- Output tokens: **{total_output_tokens:,}**\n"


        # TOP 5 MODELS BY COST
        report += "\n#### Top 5 Models by Cost:\n"

        # Group data and select top 5 models by USD cost (currency-converted)

        model_costs = df.groupby(['model', 'currency'])['total_cost'].sum().reset_index()

        model_costs['cost_usd'] = model_costs.apply(
            lambda row: row['total_cost'] / exchange_rates.get(row['currency'], 1.0),
            axis=1
        )

        top_models = model_costs.groupby('model')['cost_usd'].sum().nlargest(5).index

        top_model_data = model_costs[model_costs['model'].isin(top_models)]

        # Prepare data for table rendering

        headers = ["Model"] + list(currencies)
        rows = []
        for model in top_models:
            row_data = [model]
            for currency in currencies:
                cost = top_model_data[(top_model_data['model'] == model) & (top_model_data['currency'] == currency)]['total_cost'].sum()
                if currency == 'USD':
                    row_data.append(f"${cost:.2f}" if cost > 0 else "")
                else:
                    row_data.append(f"{cost:.2f} {currency}" if cost > 0 else "")
            rows.append(row_data)

        # Render an ASCII table
        col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

        table = "```\n"  # Start code block for fixed-width formatting
        table += " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths)) + "\n"
        table += "-|-".join("-" * width for width in col_widths) + "\n"
        for row in rows:
            table += " | ".join(f"{cell:<{width}}" for cell, width in zip(row, col_widths)) + "\n"
        table += "```\n"  # End code block

        report += table

        return report
    
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
                    formatted_row = [str(getattr(row, col, '')) for col in headers]
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
