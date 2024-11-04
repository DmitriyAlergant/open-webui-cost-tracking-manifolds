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

    def pipe(self, body: dict, __user__: dict) -> str:

        command = get_last_user_message(body["messages"])

        if self.valves.DEBUG:
            print(f"usage-reporting-bot ({__user__['email']}): {command}")

        if command == "/help":
            return self.print_help()

        else:
            return self.handle_command(__user__, command)

    def handle_command(self, __user__, command):
        if command == "/help":
            return self.help_command()

        if match := re.match(r"/usage_stats\s+all(?:\s+(\d+)d)?", command):
            days = int(match.group(1)) if match.group(1) else 30

            if __user__["role"] == "admin":
                return self.generate_all_users_report(days)
            else:
                return "Sorry, this feature is only available to Admins"

        if match := re.match(r"/usage_stats\s+([^\s]+@[^\s]+)(?:\s+(\d+)d)?", command):
            specific_user = match.group(1)
            days = int(match.group(2)) if match.group(2) else 30

            if __user__["role"] == "admin" or specific_user == __user__["email"]:
                return self.generate_single_user_report(days, specific_user)
            else:
                return "Sorry, this feature is only available to Admins"

        if match := re.match(r"/usage_stats(?:\s+(\d+)d)?", command):
            days = int(match.group(1)) if match.group(1) else 30

            return self.generate_single_user_report(days, __user__["email"])

        return "Invalid command format. Use '/help' for list of available commands."

    def print_help(self):
        return (
            "**Available Commands**\n"
            "* **/usage_stats 30d** - my own usage stats for 30 days\n"
            "* **/usage_stats all 45d** - stats by all users for 45 days\n"
            "* **/usage_stats user@email.com** - stats for the indicated user (default is 30 days)\n"
        )

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

        # Total usage costs by currency
        currency_totals = df.groupby("currency")["total_cost"].sum().round(2)

        # Prepare the report
        report = f"## Usage Report for All Users\n"
        report += f"### Period: {start_date.date()} to {end_date.date()}\n\n"
        report += "#### Total Usage Costs:\n"

        for currency, total in currency_totals.items():
            if currency in ["RUB", "RUR"]:
                report += f"- **{total:.2f} ₽**\n"
            elif currency == "USD":
                report += f"- **{total:.2f} $**\n"
            else:
                report += f"- **{total:.2f} {currency}**\n"

        # Top 20 users
        df["cost_rub"] = df.apply(
            lambda row: (
                row["total_cost"]
                if row["currency"] == "RUB"
                else row["total_cost"] * 100
            ),
            axis=1,
        )
        user_totals = (
            df.groupby("user_email")
            .agg(
                {
                    "cost_rub": "sum",
                }
            )
            .round(2)
            .nlargest(20, "cost_rub")
        )

        report += "\n#### Top 20 Users by Cost:\n"
        report += "| User | Cost (RUB) |\n"
        report += "|------|------------|\n"
        for user, row in user_totals.iterrows():
            report += f"| {user} | {row['cost_rub']:.2f} ₽ |\n"

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

        # Group by currency and sum the total cost
        currency_totals = df.groupby("currency")["total_cost"].sum().round(2)

        # Prepare the report
        report = f"## Usage Report for {user_email}\n"
        report += f"### Period: {start_date.date()} to {end_date.date()}\n\n"
        report += "#### Total Usage Costs:\n"

        for currency, total in currency_totals.items():
            if currency in ["RUB", "RUR"]:
                report += f"- **{total:.2f} ₽**\n"
            elif currency == "USD":
                report += f"- **{total:.2f} $**\n"
            else:
                report += f"- **{total:.2f} {currency}**\n"

        # Add total tokens information
        total_input_tokens = df["total_input_tokens"].sum()
        total_output_tokens = df["total_output_tokens"].sum()
        report += "\n#### Total Tokens Used:\n"
        report += f"- Input tokens:  **{total_input_tokens:,}**\n"
        report += f"- Output tokens: **{total_output_tokens:,}**\n"

        # Convert costs to USD for ranking
        df["cost_usd"] = df.apply(
            lambda row: (
                row["total_cost"]
                if row["currency"] == "USD"
                else row["total_cost"] / 100
            ),
            axis=1,
        )

        # Add top 5 models by usage
        top_models = df.groupby("model")["cost_usd"].sum().nlargest(5).round(2)
        report += "\n#### Top 5 Models by Cost:\n"
        for model, cost in top_models.items():
            original_currency = df[df["model"] == model]["currency"].iloc[0]
            original_cost = df[df["model"] == model]["total_cost"].sum().round(2)
            if original_currency in ["RUB", "RUR"]:
                report += f"- **{model}**: {original_cost:.2f} ₽\n"
            elif original_currency == "USD":
                report += f"- **{model}**: {original_cost:.2f} $\n"
            else:
                report += f"- **{model}**: {original_cost:.2f} {original_currency}\n"

        return report
