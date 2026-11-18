"""Deterministic cost math. The LLM never does arithmetic; it only quotes these numbers."""
import math


def cost_facts(robot: dict, quantity: int | None) -> dict | None:
    buy, lease = robot["buy_usd"], robot["lease_usd_month"]
    if buy is None or lease is None:
        return None
    qty = quantity if quantity and quantity > 0 else 1
    return {
        "id": robot["id"],
        "quantity": qty,
        "buy_total_usd": buy * qty,
        "lease_monthly_total_usd": lease * qty,
        "lease_12_months_usd": lease * qty * 12,
        "lease_24_months_usd": lease * qty * 24,
        # Months of leasing after which buying would have been cheaper.
        "break_even_months": math.ceil(buy / lease),
    }


def cost_text(c: dict) -> str:
    return (
        f"  Cost for {c['quantity']} unit(s): buy total ${c['buy_total_usd']:,}; "
        f"lease ${c['lease_monthly_total_usd']:,}/month "
        f"(${c['lease_12_months_usd']:,} over 12 months, ${c['lease_24_months_usd']:,} over 24 months); "
        f"buying is cheaper than leasing after {c['break_even_months']} months."
    )
