"""Failure-aware gate: check a drafted reply against the facts before the customer sees it.

A sales assistant that invents a price or a spec creates a real business problem
(a quote the company can't honour). So every number and every product name in the
reply must trace back to the fact sheets that were given to the model.
"""
import re

from .catalog import Catalog

# $9,900 | 825 | 1.2 | 40 — but not list markers like "1." at a line start.
_NUM = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)


def _numbers(text: str) -> set[str]:
    text = _LIST_MARKER.sub(" ", text)
    out = set()
    for m in _NUM.findall(text):
        n = m.replace("$", "").replace(",", "")
        if n.endswith(".0"):
            n = n[:-2]
        out.add(n)
    return out


def check_reply(reply: str, allowed_text: str, user_text: str,
                recommended_ids: list[str], allowed_ids: set[str], catalog: Catalog) -> list[str]:
    """Return a list of problems. Empty list = grounded."""
    problems = []

    unknown_ids = [i for i in recommended_ids if i not in allowed_ids]
    if unknown_ids:
        problems.append(f"Recommended products that were not in the retrieved facts: {unknown_ids}")

    mentioned = catalog.mentioned_names(reply)
    ungrounded = mentioned - allowed_ids
    if ungrounded:
        names = [catalog.all_by_id[i]["name"] for i in sorted(ungrounded)]
        problems.append(f"Mentions products without facts for them: {names}")

    allowed_numbers = _numbers(allowed_text) | _numbers(user_text)
    invented = sorted(n for n in _numbers(reply) - allowed_numbers if n not in {"0", "1"})
    if invented:
        problems.append(f"Numbers not found in the facts (possible invented price/spec): {invented}")

    for rid in mentioned & allowed_ids:
        robot = catalog.all_by_id[rid]
        if robot["buy_usd"] is None and re.search(re.escape(robot["name"]) + r"[^.\n]{0,80}\$\d", reply, re.I):
            problems.append(f"Quoted a price for {robot['name']}, whose price is not published")

    return problems
