"""Product catalog: loading, BM25 retrieval, and fact sheets the LLM is allowed to use."""
import json
import math
import re
from collections import Counter
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the and or for of to in on with is are be i we our my need want looking robot robots".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class Catalog:
    def __init__(self, path: Path = CATALOG_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.robots: list[dict] = data["robots"]
        self.virtual_agents: list[dict] = data["virtual_agents"]
        self.by_id = {r["id"]: r for r in self.robots}
        for a in self.virtual_agents:   # agents have no published price
            a.setdefault("category", "Virtual AI agent")
            a.setdefault("buy_usd", None)
            a.setdefault("lease_usd_month", None)
        self.agents_by_id = {a["id"]: a for a in self.virtual_agents}
        self.all_by_id = {**self.by_id, **self.agents_by_id}
        # Use-case tags are repeated so a direct match ("restaurant") outweighs a passing mention.
        self._docs = [
            tokenize(" ".join([r["name"], r["category"], r["summary"]] + r["use_cases"] * 3))
            for r in self.robots
        ]
        self._df = Counter(t for doc in self._docs for t in set(doc))
        self._avgdl = sum(len(d) for d in self._docs) / len(self._docs)

    def search(self, query: str, k: int = 4, k1: float = 1.5, b: float = 0.75) -> list[dict]:
        """BM25 over name + category + summary + use-case tags. Returns top-k robots with score > 0."""
        q = tokenize(query)
        n = len(self._docs)
        scored = []
        for robot, doc in zip(self.robots, self._docs):
            tf = Counter(doc)
            score = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / self._avgdl))
            if score > 0:
                scored.append((score, robot))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:k]]

    def mentions_in_order(self, text: str) -> list[str]:
        """Ids of robots/agents named in text, in order of first appearance.
        Longest names are matched first so 'D1 Edu Pro' is not also read as 'D1 Edu'."""
        low = text.lower()
        hits: dict[str, int] = {}
        for item in sorted(self.all_by_id.values(), key=lambda r: -len(r["name"])):
            name = item["name"].lower()
            for variant in dict.fromkeys((name, name.replace("agibot ", ""))):
                for m in re.finditer(r"(?<![a-z0-9])" + re.escape(variant) + r"(?![a-z0-9])", low):
                    hits[item["id"]] = min(hits.get(item["id"], m.start()), m.start())
                    # Blank out every occurrence of every variant so 'X2 Ultra' is never re-read as 'X2'.
                    low = low[:m.start()] + " " * len(variant) + low[m.end():]
        return sorted(hits, key=hits.get)

    def mentioned_names(self, text: str) -> set[str]:
        return set(self.mentions_in_order(text))

    def overview(self, include_agents: bool = False) -> str:
        """One line per product, for broad questions ('what do you sell?', 'cheapest robot?')."""
        lines = [f"[{r['id']}] {r['name']} ({r['category']}): {price_text(r)}. Good for: {', '.join(r['use_cases'][:7])}"
                 for r in self.robots]
        if include_agents:
            lines += [f"[{a['id']}] {a['name']} (virtual AI agent): {a['summary']}" for a in self.virtual_agents]
        return "FULL CATALOG OVERVIEW:\n" + "\n".join(lines)


def price_text(robot: dict) -> str:
    if robot.get("buy_usd") is None:
        return "Price: not published (contact USTARX for a quote)"
    return f"Buy: ${robot['buy_usd']:,} | Lease: ${robot['lease_usd_month']:,}/month"


def fact_sheet(robot: dict) -> str:
    if "specs" not in robot:   # virtual AI agent
        return (f"[{robot['id']}] {robot['name']} (Virtual AI agent)\n"
                f"  Price: not published (custom quote)\n  About: {robot['summary']}\n  Page: {robot['url']}")
    specs = "; ".join(f"{k}: {v}" for k, v in robot["specs"].items())
    return (
        f"[{robot['id']}] {robot['name']} ({robot['category']})\n"
        f"  {price_text(robot)}\n"
        f"  About: {robot['summary']}\n"
        f"  Good for: {', '.join(robot['use_cases'])}\n"
        f"  Specs: {specs}\n"
        f"  Page: {robot['url']}"
    )
