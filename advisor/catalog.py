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

    def mentioned_names(self, text: str) -> set[str]:
        """Product ids whose name appears in text. Longest names first so 'D1 Edu Pro' is not also read as 'D1 Edu'."""
        found, low = set(), text.lower()
        for r in sorted(self.robots, key=lambda r: -len(r["name"])):
            name = r["name"].lower()
            short = name.replace("agibot ", "")
            for variant in (name, short):
                if re.search(r"(?<![a-z0-9])" + re.escape(variant) + r"(?![a-z0-9])", low):
                    found.add(r["id"])
                    low = low.replace(variant, " ")
                    break
        return found


def price_text(robot: dict) -> str:
    if robot["buy_usd"] is None:
        return "Price: not published (contact USTARX for a quote)"
    return f"Buy: ${robot['buy_usd']:,} | Lease: ${robot['lease_usd_month']:,}/month"


def fact_sheet(robot: dict) -> str:
    specs = "; ".join(f"{k}: {v}" for k, v in robot["specs"].items())
    return (
        f"[{robot['id']}] {robot['name']} ({robot['category']})\n"
        f"  {price_text(robot)}\n"
        f"  About: {robot['summary']}\n"
        f"  Good for: {', '.join(robot['use_cases'])}\n"
        f"  Specs: {specs}\n"
        f"  Page: {robot['url']}"
    )
