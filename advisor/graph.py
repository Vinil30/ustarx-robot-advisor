"""Robot Advisor agent as a LangGraph state machine.

understand -> retrieve -> draft -> verify --ok--> qualify_lead -> END
                            ^         |
                            +-- retry-+  (max 2 retries with the verifier's feedback)
                                      |
                                      +-- still failing --> safe_fallback -> qualify_lead
understand --off topic--> off_topic -> END
"""
import json
import time
from pathlib import Path
from typing import Literal, Optional, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .catalog import Catalog, fact_sheet, price_text
from .pricing import cost_facts, cost_text
from .verify import check_reply

MAX_RETRIES = 2
LEADS_PATH = Path(__file__).resolve().parent.parent / "data" / "leads.jsonl"


# ---------- LLM interface (Groq in the app, a fake in tests) ----------

class LLM(Protocol):
    def structured(self, schema: type[BaseModel], system: str, user: str) -> BaseModel: ...


class Needs(BaseModel):
    """What the customer is looking for, extracted from the conversation so far."""
    off_topic: bool = Field(description="True only if the message has nothing to do with robots, automation or USTARX")
    industry: Optional[str] = Field(None, description="e.g. restaurant, hospital, school, factory")
    use_case: Optional[str] = Field(None, description="What the robot should do, in a few words")
    search_query: str = Field(description="Keywords to search the robot catalog, e.g. 'restaurant food delivery'")
    budget_usd: Optional[int] = Field(None, description="Total budget in USD if stated")
    quantity: Optional[int] = Field(None, description="Number of robots if stated")
    prefers: Literal["buy", "lease", "unknown"] = "unknown"
    wants_virtual_agents: bool = Field(False, description="Asks about chatbots, AI agents, CRM, software automation")
    wants_demo_or_quote: bool = Field(False, description="True only if the customer explicitly asks to book a demo, "
                                      "receive a formal quote, or be contacted. A price question alone is NOT this.")
    name: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None


class Draft(BaseModel):
    reply: str = Field(description="The message to the customer")
    recommended_ids: list[str] = Field(description="Catalog ids of the products recommended in the reply")


# ---------- State ----------

class State(TypedDict, total=False):
    user_msg: str
    history: list[dict]          # [{"role": "user"|"assistant", "text": ...}]
    lead: dict                   # merged across turns
    needs: dict
    retrieved_ids: list[str]
    facts: str
    reply: str
    recommended_ids: list[str]
    problems: list[str]
    attempts: int
    status: str                  # grounded | fallback | off_topic
    trace: list[dict]


UNDERSTAND_SYSTEM = """You extract a customer's needs for USTARX AI, a company that sells and leases
service robots (delivery, reception humanoids, quadrupeds for inspection/security, cleaning,
education/ROS kits, dexterous hands) and virtual AI agents. Read the whole conversation and fill
the schema using only what the customer actually said. Leave unknown fields empty."""

DRAFT_SYSTEM = """You are the USTARX AI Robot Advisor on the company website.
Rules (strict):
- Recommend ONLY products listed in FACTS. Use their exact names.
- Every price, number or spec you write must appear in FACTS. Never estimate or round.
- If a product's price is not published, say so and offer a quote. Do not guess a price.
- If nothing in FACTS fits, say so honestly and offer to connect them with a specialist.
- If budget is given, say clearly which options fit it.
- At most 2 recommendations, plain language, under 130 words.
Format (markdown):
1. One short opening sentence that reflects their need.
2. One bullet per product: "- **Exact Product Name**: why it fits, in one line. Price."
3. If useful, one short line on buy vs lease using the given numbers.
4. End with ONE helpful question on its own line (e.g. venue size, buy vs lease, or a demo).
No headings, no tables, no emojis."""


def _history_text(history: list[dict], limit: int = 6) -> str:
    return "\n".join(f"{m['role']}: {m['text']}" for m in history[-limit:])


def build_graph(llm: LLM, catalog: Catalog, leads_path: Path = LEADS_PATH):

    def understand(state: State) -> State:
        convo = _history_text(state.get("history", [])) + f"\nuser: {state['user_msg']}"
        needs = llm.structured(Needs, UNDERSTAND_SYSTEM, convo).model_dump()
        lead = dict(state.get("lead") or {})
        for k, v in needs.items():
            if v not in (None, "", "unknown", False) or k not in lead:
                lead[k] = v
        return {"needs": needs, "lead": lead, "attempts": 0,
                "trace": [{"step": "understand", "needs": needs}]}

    def route_topic(state: State) -> str:
        return "off_topic" if state["needs"]["off_topic"] else "retrieve"

    def off_topic(state: State) -> State:
        return {"status": "off_topic", "recommended_ids": [],
                "reply": "I'm the USTARX Robot Advisor, so I can help you pick a robot or AI agent for your "
                         "business, compare buying vs leasing, or book a demo. What would you like a robot to do for you?",
                "trace": state["trace"] + [{"step": "off_topic"}]}

    def retrieve(state: State) -> State:
        lead = state["lead"]
        raw = " ".join(filter(None, [state["needs"]["search_query"], lead.get("use_case"), lead.get("industry")]))
        query = " ".join(dict.fromkeys(raw.lower().split()))   # drop repeated words, keep order
        robots = catalog.search(query, k=4)
        blocks = []
        for r in robots:
            block = fact_sheet(r)
            c = cost_facts(r, lead.get("quantity"))
            if c:
                block += "\n" + cost_text(c)
                if lead.get("budget_usd"):
                    fits = c["buy_total_usd"] <= lead["budget_usd"]
                    block += f"\n  Within stated budget of ${lead['budget_usd']:,} (buying): {'yes' if fits else 'no'}"
            blocks.append(block)
        if lead.get("wants_virtual_agents"):
            blocks += [f"[{a['id']}] {a['name']} (virtual AI agent): {a['summary']} Page: {a['url']}"
                       for a in catalog.virtual_agents]
        facts = "\n\n".join(blocks) if blocks else "(no matching products found)"
        return {"retrieved_ids": [r["id"] for r in robots], "facts": facts,
                "trace": state["trace"] + [{"step": "retrieve", "query": query,
                                            "results": [r["name"] for r in robots]}]}

    def draft(state: State) -> State:
        lead = {k: v for k, v in state["lead"].items() if v not in (None, "", False, "unknown")}
        prompt = (f"CUSTOMER NEEDS SO FAR: {json.dumps(lead)}\n\n"
                  f"CONVERSATION:\n{_history_text(state.get('history', []))}\nuser: {state['user_msg']}\n\n"
                  f"FACTS:\n{state['facts']}")
        if state.get("problems"):
            prompt += ("\n\nYOUR PREVIOUS DRAFT WAS REJECTED BY THE FACT CHECKER:\n- "
                       + "\n- ".join(state["problems"])
                       + f"\nPrevious draft:\n{state['reply']}\nWrite a corrected reply.")
        d = llm.structured(Draft, DRAFT_SYSTEM, prompt)
        return {"reply": d.reply, "recommended_ids": d.recommended_ids,
                "attempts": state.get("attempts", 0) + 1,
                "trace": state["trace"] + [{"step": "draft", "attempt": state.get("attempts", 0) + 1,
                                            "reply": d.reply}]}

    def verify(state: State) -> State:
        agent_ids = {a["id"] for a in catalog.virtual_agents}
        allowed = set(state["retrieved_ids"]) | (agent_ids if state["lead"].get("wants_virtual_agents") else set())
        problems = check_reply(state["reply"], state["facts"], state["user_msg"],
                               state["recommended_ids"], allowed, catalog)
        return {"problems": problems,
                "trace": state["trace"] + [{"step": "verify", "ok": not problems, "problems": problems}]}

    def route_verify(state: State) -> str:
        if not state["problems"]:
            return "ok"
        return "retry" if state["attempts"] <= MAX_RETRIES else "give_up"

    def mark_grounded(state: State) -> State:
        return {"status": "grounded"}

    def safe_fallback(state: State) -> State:
        """Model kept failing the fact check: answer only with catalog text, no generated claims."""
        robots = [catalog.by_id[i] for i in state["retrieved_ids"][:2]]
        if robots:
            lines = ["Here is what I can confirm from our catalog:"]
            lines += [f"- {r['name']}: {price_text(r)}. {r['url']}" for r in robots]
            lines.append("A USTARX specialist can confirm details for your exact setup. Shall I book a demo?")
        else:
            lines = ["I couldn't find a product that clearly fits that need. A USTARX specialist can help. "
                     "Could you share your email so we can contact you?"]
        return {"status": "fallback", "reply": "\n".join(lines), "recommended_ids": [r["id"] for r in robots],
                "trace": state["trace"] + [{"step": "safe_fallback"}]}

    def qualify_lead(state: State) -> State:
        lead = state["lead"]
        score = (25 * bool(lead.get("use_case")) + 20 * bool(lead.get("budget_usd")) +
                 15 * bool(lead.get("quantity")) + 25 * bool(lead.get("wants_demo_or_quote")) +
                 15 * bool(lead.get("email")))
        tier = "hot" if score >= 70 else "warm" if score >= 40 else "cold"
        reply = state["reply"]
        if lead.get("wants_demo_or_quote") and not lead.get("email"):
            reply += "\n\nTo set up the demo or quote, could you share your name and email?"
        saved = False
        if lead.get("email"):
            record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "score": score, "tier": tier,
                      "recommended": state.get("recommended_ids", []), **lead}
            leads_path.parent.mkdir(parents=True, exist_ok=True)
            with leads_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            saved = True
        return {"reply": reply, "lead": {**lead, "score": score, "tier": tier},
                "trace": state["trace"] + [{"step": "qualify_lead", "score": score, "tier": tier, "saved": saved}]}

    g = StateGraph(State)
    for name, fn in [("understand", understand), ("off_topic", off_topic), ("retrieve", retrieve),
                     ("draft", draft), ("verify", verify), ("mark_grounded", mark_grounded),
                     ("safe_fallback", safe_fallback), ("qualify_lead", qualify_lead)]:
        g.add_node(name, fn)
    g.add_edge(START, "understand")
    g.add_conditional_edges("understand", route_topic, {"off_topic": "off_topic", "retrieve": "retrieve"})
    g.add_edge("off_topic", END)
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route_verify,
                            {"ok": "mark_grounded", "retry": "draft", "give_up": "safe_fallback"})
    g.add_edge("mark_grounded", "qualify_lead")
    g.add_edge("safe_fallback", "qualify_lead")
    g.add_edge("qualify_lead", END)
    return g.compile()


class GroqLLM:
    def __init__(self, model: str, api_key: str):
        from langchain_groq import ChatGroq
        self.chat = ChatGroq(model=model, api_key=api_key, temperature=0.2)

    def structured(self, schema, system, user):
        return self.chat.with_structured_output(schema).invoke([("system", system), ("user", user)])
