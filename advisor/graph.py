"""Robot Advisor agent as a LangGraph state machine with adaptive (agentic) retrieval.

                      ┌─ off_topic ──────────────────────────────────────────► END
                      ├─ smalltalk ──► chat (no retrieval) ──────────────────► END
understand ─(intent)──┤
                      └─ everything else ─► retrieve ─► select ─► draft ─► verify ──ok──► qualify_lead ─► END
                                                                    ▲        │
                                                                    └─retry──┤ (max 2, with checker feedback)
                                                                             └─still failing─► safe_fallback

retrieve decides HOW to look things up:
  * products the customer named          -> direct lookup (no search)
  * follow-up about earlier products     -> reuse products from previous turns (memory)
  * a need ("deliver food", "cook food") -> BM25 candidates
select: an LLM picks the best 1-3 products from the whole catalog (with BM25 candidates as hints), and says
whether they are an exact match or only the closest alternatives. This is what lets the agent suggest
alternatives when keyword search finds nothing.
"""
import json
import time
from pathlib import Path
from typing import Literal, Optional, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .catalog import Catalog, fact_sheet, normalize, price_text
from .pricing import cost_facts, cost_text
from .verify import check_reply

MAX_RETRIES = 2
LEADS_PATH = Path(__file__).resolve().parent.parent / "data" / "leads.jsonl"

Intent = Literal["smalltalk", "recommend", "product_info", "compare", "browse",
                 "general_question", "follow_up", "contact_details", "off_topic"]


# ---------- LLM interface (Groq in the app, a fake in tests) ----------

class LLM(Protocol):
    def structured(self, schema: type[BaseModel], system: str, user: str) -> BaseModel: ...
    def text(self, system: str, user: str) -> str: ...


class Needs(BaseModel):
    """What the customer wants, extracted from the conversation so far."""
    intent: Intent = Field(description=(
        "smalltalk: greeting/thanks/ok with no request. "
        "recommend: describes a need or task and wants a suggestion. "
        "product_info: asks about a specific product (price, specs, what it does). "
        "compare: compares specific products. "
        "browse: asks what is sold, cheapest, most capable, overview. "
        "general_question: robotics/AI knowledge question (e.g. what is SLAM). "
        "follow_up: continues about products already discussed (e.g. 'what about leasing it?'). "
        "contact_details: mainly giving name/email or asking for a demo/quote. "
        "off_topic: unrelated to robots, AI agents, automation or USTARX."))
    industry: Optional[str] = Field(None, description="e.g. restaurant, hospital, school, factory")
    use_case: Optional[str] = Field(None, description="What the robot should do, in a few words")
    search_query: str = Field("", description="Keywords to search the catalog, e.g. 'restaurant food delivery'")
    budget_usd: Optional[int] = Field(None, description="Total budget in USD if stated")
    quantity: Optional[int] = Field(None, description="Number of robots if stated")
    prefers: Literal["buy", "lease", "unknown"] = "unknown"
    wants_virtual_agents: bool = Field(False, description="Asks about chatbots, AI agents, CRM, software automation")
    wants_demo_or_quote: bool = Field(False, description="True only if the customer explicitly asks to book a demo, "
                                      "receive a formal quote, or be contacted. A price question alone is NOT this.")
    name: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None


class Selection(BaseModel):
    product_ids: list[str] = Field(description="1-3 catalog ids, best first. Empty only if truly nothing is relevant.")
    exact_match: bool = Field(description="True if these products directly do what the customer asked. "
                                          "False if they are only the closest alternatives.")
    reason: str = Field(description="One short sentence on why these were chosen")


# ---------- State ----------

class State(TypedDict, total=False):
    user_msg: str
    history: list[dict]          # [{"role": "user"|"assistant", "text": ...}]
    lead: dict                   # merged across turns; lead["last_products"] = products discussed last turn
    needs: dict
    candidates: list[str]
    selected: list[str]
    exact_match: bool
    retrieved_ids: list[str]     # products the reply is allowed to mention
    facts: str
    reply: str
    recommended_ids: list[str]
    problems: list[str]
    attempts: int
    status: str                  # grounded | fallback | conversational | off_topic
    trace: list[dict]


UNDERSTAND_SYSTEM = """You read a conversation on the USTARX AI website. USTARX sells and leases service robots
(food delivery, reception and performance humanoids, quadrupeds for inspection/security/education, a cleaning
robot, a wheeled dual-arm factory robot, dexterous robot hands) and virtual AI agents (customer service, sales,
operations, industry assistants). Classify the latest message's intent and fill the schema using only what the
customer actually said. Leave unknown fields empty."""

SELECT_SYSTEM = """You pick products from the USTARX catalog for a customer. Choose the 1-3 products that best
serve their need, best first. If nothing does exactly what they ask, choose the closest alternatives that could
still help (for example, for 'a robot that cooks', a robot that delivers food or handles items) and set
exact_match to false. Only use ids that appear in the catalog. Prefer the hinted candidates when they fit."""

DRAFT_SYSTEM = """You are the USTARX AI Robot Advisor: a knowledgeable, warm sales engineer on the company website.

Grounding rules (strict, a fact checker verifies them):
- Mention ONLY products listed in FACTS, with their exact names.
- Every price, number or spec you write must appear in FACTS or in the customer's message. Never estimate,
  round, or do your own arithmetic; use the cost lines given.
- Describe a product's purpose only from its "About" and "Good for" lines. Write it in your own words;
  never paste the "Good for" list verbatim.
- Never say something is included (deployment, mapping, training, support, service, upgrades, warranty)
  unless FACTS states it for that specific product. Don't attribute one product's specs to another.
- If a price is not published, say so and offer a quote.

How to answer:
- Be genuinely helpful and descriptive (about 150-250 words), in plain language.
- If MATCH is "closest alternatives": first say honestly in one sentence that USTARX doesn't offer exactly that,
  then present the alternatives and explain concretely how each could still help with their goal.
- For each product: a bullet starting with "**Exact Name**", then 2-3 sentences: what it is, why it fits
  THEIR situation, and 2-3 relevant specs from FACTS. Then its price (buy and lease).
- Then a short paragraph with practical advice: buy vs lease using the given cost lines, and what to check
  before choosing (only mention inclusions FACTS states).
- For general knowledge questions: answer the question clearly and accurately first (3-5 sentences, no
  numbers), then briefly mention relevant products from FACTS if there are any; if FACTS has no products,
  don't name any.
- For comparisons: compare on the dimensions that matter for their use, then give a clear recommendation.
- End with ONE specific follow-up question on its own line.
Format: markdown with short paragraphs and "- " bullets. No headings, no tables, no emojis.
Reply with the message text only."""

CHAT_SYSTEM = """You are the USTARX AI Robot Advisor. The customer sent a greeting or short social message.
Reply warmly in 1-2 sentences and ask what they'd like a robot or AI agent to help with (for example in a
restaurant, hospital, school, factory or office). Do not name any products and do not use numbers."""


def _history_text(history: list[dict], limit: int = 6) -> str:
    return "\n".join(f"{m['role']}: {m['text']}" for m in history[-limit:])


def build_graph(llm: LLM, catalog: Catalog, leads_path: Path = LEADS_PATH):

    def understand(state: State) -> State:
        convo = _history_text(state.get("history", [])) + f"\nuser: {state['user_msg']}"
        try:
            needs = llm.structured(Needs, UNDERSTAND_SYSTEM, convo).model_dump()
        except Exception:
            # Extraction failed: carry on as a recommendation request using the raw message.
            needs = Needs(intent="recommend", search_query=state["user_msg"]).model_dump()
        lead = dict(state.get("lead") or {})
        for k, v in needs.items():
            if k == "intent":
                continue
            if v not in (None, "", "unknown", False) or k not in lead:
                lead[k] = v
        return {"needs": needs, "lead": lead, "attempts": 0, "problems": [],
                "trace": [{"step": "understand", "intent": needs["intent"], "needs": needs}]}

    def route_intent(state: State) -> str:
        intent = state["needs"]["intent"]
        return intent if intent in ("off_topic", "smalltalk") else "retrieve"

    def off_topic(state: State) -> State:
        return {"status": "off_topic", "recommended_ids": [],
                "reply": "I'm the USTARX Robot Advisor, so I can help you choose a robot or AI agent for your "
                         "business, compare buying and leasing, or arrange a demo.\n\n"
                         "What would you like a robot to help you with?",
                "trace": state["trace"] + [{"step": "route", "decision": "off topic: no lookup"}]}

    def chat(state: State) -> State:
        try:
            reply = llm.text(CHAT_SYSTEM, f"Customer: {state['user_msg']}").strip()
        except Exception:
            reply = ""
        # Small talk must not make product or price claims; if it does, use a safe greeting instead.
        if not reply or catalog.mentions_in_order(reply) or any(ch.isdigit() for ch in reply):
            reply = ("Hello! I'm the USTARX Robot Advisor. Tell me a little about your business and what you'd "
                     "like a robot or AI agent to help with, and I'll suggest the right fit.")
        return {"status": "conversational", "reply": reply, "recommended_ids": [],
                "trace": state["trace"] + [{"step": "route", "decision": "small talk: no catalog lookup needed"}]}

    def retrieve(state: State) -> State:
        needs, lead = state["needs"], state["lead"]
        intent = needs["intent"]
        named = [i for i in catalog.mentions_in_order(state["user_msg"])]
        previous = [i for i in lead.get("last_products", []) if i in catalog.all_by_id]
        raw = " ".join(filter(None, [needs.get("search_query"), needs.get("use_case") or lead.get("use_case"),
                                     needs.get("industry") or lead.get("industry")]))
        query = " ".join(dict.fromkeys(raw.lower().split()))
        bm25 = [r["id"] for r in catalog.search(query, k=6)] if query else []

        if named:
            how = "direct lookup of the products you named"
            candidates = named + [i for i in bm25 if i not in named]
        elif intent in ("follow_up", "contact_details") and previous:
            how = "reused products from earlier in the conversation"
            candidates = previous
        else:
            how = f"BM25 keyword search for \"{query}\"" if query else "no keywords; full catalog"
            candidates = previous + [i for i in bm25 if i not in previous] if intent == "follow_up" else bm25
        if needs["wants_virtual_agents"]:
            candidates = list(dict.fromkeys(candidates + list(catalog.agents_by_id)))
        return {"candidates": candidates,
                "trace": state["trace"] + [{"step": "retrieve", "how": how,
                                            "results": [catalog.all_by_id[i]["name"] for i in candidates[:6]]}]}

    def select(state: State) -> State:
        needs, lead = state["needs"], state["lead"]
        intent = needs["intent"]
        cands = state["candidates"]
        named = catalog.mentions_in_order(state["user_msg"])
        # Deterministic paths: the customer named products, or is continuing about earlier ones.
        if named and intent in ("product_info", "compare"):
            chosen, exact, reason = named[:3], True, "products named by the customer"
        elif intent in ("follow_up", "contact_details") and lead.get("last_products") and not named:
            chosen, exact, reason = cands[:3], True, "continuing with the products already discussed"
        else:
            hint = ", ".join(cands[:6]) or "(none)"
            ask = (f"Customer message: {state['user_msg']}\n"
                   f"Known needs: {json.dumps({k: v for k, v in lead.items() if v not in (None, '', False, 'unknown') and k != 'last_products'})}\n"
                   f"Keyword-search candidates (hints): {hint}\n\n"
                   f"{catalog.overview(include_agents=needs['wants_virtual_agents'])}")
            try:
                sel = llm.structured(Selection, SELECT_SYSTEM, ask)
                chosen = [i for i in sel.product_ids if i in catalog.all_by_id][:3]
                exact, reason = sel.exact_match, sel.reason
            except Exception:
                chosen, exact, reason = [], bool(cands), "selector unavailable; using search results"
            if not chosen:
                chosen, exact = cands[:3], bool(cands) and exact
            if intent == "browse":
                exact = True

        blocks = []
        for pid in chosen:
            item = catalog.all_by_id[pid]
            block = fact_sheet(item)
            c = cost_facts(item, lead.get("quantity")) if pid in catalog.by_id else None
            if c:
                block += "\n" + cost_text(c)
                if lead.get("budget_usd"):
                    fits = c["buy_total_usd"] <= lead["budget_usd"]
                    block += f"\n  Within stated budget of ${lead['budget_usd']:,} (buying): {'yes' if fits else 'no'}"
            blocks.append(block)
        allowed = list(chosen)
        if intent == "browse" or (not chosen and intent != "general_question"):
            blocks.append(catalog.overview(include_agents=needs["wants_virtual_agents"]))
            allowed = list(dict.fromkeys(allowed + list(catalog.all_by_id)))
        facts = "\n\n".join(blocks) if blocks else "(no products)"
        return {"selected": chosen, "exact_match": exact, "retrieved_ids": allowed, "facts": facts,
                "trace": state["trace"] + [{"step": "select", "chosen": [catalog.all_by_id[i]["name"] for i in chosen],
                                            "exact": exact, "reason": reason}]}

    def draft(state: State) -> State:
        lead = {k: v for k, v in state["lead"].items() if v not in (None, "", False, "unknown") and k != "last_products"}
        match = "exact" if state.get("exact_match", True) else "closest alternatives"
        prompt = (f"INTENT: {state['needs']['intent']}\nMATCH: {match}\n"
                  f"CUSTOMER NEEDS SO FAR: {json.dumps(lead)}\n\n"
                  f"CONVERSATION:\n{_history_text(state.get('history', []))}\nuser: {state['user_msg']}\n\n"
                  f"FACTS:\n{state['facts']}")
        if state.get("problems"):
            prompt += ("\n\nYOUR PREVIOUS DRAFT WAS REJECTED BY THE FACT CHECKER:\n- "
                       + "\n- ".join(state["problems"])
                       + f"\nPrevious draft:\n{state['reply']}\nWrite a corrected reply.")
        reply = normalize(llm.text(DRAFT_SYSTEM, prompt)).strip()
        # Which products the reply recommends is read from the text itself, not trusted from the model.
        recommended = catalog.mentions_in_order(reply)[:3]
        return {"reply": reply, "recommended_ids": recommended,
                "attempts": state.get("attempts", 0) + 1,
                "trace": state["trace"] + [{"step": "draft", "attempt": state.get("attempts", 0) + 1}]}

    def verify(state: State) -> State:
        problems = check_reply(state["reply"], state["facts"], state["user_msg"],
                               state["recommended_ids"], set(state["retrieved_ids"]), catalog)
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
        items = [catalog.all_by_id[i] for i in (state.get("selected") or state["retrieved_ids"])[:2]]
        if items:
            lines = ["Here is what I can confirm from the USTARX catalog:"]
            lines += [f"- **{r['name']}**: {r['summary']} {price_text(r)}." for r in items]
            lines.append("A USTARX specialist can confirm the details for your exact setup.")
            lines.append("Would you like me to arrange a demo?")
        else:
            lines = ["I couldn't find a product that clearly fits that need, but a USTARX specialist can help.",
                     "Could you share your email so the team can contact you?"]
        return {"status": "fallback", "reply": "\n".join(lines), "recommended_ids": [r["id"] for r in items],
                "trace": state["trace"] + [{"step": "safe_fallback"}]}

    def qualify_lead(state: State) -> State:
        lead = dict(state["lead"])
        score = (25 * bool(lead.get("use_case")) + 20 * bool(lead.get("budget_usd")) +
                 15 * bool(lead.get("quantity")) + 25 * bool(lead.get("wants_demo_or_quote")) +
                 15 * bool(lead.get("email")))
        tier = "hot" if score >= 70 else "warm" if score >= 40 else "cold"
        reply = state["reply"]
        if lead.get("wants_demo_or_quote") and not lead.get("email"):
            reply += "\n\nTo set up the demo or quote, could you share your name and email?"
        if state.get("recommended_ids"):
            lead["last_products"] = state["recommended_ids"]       # memory for follow-up questions
        saved = False
        if lead.get("email"):
            record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "score": score, "tier": tier,
                      "recommended": lead.get("last_products", []), **{k: v for k, v in lead.items() if k != "last_products"}}
            leads_path.parent.mkdir(parents=True, exist_ok=True)
            with leads_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            saved = True
        return {"reply": reply, "lead": {**lead, "score": score, "tier": tier},
                "trace": state["trace"] + [{"step": "qualify_lead", "score": score, "tier": tier, "saved": saved}]}

    g = StateGraph(State)
    for name, fn in [("understand", understand), ("off_topic", off_topic), ("chat", chat),
                     ("retrieve", retrieve), ("select", select), ("draft", draft), ("verify", verify),
                     ("mark_grounded", mark_grounded), ("safe_fallback", safe_fallback),
                     ("qualify_lead", qualify_lead)]:
        g.add_node(name, fn)
    g.add_edge(START, "understand")
    g.add_conditional_edges("understand", route_intent,
                            {"off_topic": "off_topic", "smalltalk": "chat", "retrieve": "retrieve"})
    g.add_edge("off_topic", END)
    g.add_edge("chat", END)
    g.add_edge("retrieve", "select")
    g.add_edge("select", "draft")
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

    def _retry(self, fn, tries: int = 3):
        """Groq sometimes rejects a generation (e.g. tool_use_failed). Retry a couple of times before giving up."""
        last = None
        for attempt in range(tries):
            try:
                return fn()
            except Exception as e:
                last = e
                time.sleep(0.4 * (attempt + 1))
        raise last

    def structured(self, schema, system, user):
        runnable = self.chat.with_structured_output(schema)
        return self._retry(lambda: runnable.invoke([("system", system), ("user", user)]))

    def text(self, system, user):
        return self._retry(lambda: self.chat.invoke([("system", system), ("user", user)]).content)
