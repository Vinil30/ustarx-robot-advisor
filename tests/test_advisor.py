"""Offline tests: a scripted fake LLM drives the real graph, retrieval, pricing and verifier."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advisor.catalog import Catalog
from advisor.graph import Needs, Selection, build_graph
from advisor.pricing import cost_facts
from advisor.verify import check_reply

CAT = Catalog()


class FakeLLM:
    """Scripted LLM. needs: queue of Needs (or Exception to raise). selections: queue of Selection
    (empty queue = selector returns nothing, so the graph falls back to search candidates). drafts: reply strings."""
    def __init__(self, needs, drafts, selections=()):
        self.needs, self.drafts, self.selections = list(needs), list(drafts), list(selections)
        self.draft_prompts, self.select_prompts, self.calls = [], [], []

    def structured(self, schema, system, user):
        self.calls.append(schema.__name__)
        if schema is Needs:
            n = self.needs.pop(0)
            if isinstance(n, Exception):
                raise n
            return n
        self.select_prompts.append(user)
        if self.selections:
            return self.selections.pop(0)
        return Selection(product_ids=[], exact_match=True, reason="")

    def text(self, system, user):
        self.calls.append("text")
        self.draft_prompts.append(user)
        return self.drafts.pop(0)


def run(llm, msg, tmp_path, lead=None, history=None):
    graph = build_graph(llm, CAT, leads_path=tmp_path / "leads.jsonl")
    return graph.invoke({"user_msg": msg, "history": history or [], "lead": lead or {}})


def restaurant_needs(**kw):
    base = dict(intent="recommend", industry="restaurant", use_case="deliver food to tables",
                search_query="restaurant food delivery", quantity=2)
    base.update(kw)
    return Needs(**base)


# ---------- retrieval, pricing, verifier ----------

def test_retrieval_finds_right_robot():
    assert CAT.search("restaurant food delivery")[0]["id"] == "bellabot-pro"
    assert CAT.search("ros2 university robotics course")[0]["id"] == "agibot-d1-edu-pro"
    assert CAT.search("floor cleaning supermarket")[0]["id"] == "agibot-c5"
    assert CAT.search("hotel lobby receptionist guided tours")[0]["id"] in {"agibot-a2-ultra", "agibot-x2-ultra"}


def test_name_matching():
    assert CAT.mentioned_names("Try the D1 Edu Pro") == {"agibot-d1-edu-pro"}
    assert CAT.mentioned_names("AGIBOT D1 Edu and BellaBot Pro") == {"agibot-d1-edu", "bellabot-pro"}
    assert CAT.mentions_in_order("**AGIBOT X2 Ultra**: $42,999. Leasing three X2 Ultra units.") == ["agibot-x2-ultra"]


def test_pricing_math():
    c = cost_facts(CAT.by_id["bellabot-pro"], 2)
    assert (c["buy_total_usd"], c["lease_monthly_total_usd"], c["break_even_months"]) == (19800, 1650, 12)
    assert cost_facts(CAT.by_id["agibot-a2-lite"], 1) is None


def test_verifier_catches_invented_price_unretrieved_product_and_unpublished_price():
    facts = "BellaBot Pro Buy: $9,900 | Lease: $825/month"
    assert any("8500" in p for p in check_reply("BellaBot Pro costs $8,500.", facts, "", [], {"bellabot-pro"}, CAT))
    assert check_reply("Get the AGIBOT G2.", "", "", [], {"bellabot-pro"}, CAT)
    assert any("not published" in p for p in
               check_reply("AGIBOT A2 Lite is $30,000.", "$30,000", "", [], {"agibot-a2-lite"}, CAT))


# ---------- graph: grounding loop ----------

def test_grounded_answer_passes_first_time(tmp_path):
    good = "- **BellaBot Pro**: 4 trays, up to 11 h runtime. Two units: $19,800 to buy or $1,650/month."
    out = run(FakeLLM([restaurant_needs()], [good]), "restaurant, 2 delivery robots", tmp_path)
    assert out["status"] == "grounded" and out["attempts"] == 1
    assert out["recommended_ids"] == ["bellabot-pro"]


def test_hallucination_is_retried_with_feedback(tmp_path):
    llm = FakeLLM([restaurant_needs(quantity=None)],
                  ["**BellaBot Pro** is $7,500.", "**BellaBot Pro**: $9,900 or $825/month."])
    out = run(llm, "restaurant delivery robot price?", tmp_path)
    assert out["status"] == "grounded" and out["attempts"] == 2
    assert "REJECTED BY THE FACT CHECKER" in llm.draft_prompts[1] and "7500" in llm.draft_prompts[1]


def test_persistent_hallucination_falls_back_safely(tmp_path):
    out = run(FakeLLM([restaurant_needs(quantity=None)], ["**BellaBot Pro** is $7,500."] * 3), "price?", tmp_path)
    assert out["status"] == "fallback"
    assert "$9,900" in out["reply"] and "7,500" not in out["reply"]


# ---------- graph: adaptive retrieval ----------

def test_smalltalk_skips_retrieval(tmp_path):
    llm = FakeLLM([Needs(intent="smalltalk")], ["Hi there! What would you like a robot to help with?"])
    out = run(llm, "hi", tmp_path)
    assert out["status"] == "conversational"
    assert "Selection" not in llm.calls
    assert not any(s["step"] in ("retrieve", "select") for s in out["trace"])


def test_smalltalk_that_names_a_product_is_replaced(tmp_path):
    out = run(FakeLLM([Needs(intent="smalltalk")], ["Hi! Try the BellaBot Pro for $9,900."]), "hello", tmp_path)
    assert "BellaBot" not in out["reply"] and "$" not in out["reply"]


def test_off_topic(tmp_path):
    out = run(FakeLLM([Needs(intent="off_topic")], []), "write me a poem", tmp_path)
    assert out["status"] == "off_topic"


def test_named_product_is_looked_up_directly(tmp_path):
    llm = FakeLLM([Needs(intent="product_info")], ["The **AGIBOT A2 Lite** price is not published."])
    out = run(llm, "How much is the AGIBOT A2 Lite?", tmp_path)
    assert "Selection" not in llm.calls
    assert "[agibot-a2-lite]" in llm.draft_prompts[0]
    assert out["status"] == "grounded"


def test_unavailable_request_gets_closest_alternatives(tmp_path):
    needs = Needs(intent="recommend", use_case="cook meals", search_query="cook meals chef")
    sel = Selection(product_ids=["bellabot-pro", "agibot-a2-w"], exact_match=False, reason="closest options")
    reply = ("USTARX doesn't offer a cooking robot, but these could help:\n"
             "- **BellaBot Pro**: delivers dishes to tables, $9,900.\n- **AGIBOT A2-W**: handles items.")
    llm = FakeLLM([needs], [reply], [sel])
    out = run(llm, "I need a robot that cooks food", tmp_path)
    assert "MATCH: closest alternatives" in llm.draft_prompts[0]
    assert out["status"] == "grounded"
    assert out["recommended_ids"] == ["bellabot-pro", "agibot-a2-w"]


def test_follow_up_reuses_previous_products(tmp_path):
    llm = FakeLLM([Needs(intent="follow_up", prefers="lease")], ["Leasing **BellaBot Pro** is $825/month."])
    out = run(llm, "what about leasing it?", tmp_path, lead={"last_products": ["bellabot-pro"]})
    retrieve = next(s for s in out["trace"] if s["step"] == "retrieve")
    assert "earlier in the conversation" in retrieve["how"]
    assert "Selection" not in llm.calls
    assert out["status"] == "grounded"


def test_browse_gets_full_catalog(tmp_path):
    sel = Selection(product_ids=["agibot-d1-pro"], exact_match=True, reason="cheapest")
    llm = FakeLLM([Needs(intent="browse", search_query="cheapest")], ["The **AGIBOT D1 Pro** is $3,699."], [sel])
    out = run(llm, "cheapest robot?", tmp_path)
    assert "FULL CATALOG OVERVIEW" in llm.draft_prompts[0]
    assert out["recommended_ids"] == ["agibot-d1-pro"]


def test_selector_failure_falls_back_to_search(tmp_path):
    out = run(FakeLLM([restaurant_needs()], ["**BellaBot Pro**: $9,900."]), "restaurant robot", tmp_path)
    select = next(s for s in out["trace"] if s["step"] == "select")
    assert "BellaBot Pro" in select["chosen"]


def test_extraction_failure_does_not_crash(tmp_path):
    out = run(FakeLLM([RuntimeError("tool_use_failed")], ["**BellaBot Pro**: $9,900."]),
              "robot to deliver food in my restaurant", tmp_path)
    assert out["status"] == "grounded"


def test_virtual_agents_are_grounded(tmp_path):
    needs = Needs(intent="recommend", search_query="customer support chatbot", wants_virtual_agents=True)
    sel = Selection(product_ids=["customer-service-ai"], exact_match=True, reason="support")
    out = run(FakeLLM([needs], ["- **Customer Service AI**: 24/7 chat and voice support. Price on request."], [sel]),
              "Do you sell chatbots?", tmp_path)
    assert out["status"] == "grounded" and out["recommended_ids"] == ["customer-service-ai"]


# ---------- leads ----------

def test_demo_request_asks_for_email_then_saves_lead(tmp_path):
    good = "**BellaBot Pro** is a good fit."
    out = run(FakeLLM([restaurant_needs(wants_demo_or_quote=True)], [good]), "book a demo", tmp_path)
    assert "email" in out["reply"].lower()
    assert not (tmp_path / "leads.jsonl").exists()
    out2 = run(FakeLLM([Needs(intent="contact_details", email="owner@example.com")], [good]),
               "owner@example.com", tmp_path, lead=out["lead"])
    assert (tmp_path / "leads.jsonl").exists()
    assert out2["lead"]["tier"] == "hot"


def test_names_with_unicode_spaces_and_hyphens_are_recognised(tmp_path):
    assert CAT.mentions_in_order("the **BellaBot\u202fPro** and AGIBOT\u00a0A2\u2011W") == ["bellabot-pro", "agibot-a2-w"]
    out = run(FakeLLM([restaurant_needs(quantity=None)], ["**BellaBot\u202fPro**: $9,900 or $825\u202f/month."]),
              "restaurant robot", tmp_path)
    assert out["recommended_ids"] == ["bellabot-pro"] and "\u202f" not in out["reply"]
