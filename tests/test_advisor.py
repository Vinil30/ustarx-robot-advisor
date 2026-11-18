"""Offline tests: a scripted fake LLM drives the real graph, retrieval, pricing and verifier."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advisor.catalog import Catalog
from advisor.graph import Needs, build_graph
from advisor.pricing import cost_facts
from advisor.verify import check_reply

CAT = Catalog()


def Draft(reply, recommended_ids=None):
    """Drafts are plain text now; recommended products are read from the text by the graph."""
    return reply


class FakeLLM:
    """Returns queued Needs objects (structured) and reply strings (text) in order.
    A Needs entry that is an Exception is raised, to simulate a provider failure."""
    def __init__(self, needs, drafts):
        self.needs, self.drafts = list(needs), list(drafts)
        self.draft_prompts = []

    def structured(self, schema, system, user):
        n = self.needs.pop(0)
        if isinstance(n, Exception):
            raise n
        return n

    def text(self, system, user):
        self.draft_prompts.append(user)
        return self.drafts.pop(0)


def run(llm, msg, tmp_path, lead=None):
    graph = build_graph(llm, CAT, leads_path=tmp_path / "leads.jsonl")
    return graph.invoke({"user_msg": msg, "history": [], "lead": lead or {}})


def restaurant_needs(**kw):
    base = dict(off_topic=False, industry="restaurant", use_case="deliver food to tables",
                search_query="restaurant food delivery", quantity=2)
    base.update(kw)
    return Needs(**base)


def test_retrieval_finds_right_robot():
    assert CAT.search("restaurant food delivery")[0]["id"] == "bellabot-pro"
    assert CAT.search("ros2 university robotics course")[0]["id"] == "agibot-d1-edu-pro"
    assert CAT.search("floor cleaning supermarket")[0]["id"] == "agibot-c5"
    assert CAT.search("hotel lobby receptionist guided tours")[0]["id"] in {"agibot-a2-ultra", "agibot-x2-ultra"}


def test_name_matching_prefers_longest():
    assert CAT.mentioned_names("Try the D1 Edu Pro") == {"agibot-d1-edu-pro"}
    assert CAT.mentioned_names("AGIBOT D1 Edu and BellaBot Pro") == {"agibot-d1-edu", "bellabot-pro"}


def test_pricing_math():
    c = cost_facts(CAT.by_id["bellabot-pro"], 2)
    assert c["buy_total_usd"] == 19800
    assert c["lease_monthly_total_usd"] == 1650
    assert c["break_even_months"] == 12          # 9900 / 825 = 12
    assert cost_facts(CAT.by_id["agibot-a2-lite"], 1) is None   # price not published


def test_verifier_catches_invented_price():
    facts = "BellaBot Pro Buy: $9,900 | Lease: $825/month"
    probs = check_reply("BellaBot Pro costs $8,500.", facts, "", ["bellabot-pro"], {"bellabot-pro"}, CAT)
    assert any("8500" in p for p in probs)


def test_verifier_catches_unretrieved_product_and_unpublished_price():
    probs = check_reply("Get the AGIBOT G2.", "", "", ["agibot-g2"], {"bellabot-pro"}, CAT)
    assert len(probs) == 2
    probs = check_reply("AGIBOT A2 Lite is $30,000.", "$30,000", "", ["agibot-a2-lite"], {"agibot-a2-lite"}, CAT)
    assert any("not published" in p for p in probs)


def test_grounded_answer_passes_first_time(tmp_path):
    good = Draft(reply="BellaBot Pro fits: 4 trays and up to 11 h runtime. Two units cost $19,800 to buy "
                       "or $1,650/month to lease; buying wins after 12 months. Want a demo?",
                 recommended_ids=["bellabot-pro"])
    out = run(FakeLLM([restaurant_needs()], [good]), "I have a restaurant, need 2 robots to deliver food", tmp_path)
    assert out["status"] == "grounded"
    assert out["attempts"] == 1


def test_hallucination_is_retried_with_feedback(tmp_path):
    bad = Draft(reply="BellaBot Pro is $7,500 and runs 20 hours.", recommended_ids=["bellabot-pro"])
    good = Draft(reply="BellaBot Pro: $9,900 to buy or $825/month to lease, up to 11 h runtime.",
                 recommended_ids=["bellabot-pro"])
    llm = FakeLLM([restaurant_needs(quantity=None)], [bad, good])
    out = run(llm, "restaurant delivery robot price?", tmp_path)
    assert out["status"] == "grounded"
    assert out["attempts"] == 2
    assert "REJECTED BY THE FACT CHECKER" in llm.draft_prompts[1]
    assert "7500" in llm.draft_prompts[1]


def test_persistent_hallucination_falls_back_safely(tmp_path):
    bad = Draft(reply="BellaBot Pro is $7,500.", recommended_ids=["bellabot-pro"])
    out = run(FakeLLM([restaurant_needs(quantity=None)], [bad, bad, bad]), "price?", tmp_path)
    assert out["status"] == "fallback"
    assert "$9,900" in out["reply"] and "7,500" not in out["reply"]


def test_off_topic(tmp_path):
    needs = Needs(off_topic=True, search_query="")
    out = run(FakeLLM([needs], []), "write me a poem", tmp_path)
    assert out["status"] == "off_topic"


def test_demo_request_asks_for_email_then_saves_lead(tmp_path):
    good = Draft(reply="BellaBot Pro is a good fit.", recommended_ids=["bellabot-pro"])
    out = run(FakeLLM([restaurant_needs(wants_demo_or_quote=True)], [good]), "book a demo", tmp_path)
    assert "email" in out["reply"].lower()
    assert not (tmp_path / "leads.jsonl").exists()

    out2 = run(FakeLLM([restaurant_needs(email="owner@example.com")], [good]),
               "owner@example.com", tmp_path, lead=out["lead"])
    assert (tmp_path / "leads.jsonl").exists()
    assert out2["lead"]["tier"] == "hot"


def test_recommended_products_are_read_from_reply_text(tmp_path):
    reply = "- **AGIBOT D1 Edu Pro**: ROS2 kit, $10,999.\n- **AGIBOT D1 Edu**: cheaper, $7,199."
    needs = Needs(off_topic=False, search_query="ros2 university course", use_case="ros2 course")
    out = run(FakeLLM([needs], [reply]), "ros2 course robot", tmp_path)
    assert out["recommended_ids"] == ["agibot-d1-edu-pro", "agibot-d1-edu"]
    assert out["status"] == "grounded"


def test_broad_question_uses_full_catalog_overview(tmp_path):
    needs = Needs(off_topic=False, search_query="cheapest", browsing_catalog=True)
    llm = FakeLLM([needs], ["The most affordable is the **AGIBOT D1 Pro** at $3,699."])
    out = run(llm, "cheapest robot?", tmp_path)
    assert "FULL CATALOG OVERVIEW" in llm.draft_prompts[0]
    assert out["status"] == "grounded"
    assert out["recommended_ids"] == ["agibot-d1-pro"]


def test_extraction_failure_does_not_crash(tmp_path):
    llm = FakeLLM([RuntimeError("tool_use_failed")], ["Hi! What would you like a robot to do for your business?"])
    out = run(llm, "hi", tmp_path)
    assert out["status"] == "grounded"


def test_virtual_agents_are_grounded_and_carded(tmp_path):
    needs = Needs(off_topic=False, search_query="customer support chatbot", wants_virtual_agents=True)
    out = run(FakeLLM([needs], ["- **Customer Service AI**: 24/7 chat and voice support. Price on request."]),
              "Do you sell chatbots?", tmp_path)
    assert out["status"] == "grounded"
    assert out["recommended_ids"] == ["customer-service-ai"]
