# Robot Advisor: a fact-checked sales agent for USTARX AI

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Vinil30/ustarx-robot-advisor)

An unofficial demo built for an interview with USTARX AI. It is a website assistant that helps a
business pick the right robot, compares buying vs leasing, and captures qualified leads, and it
**checks its own answer against the catalog before the customer sees it**.

All product data comes from the public pages at https://www.ustarx.ai/products (16 robots, 4 virtual
agents), stored in `data/catalog.json`.

## Why this problem

A sales assistant that invents a price or a spec creates a real liability: a quote the company
can't honour, or a customer who buys the wrong robot. USTARX lists **Sales AI** (lead qualification,
quote generation, CRM) as one of its own products, so this is the same kind of agent, applied to
USTARX's own catalog.

## How it works (LangGraph)

```
                     ┌─ off topic ─► polite redirect
understand ─(intent)─┼─ small talk ─► chat (no retrieval)
                     └─ otherwise ─► retrieve ─► select ─► draft ─► verify ──ok──► qualify_lead ─► reply
                                                             ▲        │
                                                             └─retry──┤ (fact-checker feedback, max 2 retries)
                                                                      └─still failing─► safe_fallback
```

Retrieval is **adaptive**: the agent decides whether and how to look things up instead of searching on every message.

| Step | What it does | LLM or code? |
|---|---|---|
| understand | Classifies intent (small talk, recommend, product info, compare, browse, general question, follow-up, contact details, off topic) and extracts industry, use case, budget, quantity, buy/lease, demo request, contact | LLM (structured output) |
| retrieve | Named products → direct lookup; follow-ups → reuse products from earlier turns (memory); needs → BM25 candidates | Code |
| select | Picks the best 1-3 products from the whole catalog using BM25 hits as hints, and flags whether they are an **exact match or only the closest alternatives** (e.g. "a robot that cooks" → delivery robot + dexterous hand) | LLM (structured output), skipped when products are named |
| pricing | Fleet totals, 12/24-month lease cost, buy-vs-lease break-even month | **Code**, never the LLM |
| draft | Descriptive reply from the selected products' fact sheets: why each fits, key specs, prices, buy-vs-lease advice, one follow-up question | LLM (plain text; products are read back from the text) |
| verify | Every product named must be in the retrieved set; every number must appear in the facts or the user's message; no price for "contact for pricing" products | **Code** (deterministic) |
| retry / fallback | A rejected draft is rewritten with the checker's exact complaints. After 2 failed retries, answer only with catalog text | Code |
| qualify_lead | Scores the lead 0–100 (use case, budget, quantity, demo intent, email), asks for email on demo requests, saves to `data/leads.jsonl` (stand-in for a CRM) | Code |

The right-hand **Agent trace** panel in the UI shows each step live, including rejected drafts.

## Design choices

- **LLMs write; code checks and calculates.** Prices and arithmetic are deterministic, so they can't be hallucinated.
- **Retrieval decides what's allowed.** The verifier knows exactly which products and numbers were in context.
- **Fail safe, not silent.** If the model keeps failing the check, the customer gets a short, correct answer and an offer to talk to a person.
- **Swappable LLM.** The graph depends on a tiny `structured(schema, system, user)` interface; Groq is used in the app, a scripted fake in tests.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste a free Groq key from https://console.groq.com/keys
python app.py               # http://127.0.0.1:5055
```

## Tests

```bash
python -m pytest -q tests
```

19 offline tests (no API key needed) using a scripted fake LLM against the real graph: retrieval
picks the right robot, break-even math, the verifier catches invented prices and unretrieved products,
a hallucinated draft is retried with feedback, persistent hallucination falls back safely, and
demo requests turn into saved leads.

## What I'd add next

- Real CRM write (HubSpot/Salesforce) and calendar booking for demos
- Embedding + BM25 hybrid retrieval once the catalog grows; case studies as extra context
- Evaluation set of real customer questions, tracking fact-check pass rate per model
- Voice mode for the reception robots (same graph behind speech-to-text / text-to-speech)
- Hand-off to a physical robot: the same agent could emit a structured task (e.g. "guide visitor to
  room 3") that a robot's ROS 2 action server executes
