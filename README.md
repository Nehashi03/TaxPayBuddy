# TaxPayBuddy

**A multi-agent Retrieval-Augmented Generation (RAG) chatbot for Sri Lankan tax law.**

TaxPayBuddy answers questions about **TIN Registration**, **Individual Income Tax (PIT)**, **Corporate Income Tax (CIT)**, and **Withholding Tax (WHT)** by routing each query to a specialist agent, retrieving grounded evidence from official IRD PDF guides via a vector store, and generating a structured answer — while refusing to answer anything outside the tax domain.

> Built for **DS205.3 – Data Science with Python** as a group final project.

---

## Why this exists

A general-purpose LLM will confidently answer Sri Lankan tax questions from memory — and just as confidently get the rates, reliefs, and thresholds wrong, because tax law changes every fiscal year and isn't reliably represented in any model's training data. TaxPayBuddy addresses that gap by grounding every answer in official IRD source documents and making the retrieval step fully traceable, so an answer can always be checked against the exact PDF chunk it came from.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐      keyword pre-check + LLM classification
│   RouterAgent    │ ───────────────────────────────────────────┐
└─────────────────┘                                              │
    │                                                            ▼
    │                                              ┌───────────────────────────┐
    ▼                                              │  Dispatch Table (registry) │
┌─────────────────────────────────────────────────────────────────────────────┐
│  agent1_tin_registration │ agent2_individual_income_tax │                    │
│  agent3_corporate_income_tax │ agent4_withholding_tax │ general_fallback     │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐     top-k similarity search over
│   ChromaStore    │ ◄── per-agent PDF collections (chunk_size=1000, overlap=200)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Gemini (LLM)    │ ── synthesises a structured answer from retrieved chunks only
└─────────────────┘
```

**Design patterns used:**
- **Strategy / Dispatch Table** — routing is a plain registry lookup (`{label: agent}`), not an `if/elif` ladder, so adding a new tax domain means adding one entry, not touching branching logic.
- **Null Object** — `FallbackAgent` implements the same `IAgent` interface as every specialist agent, so the router never needs to special-case "no match found."
- **Dependency Injection** — every agent receives the same shared `GeminiClient` and `ChromaStore` instances rather than constructing its own.
- **Factory-style construction** — agents are built once in `RouterAgent.__init__` and reused across queries.

---

## Project structure

```
TaxPayBuddy/
├── src/
│   ├── agents/
│   │   ├── router_agent/          
│   │   ├── agent1_tin_registration/
│   │   ├── agent2_individual_income_tax/
│   │   ├── agent3_corporate_income_tax/
│   │   └── agent4_withholding_tax/
│   └── framework/
│       ├── core/                  
│       ├── database/              
│       ├── interfaces/            
│       ├── llm/                   
│       ├── loaders/                
│       └── rag/                   
├── evaluation/
│   ├── ground_truth.json          
│   ├── run_evaluation.py          
│   ├── metrics.py                 
│   ├── llm_judge.py                
│   └── results.csv                
├── tests/                          
├── data/
│   └── raw_pdfs/                  
├── requirements.txt
└── .env                            
```

---

## Getting started

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set your Gemini API key**
```bash
# .env
GEMINI_API_KEY=your_key_here
```

**3. Talk to a single agent**
```bash
python -m src.agents.agent1_tin_registration.main
```

**4. Talk to the full router (recommended)**
```bash
python -m src.agents.router_agent.router_main
```
```
==================================================
TaxPayBuddy - Router Agent
Ask about TIN registration, individual/corporate income tax,
or withholding tax. Type 'exit' to quit.
==================================================

Ask a question: What is a TIN?
[ROUTER AGENT LOG] Routing query to: --> agent1_tin_registration
...
Answer:
A Taxpayer Identification Number (TIN) is a unique identification number...
```

---

## Evaluation

Run the full evaluation harness against the annotated ground-truth set:
```bash
python -m evaluation.run_evaluation
```

This scores every question on:
| Metric | What it measures |
|---|---|
| **Routing accuracy** | Did the query reach the correct specialist agent (or `general_fallback` for out-of-domain questions)? |
| **Precision@K / Recall@K** | Are the retrieved chunks actually relevant to the question? |
| **Cosine similarity** | How close is the generated answer to a human reference answer? |
| **Faithfulness (LLM-judged)** | Is the answer grounded in the retrieved chunks, or hallucinated? |

The output includes a per-agent **routing confusion matrix** (TP/FP/FN/TN, one-vs-rest), which also verifies that out-of-domain questions (e.g. "What's the weather today?") are correctly rejected by the `FallbackAgent` rather than answered by a specialist.

> ⚠️ The Gemini free tier caps requests per day. If you see `429 RESOURCE_EXHAUSTED` errors mid-run, that's the quota, not a routing bug — either wait for the quota to reset, distribute calls across multiple API keys, or reduce `--limit`.

---

## Testing

```bash
pytest tests/ -v
```

53 tests across four files, all running against **mocked** LLM and vector-store clients — no API calls, no quota usage, safe to run continuously in development:
- `test_agents.py` — each specialist agent (and the fallback) returns the correct response shape
- `test_retrieval.py` — retriever respects `top_k` and chunk ordering
- `test_ingestion.py` — PDF chunking produces correctly sized, non-empty chunks with configured overlap
- `test_evaluation.py` — scoring functions in `metrics.py` are correct in isolation

---

## Known limitations

- **Routing keyword matching is substring-based, not word-boundary based** — a keyword can accidentally match inside an unrelated word. Documented as a Future Work item (regex-based word-boundary matching is the planned fix).
- **Keyword-overlap scoring under-rewards correct, paraphrased answers.** The LLM-judged faithfulness score is the more reliable signal for answer quality.
- **Gemini free-tier rate limits** constrain how large an evaluation run can be completed in one sitting.

---

## Team

Built by a 4-person team for DS205.3 (Data Science with Python):
Data Engineer · Infrastructure Lead · Agents Lead · QA/Evaluation Lead.
