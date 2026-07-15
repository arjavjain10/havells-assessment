# Customer Voice Intelligence Agent — Havells

## What this is

Havells sells a huge range of appliances, and the reviews pile up faster
than a product or marketing team can read them. The question they actually
care about is simple: **what are people unhappy about, on which product,
and is it getting better or worse over time?**

This project is an agentic system that reads raw reviews, works out the
recurring themes and how sentiment around them moves month to month, and
then answers a PM's question in plain English — with every answer traceable
back to specific reviews. If the data doesn't support a claim, the system
says so instead of inventing one. That grounding requirement shaped almost
every design decision below.

---

## System design — how I split the problem

I deliberately split this into **three agents with one clear job each**,
rather than one large prompt trying to do extraction, aggregation, and
answering all at once. The reasoning:

- A single mega-prompt can't be independently tested or trusted. If the
  final answer is wrong, you can't tell whether the model misread the
  reviews, miscounted the trend, or just answered badly — there's no seam
  to debug at.
- Splitting the pipeline means each stage has a narrow, checkable
  contract: extraction only has to be faithful to one review at a time;
  aggregation only has to do arithmetic correctly; the QA agent only has
  to answer from what it's given. Each of those is much easier to get
  right — and to verify — than "read all reviews and answer anything."
- It also means **only one stage (extraction) ever touches unstructured
  text with an LLM at scale**. Aggregation is deliberately plain code, not
  a model call, so the numbers a PM sees are reproducible and not subject
  to model variance.

### Architecture

```
reviews.csv
     │
     ▼
┌─────────────────────┐   per review → structured JSON
│  Extraction Agent    │   {aspect, sentiment, evidence}
│  (Gemini or          │   "evidence" = an exact substring copied from
│   rule-based)         │   the review — nothing else is allowed through
└─────────┬────────────┘
          ▼
┌─────────────────────┐   deterministic, NO LLM call
│  Aggregation Agent   │   groups by product × aspect × month,
│  (pure Python)        │   computes mention-rate trend via regression
└─────────┬────────────┘   → rising / falling / stable
          ▼
┌─────────────────────┐   question → retrieve the matching (product,
│  Grounded QA Agent   │   aspect) slice + a few raw review snippets
│  (retrieval + Gemini)│   → LLM answers strictly from that context
└─────────────────────┘   → post-hoc check: every number in the answer
                             must match the aggregation table, or it's
                             flagged
```

### Why each stage is built the way it is

**Extraction Agent.** This is the only place raw text gets interpreted.
Every aspect it produces must carry an `evidence` field that is an *exact
substring* of the source review. If the model can't point to where in the
text it got something from, that fact is dropped before it's allowed
downstream. This is the single biggest lever against hallucination — it's
enforced structurally, not just by asking the model nicely.

**Aggregation Agent.** No LLM call here at all — plain grouping and a
least-squares slope over monthly mention-rates to classify a theme as
rising, falling, or stable. This was a deliberate choice: numbers and
trend directions need to be the same every time you re-run them on the
same data. An LLM re-summarizing counts introduces variance exactly where
you can't afford it.

**Grounded QA Agent.** When a PM asks a question, this agent doesn't let
the model answer from general knowledge. It:
1. Retrieves only the relevant slice of the aggregation table (matching
   product/aspect) plus a handful of raw review snippets as evidence.
2. Hands *only that retrieved context* to Gemini, with an explicit
   instruction to answer strictly from it and say plainly when the data
   doesn't support a conclusion.
3. Runs a post-hoc check on the answer: every percentage or count the
   model states must match something actually present in the retrieved
   aggregation data. If it doesn't, the answer is flagged rather than
   trusted silently.
4. If retrieval finds nothing relevant to the question at all (e.g. a
   product or issue that isn't in the data), the agent says "I don't have
   supporting data" instead of guessing — I specifically tested this with
   an out-of-scope question to make sure it doesn't silently fall back to
   unrelated reviews.

---

## Workflow — what actually happens end to end

1. **Data generation** (`data/generate_data.py`) — produces 240 synthetic
   reviews across three product lines, with three patterns seeded on
   purpose so the pipeline's output can be checked against a known answer
   rather than eyeballed. See the Dataset section below.

2. **Build** (`python main.py --build`) —
   - Loads `data/reviews.csv`.
   - Runs the Extraction Agent over every review (Gemini if
     `GEMINI_API_KEY` is set, otherwise a keyword-based offline fallback
     — same interface either way).
   - Caches the structured facts to `outputs/extractions.json`.
   - Runs the Aggregation Agent over those facts, caches the result to
     `outputs/aggregation.json`.

3. **Ask** (`python main.py --ask "..."`) — the QA agent retrieves the
   relevant slice of the cached aggregation table, pulls supporting
   review snippets, and returns a grounded answer with cited `review_id`s
   and a `grounded: True/False` flag.

4. **Evaluate** (`python eval.py`) — runs three checks against the
   pipeline's own cached output, described below.

---

## Dataset

I built a synthetic dataset (`data/reviews.csv`, generated by
`data/generate_data.py`) rather than using a found dataset, because the
brief explicitly allows this ("pull from any open-source dataset, or
build a realistic one yourself") — and because building it myself let me
**seed known patterns and then prove the pipeline recovers them**, which
is what makes the evaluation section meaningful rather than a plausibility
check.

240 reviews, three Havells product lines, Jan–Jun 2026:

| Product | Seeded pattern |
|---|---|
| Havells Fan X1 | "motor noise" complaints rise steadily month over month |
| Havells Instanio Water Heater | "thermostat" complaints start high, then drop sharply after March (simulating a fix shipping) |
| Havells Mixer Grinder MG100 | "blade durability" stays flat — a control theme, deliberately not trending either way |

Swapping in a real-world dataset later is a one-line change: any CSV with
columns `review_id, product, date, month, rating, review_text` drops
straight into `data/reviews.csv` and the pipeline runs unmodified.

---

## Evaluation — how I measured whether it's actually any good

`eval.py` checks the three things that matter most for this system,
against the seeded ground truth in `data/ground_truth.json`:

1. **Extraction accuracy** — precision/recall of extracted
   (aspect, sentiment) pairs against what was actually seeded into each
   review.
2. **Trend detection accuracy** — does the Aggregation Agent correctly
   recover all three seeded patterns (rising / falling / stable)?
3. **QA groundedness** — for a fixed set of test questions, is every
   number in the answer traceable to the aggregation table, and is a
   deliberately out-of-scope question correctly answered with "no
   supporting data" instead of a guess?

Last run:
```
Extraction   — precision=0.81  recall=0.90  f1=0.85
Trend check  — 3/3 seeded patterns correctly recovered
QA grounded  — 4/4 test questions passed, including correct rejection
               of an out-of-scope question
```

In a production setting this would extend to a rolling human-labeled
sample for extraction QA, and a periodically-regenerated set of "trap"
questions to catch grounding regressions before they reach a PM.

---

## What would have to change to cover the whole catalogue

Right now this runs comfortably on one product line's worth of reviews in
memory. Scaling to the full Havells catalogue would mean:

- **Extraction** is embarrassingly parallel — one review at a time, no
  shared state — so it moves behind a queue (SQS/Kafka) with workers
  scaling horizontally. This is also the dominant cost driver, so batching
  LLM calls matters here more than anywhere else.
- **Aggregation** becomes a scheduled batch job per product line instead
  of an in-memory recompute, writing into a proper time-series store
  (e.g. a Postgres table keyed by product × aspect × month) instead of a
  JSON file.
- **QA retrieval** currently does keyword matching over a small in-memory
  table, which is fine for one product but won't generalize across a
  catalogue. At scale this becomes a real vector index over aspect
  descriptions and evidence snippets — while keeping the exact same
  contract: only answer from retrieved context, verify every number
  afterward.
- **Aspect taxonomy drift** — with three products, a fixed aspect list is
  fine. Across a full catalogue, near-duplicate aspects ("motor noise" vs
  "loud motor" vs "buzzing sound") need a normalization/clustering step
  between extraction and aggregation, or trend numbers start splitting
  across what should be one theme.

---

## How to run it

```bash
pip install -r requirements.txt

# .env should contain:
#   GEMINI_API_KEY=your_key
#   GEMINI_MODEL=gemini-2.5-flash

python main.py --build --ask "Is the fan getting noisier over time?"
python main.py --ask "Which product has the most complaints?"
python main.py --ask "Has sentiment improved over time for the water heater?"
python eval.py
```

Without a Gemini key set, the whole pipeline still runs end to end on the
rule-based offline fallback — useful for a quick check with no API access,
and a deliberate reliability choice rather than a missing feature: the
system degrades to something simpler rather than failing outright.

---

## File structure

```
havells_review_agent/
├── README.md
├── main.py                      # orchestrator + CLI
├── eval.py                      # evaluation harness
├── requirements.txt
├── .env.example                 # template — actual .env is gitignored
├── data/
│   ├── generate_data.py         # synthetic dataset generator (seeded patterns)
│   ├── reviews.csv              # the dataset the pipeline reads
│   └── ground_truth.json        # seeded labels, used only by eval.py
├── agents/
│   ├── extraction_agent.py      # review text -> structured, evidence-backed aspects
│   ├── aggregation_agent.py     # structured facts -> trends (pure code, no LLM)
│   └── qa_agent.py              # retrieval + grounded Q&A + numeric verification
└── outputs/
    ├── extractions.json         # cached extraction output (gitignored, regenerated)
    └── aggregation.json         # cached aggregation table (gitignored, regenerated)
```