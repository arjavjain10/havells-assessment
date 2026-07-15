"""
Orchestrator for the Customer Voice Intelligence Agent.

Pipeline:
    reviews.csv
        -> Extraction Agent (per review)      -> extractions.json
        -> Aggregation Agent (deterministic)  -> aggregation.json
        -> QA Agent (retrieval + grounded LLM)-> answers PM questions

Usage:
    python main.py --build                     # run extraction + aggregation, cache to disk
    python main.py --ask "Is the fan getting noisier over time?"
    python main.py --ask "..." --build          # rebuild then ask, in one go
    python main.py                              # interactive Q&A loop (builds if needed)

Set ANTHROPIC_API_KEY in your environment to use real LLM calls for
extraction and answering. Without it, the pipeline runs fully offline
using rule-based fallbacks (still produces grounded, sensible output —
useful for a 1-hour demo with no API access).
"""

import os
import sys
import csv
import json
import argparse

from dotenv import load_dotenv
load_dotenv()  # read .env into os.environ

sys.path.insert(0, os.path.dirname(__file__))
from agents.extraction_agent import extract
from agents.aggregation_agent import aggregate
from agents.qa_agent import answer_question

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
REVIEWS_CSV = os.path.join(DATA_DIR, "reviews.csv")
EXTRACTIONS_JSON = os.path.join(OUT_DIR, "extractions.json")
AGGREGATION_JSON = os.path.join(OUT_DIR, "aggregation.json")


def load_reviews():
    if not os.path.exists(REVIEWS_CSV):
        print("No dataset found — generating synthetic reviews first...")
        os.system(f"cd {os.path.dirname(__file__)} && python3 data/generate_data.py")
    with open(REVIEWS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_pipeline():
    os.makedirs(OUT_DIR, exist_ok=True)
    reviews = load_reviews()
    use_llm = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    mode = "LLM" if use_llm else "OFFLINE (rule-based fallback)"
    print(f"Running extraction on {len(reviews)} reviews in {mode} mode...")

    extractions = []
    for i, r in enumerate(reviews, 1):
        rec = extract(int(r["review_id"]), r["product"], r["date"], r["review_text"], use_llm=use_llm)
        extractions.append(rec)
        if i % 50 == 0:
            print(f"  ...{i}/{len(reviews)} extracted")

    with open(EXTRACTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(extractions, f, indent=2)
    print(f"Saved extractions -> {EXTRACTIONS_JSON}")

    agg_table = aggregate(extractions)
    with open(AGGREGATION_JSON, "w", encoding="utf-8") as f:
        json.dump(agg_table, f, indent=2)
    print(f"Saved aggregation -> {AGGREGATION_JSON}")

    return extractions, agg_table


def load_cached():
    with open(EXTRACTIONS_JSON, encoding="utf-8") as f:
        extractions = json.load(f)
    with open(AGGREGATION_JSON, encoding="utf-8") as f:
        agg_table = json.load(f)
    return extractions, agg_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="(re)run extraction + aggregation")
    parser.add_argument("--ask", type=str, help="ask a single question and exit")
    args = parser.parse_args()

    if args.build or not (os.path.exists(EXTRACTIONS_JSON) and os.path.exists(AGGREGATION_JSON)):
        extractions, agg_table = build_pipeline()
    else:
        extractions, agg_table = load_cached()

    if args.ask:
        result = answer_question(args.ask, agg_table, extractions)
        print("\nQ:", args.ask)
        print("A:", result["answer"])
        print("Grounded:", result["grounded"], "| Cited review_ids:", result["cited_review_ids"])
        return

    print("\nCustomer Voice Intelligence Agent — ask a question (blank line to quit)")
    while True:
        try:
            q = input("\nPM question> ").strip()
        except EOFError:
            break
        if not q:
            break
        result = answer_question(q, agg_table, extractions)
        print("\nA:", result["answer"])
        print("Grounded:", result["grounded"], "| Cited review_ids:", result["cited_review_ids"])


if __name__ == "__main__":
    main()
