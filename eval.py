"""
Evaluation harness.

Three things get measured, matching what the brief asks for
("how you'd actually measure whether it's any good"):

1. Extraction accuracy — compare extracted aspects/sentiment against the
   seeded ground truth from data/generate_data.py (precision/recall).
2. Trend detection accuracy — did the aggregation agent correctly recover
   the three seeded patterns (fan noise rising / thermostat falling /
   blades stable)?
3. QA groundedness — run a fixed set of test questions and check that
   every numeric claim traces back to the aggregation table, and that
   out-of-scope questions are correctly answered with "no data" instead
   of a hallucinated guess.

Run: python eval.py   (after python main.py --build)
"""

import json
import os
from agents.qa_agent import answer_question

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def eval_extraction():
    with open("data/ground_truth.json") as f:
        gt = {r["review_id"]: set((a[0], a[1]) for a in r["aspects"]) for r in json.load(f)}
    with open(os.path.join(OUT_DIR, "extractions.json")) as f:
        pred = {r["review_id"]: set((a["aspect"], a["sentiment"]) for a in r["aspects"]) for r in json.load(f)}

    tp = fp = fn = 0
    for rid, gold in gt.items():
        got = pred.get(rid, set())
        tp += len(gold & got)
        fp += len(got - gold)
        fn += len(gold - got)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"Extraction  — precision={precision:.2f}  recall={recall:.2f}  f1={f1:.2f}")


def eval_trends():
    with open(os.path.join(OUT_DIR, "aggregation.json")) as f:
        agg = json.load(f)
    expected = {
        ("Havells Fan X1", "motor noise"): "rising",
        ("Havells Instanio Water Heater", "thermostat"): "falling",
        ("Havells Mixer Grinder MG100", "blade durability"): "stable",
    }
    correct = 0
    for (product, aspect), exp_trend in expected.items():
        got_trend = agg.get(product, {}).get(aspect, {}).get("trend", "MISSING")
        ok = got_trend == exp_trend
        correct += ok
        print(f"Trend check — {product}/{aspect}: expected={exp_trend} got={got_trend} {'OK' if ok else 'MISMATCH'}")
    print(f"Trend detection accuracy: {correct}/{len(expected)}")


def eval_qa_groundedness():
    with open(os.path.join(OUT_DIR, "extractions.json")) as f:
        extractions = json.load(f)
    with open(os.path.join(OUT_DIR, "aggregation.json")) as f:
        agg = json.load(f)

    test_questions = [
        "Is the fan getting noisier over time?",
        "How has the water heater thermostat issue trended?",
        "What's going on with the mixer grinder blades?",
        "Is the toaster catching fire?",  # out of scope: no such product/aspect
    ]
    grounded_count = 0
    for q in test_questions:
        result = answer_question(q, agg, extractions)
        print(f"\nQ: {q}\nA: {result['answer']}\nGrounded: {result['grounded']}")
        if result["grounded"]:
            grounded_count += 1
    print(f"\nQA groundedness: {grounded_count}/{len(test_questions)} answers passed the grounding check")


if __name__ == "__main__":
    print("=" * 60)
    eval_extraction()
    print("=" * 60)
    eval_trends()
    print("=" * 60)
    eval_qa_groundedness()
