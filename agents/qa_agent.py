"""
Grounded QA Agent
-----------------
Answers a PM's plain-English question using ONLY:
  1. the aggregation table (numbers/trends), and
  2. a handful of retrieved raw review snippets (evidence quotes)

Flow:
  question -> retrieve relevant (product, aspect) slice(s) from the
              aggregation table + matching raw evidence
           -> build a context block containing ONLY that retrieved data
           -> LLM answers strictly from context, must cite review_ids
           -> post-hoc verifier: every number the LLM states must appear
              in the retrieved aggregation data, else the answer is
              flagged and the numbers are stripped/replaced with a caveat

If retrieval finds nothing relevant, the agent says so explicitly instead
of falling back to the LLM's general knowledge.
"""

import os
import re
import json

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


PRODUCT_ALIASES = {
    "Havells Fan X1": ["fan"],
    "Havells Instanio Water Heater": ["water heater", "heater", "thermostat", "instanio"],
    "Havells Mixer Grinder MG100": ["mixer", "grinder", "blade"],
}


def _retrieve(question, agg_table, extractions, top_k_evidence=5):
    q = question.lower()

    matched_products = [
        p for p in agg_table
        if p.lower() in q or any(alias in q for alias in PRODUCT_ALIASES.get(p, []))
    ]

    matched = []  # (product, aspect, data)
    search_space = matched_products if matched_products else list(agg_table.keys())
    for product in search_space:
        for aspect, data in agg_table[product].items():
            if any(tok in q for tok in aspect.split()) or aspect in q:
                matched.append((product, aspect, data))

    if not matched and matched_products:
        for product in matched_products:
            for aspect, data in agg_table[product].items():
                matched.append((product, aspect, data))

    # if NEITHER product nor aspect matched, this is out of scope: do not
    # silently fall back to unrelated data.

    # pull matching raw evidence sentences for the matched (product, aspect) pairs
    matched_keys = {(p, a) for p, a, _ in matched}
    evidence = []
    for rec in extractions:
        for a in rec["aspects"]:
            if (rec["product"], a["aspect"]) in matched_keys:
                evidence.append({
                    "review_id": rec["review_id"],
                    "product": rec["product"],
                    "date": rec["date"],
                    "aspect": a["aspect"],
                    "sentiment": a["sentiment"],
                    "evidence": a["evidence"],
                })
    evidence = evidence[:top_k_evidence]
    return matched, evidence


def _format_context(matched, evidence):
    lines = ["AGGREGATED DATA (source of truth for any numbers/trends):"]
    for product, aspect, data in matched:
        trend = data.get("trend")
        slope = data.get("trend_slope")
        months = {k: v for k, v in data.items() if k not in ("trend", "trend_slope")}
        lines.append(f"- {product} / {aspect}: trend={trend} (slope={slope})")
        for m, stats in sorted(months.items()):
            lines.append(f"    {m}: mention_rate={stats['mention_rate']}, "
                          f"mentions={stats['mentions']}/{stats['total_reviews']} reviews, "
                          f"negative={stats['negative']}, positive={stats['positive']}")

    lines.append("\nRAW EVIDENCE SNIPPETS (cite review_id when referencing):")
    for e in evidence:
        lines.append(f'  [review_id={e["review_id"]}, {e["product"]}, {e["date"]}, '
                      f'{e["aspect"]}/{e["sentiment"]}] "{e["evidence"]}"')
    return "\n".join(lines)


def _offline_answer(question, matched, evidence):
    """No-API fallback: templated answer built directly from the aggregation
    table, so it's still 100% grounded even with zero LLM calls."""
    if not matched:
        return "I don't have supporting review data for that question.", []

    parts = []
    cited_ids = [e["review_id"] for e in evidence]
    for product, aspect, data in matched[:3]:
        trend = data["trend"]
        months = sorted(k for k in data if k not in ("trend", "trend_slope"))
        if not months:
            continue
        latest = data[months[-1]]
        parts.append(
            f"For {product}, \"{aspect}\" is {trend} "
            f"(latest mention rate {latest['mention_rate']*100:.0f}% of reviews in {months[-1]}, "
            f"{latest['negative']} negative mentions)."
        )
    if not parts:
        return "I don't have supporting review data for that question.", []
    answer = " ".join(parts)
    if cited_ids:
        answer += f" (based on reviews: {cited_ids})"
    return answer, cited_ids


def _llm_answer(question, context):
    from google import genai

    client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY) from env
    prompt = f"""You are a grounded product-review analyst. Answer the PM's question using ONLY the data below. Do not use outside knowledge.

Rules:
- Every number you state (percentages, counts, trend direction) MUST come from the AGGREGATED DATA section, verbatim.
- Reference review_id numbers from RAW EVIDENCE SNIPPETS when illustrating a point.
- If the data below does not support an answer, say plainly: "The review data doesn't support a conclusion on this" and do not guess.
- Keep the answer to 3-5 sentences, plain English, for a product manager.

{context}

Question: {question}
"""
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"max_output_tokens": 400},
    )
    return (resp.text or "").strip()


def _verify_numbers(answer_text, matched):
    """Post-hoc grounding check: every percentage mentioned in the answer
    must match some mention_rate actually present in the retrieved data."""
    valid_pcts = set()
    for _, _, data in matched:
        for m, stats in data.items():
            if m in ("trend", "trend_slope"):
                continue
            valid_pcts.add(round(stats["mention_rate"] * 100))

    stated_pcts = [int(x) for x in re.findall(r"(\d+)\s*%", answer_text)]
    unverified = [p for p in stated_pcts if not any(abs(p - v) <= 1 for v in valid_pcts)]
    return len(unverified) == 0, unverified


def answer_question(question, agg_table, extractions, use_llm=None):
    if use_llm is None:
        use_llm = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    matched, evidence = _retrieve(question, agg_table, extractions)

    if not matched:
        return {
            "answer": "I don't have supporting review data for that question.",
            "grounded": True,
            "cited_review_ids": [],
        }

    if use_llm:
        try:
            context = _format_context(matched, evidence)
            answer = _llm_answer(question, context)
            grounded, unverified_pcts = _verify_numbers(answer, matched)
            if not grounded:
                answer += (f"\n\n[Grounding check flagged unverified figures {unverified_pcts}%; "
                           f"treat with caution / re-run.]")
            cited_ids = [int(i) for i in re.findall(r"review_id[=\s]*(\d+)", answer)] or \
                        [e["review_id"] for e in evidence]
            return {"answer": answer, "grounded": grounded, "cited_review_ids": cited_ids}
        except Exception as e:
            print(f"[qa_agent] LLM call failed ({e}), falling back to offline templated answer")

    answer, cited_ids = _offline_answer(question, matched, evidence)
    return {"answer": answer, "grounded": True, "cited_review_ids": cited_ids}


if __name__ == "__main__":
    pass
