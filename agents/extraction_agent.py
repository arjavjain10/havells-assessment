"""
Extraction Agent
----------------
Turns ONE raw review into structured facts:

    {
      "review_id": 51,
      "product": "Havells Fan X1",
      "date": "2026-02-05",
      "aspects": [
        {"aspect": "motor noise", "sentiment": "negative", "evidence": "..."}
      ]
    }

Two modes:
  - LLM mode  (ANTHROPIC_API_KEY set): calls Claude, forces strict JSON output.
  - Offline fallback mode (no key): keyword/lexicon based extractor.
    Runs with zero setup so the whole pipeline is demoable without an API key.

Every extracted aspect carries the exact "evidence" substring it came from,
in the review text. This is the atomic unit that the QA agent later cites —
nothing downstream is allowed to state a fact that isn't traceable back to
one of these evidence strings.
"""

import os
import json
import re

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

ASPECT_KEYWORDS = {
    "motor noise": ["motor", "noise", "noisy", "humming", "rattling", "sound"],
    "thermostat": ["thermostat", "temperature control", "cut off", "cut-off"],
    "airflow": ["airflow", "cools", "cooling", "cool the"],
    "build quality": ["sturdy", "built", "build quality", "wobble", "plastic"],
    "heating speed": ["heats water", "hot water", "heating"],
    "installation": ["install", "plumber", "fitting"],
    "blade durability": ["blade", "blunt", "chipped"],
    "motor power": ["powerful motor", "grinds", "spices"],
    "jar quality": ["jar", "leak", "seal", "lid"],
    "price": ["price", "afford", "value", "budget", "worth the money"],
}

NEGATIVE_MARKERS = ["doesn't", "not", "unreliable", "loud", "worse", "blunt",
                     "chipped", "stopped working", "inconsistent", "wobble",
                     "leak", "rattling", "noisy"]


def _offline_extract(review_id, product, date_str, text):
    text_l = text.lower()
    aspects = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_l:
                # find the sentence containing the keyword as evidence
                sentence = next(
                    (s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if kw in s.lower()),
                    text,
                )
                sentiment = "negative" if any(m in sentence.lower() for m in NEGATIVE_MARKERS) else "positive"
                aspects.append({"aspect": aspect, "sentiment": sentiment, "evidence": sentence})
                break  # one hit per aspect is enough
    return {
        "review_id": review_id,
        "product": product,
        "date": date_str,
        "aspects": aspects,
    }


def _llm_extract(review_id, product, date_str, text):
    from google import genai

    client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY) from env
    prompt = f"""You extract structured facts from a single product review. Return ONLY valid JSON, no prose, no markdown fences.

Schema:
{{"aspects": [{{"aspect": "<short noun phrase>", "sentiment": "positive"|"negative"|"neutral", "evidence": "<exact substring copied from the review that supports this>"}}]}}

Rules:
- "evidence" MUST be an exact substring of the review text below (copy verbatim, do not paraphrase).
- Only extract aspects that are actually discussed. If nothing specific is discussed, return an empty list.
- Do not invent aspects or sentiment not supported by the text.

Review: "{text}"
"""
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json", "max_output_tokens": 500},
    )
    raw = (resp.text or "").strip()
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
    parsed = json.loads(raw)

    # Grounding check: drop any aspect whose "evidence" isn't actually in the text
    clean_aspects = [a for a in parsed.get("aspects", []) if a.get("evidence", "") in text]

    return {
        "review_id": review_id,
        "product": product,
        "date": date_str,
        "aspects": clean_aspects,
    }


def extract(review_id, product, date_str, text, use_llm=None):
    if use_llm is None:
        use_llm = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if use_llm:
        try:
            return _llm_extract(review_id, product, date_str, text)
        except Exception as e:
            print(f"[extraction_agent] LLM call failed ({e}), falling back to offline mode for review {review_id}")
            return _offline_extract(review_id, product, date_str, text)
    return _offline_extract(review_id, product, date_str, text)


if __name__ == "__main__":
    demo = extract(1, "Havells Fan X1", "2026-02-05",
                    "Motor makes a loud humming noise. Great airflow even on low speed.")
    print(json.dumps(demo, indent=2))
