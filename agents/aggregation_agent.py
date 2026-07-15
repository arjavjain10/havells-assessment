"""
Aggregation Agent
------------------
Pure deterministic code (NO LLM) — this is intentional. This is the "source
of numeric truth" the QA agent must ground its answers against, so it stays
auditable and reproducible rather than another LLM call that could drift.

Input:  list of extraction-agent outputs (one dict per review)
Output: a nested table:

  {
    "Havells Fan X1": {
      "motor noise": {
        "2026-01": {"mentions": 3, "negative": 3, "positive": 0, "total_reviews": 12},
        "2026-02": {...},
        ...
        "trend": "rising" | "falling" | "stable",
        "trend_slope": 0.083   # change in mention-rate per month
      },
      ...
    },
    ...
  }
"""

import json
from collections import defaultdict


def month_of(date_str):
    return date_str[:7]  # "2026-02-05" -> "2026-02"


def aggregate(extractions):
    # product -> aspect -> month -> counts
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"mentions": 0, "negative": 0, "positive": 0, "neutral": 0})))
    reviews_per_product_month = defaultdict(lambda: defaultdict(int))

    for rec in extractions:
        product = rec["product"]
        m = month_of(rec["date"])
        reviews_per_product_month[product][m] += 1
        for a in rec["aspects"]:
            bucket = table[product][a["aspect"]][m]
            bucket["mentions"] += 1
            bucket[a["sentiment"]] = bucket.get(a["sentiment"], 0) + 1

    result = {}
    for product, aspects in table.items():
        result[product] = {}
        for aspect, months in aspects.items():
            months_sorted = sorted(months.keys())
            series = []
            for m in months_sorted:
                total_reviews = reviews_per_product_month[product][m] or 1
                mention_rate = months[m]["mentions"] / total_reviews
                months[m]["total_reviews"] = total_reviews
                months[m]["mention_rate"] = round(mention_rate, 3)
                series.append(mention_rate)

            slope = _slope(series)
            if slope > 0.03:
                trend = "rising"
            elif slope < -0.03:
                trend = "falling"
            else:
                trend = "stable"

            result[product][aspect] = {
                **{m: months[m] for m in months_sorted},
                "trend": trend,
                "trend_slope": round(slope, 4),
            }
    return result


def _slope(y):
    """Simple least-squares slope, no numpy dependency needed."""
    n = len(y)
    if n < 2:
        return 0.0
    x = list(range(n))
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den = sum((x[i] - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


if __name__ == "__main__":
    demo = [
        {"product": "Havells Fan X1", "date": "2026-01-05",
         "aspects": [{"aspect": "motor noise", "sentiment": "negative", "evidence": "x"}]},
        {"product": "Havells Fan X1", "date": "2026-02-05",
         "aspects": [{"aspect": "motor noise", "sentiment": "negative", "evidence": "x"}]},
        {"product": "Havells Fan X1", "date": "2026-02-06", "aspects": []},
    ]
    print(json.dumps(aggregate(demo), indent=2))
