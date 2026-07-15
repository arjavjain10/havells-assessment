"""
Generates a synthetic Havells product-review dataset with deliberately
seeded patterns, so the pipeline's output can be checked against known
ground truth.

Seeded patterns (used later for evaluation):
  1. Fans          -> "motor noise" complaints RISE steadily Jan -> Jun 2026
  2. Water Heater   -> "thermostat" complaints HIGH in Jan, DROP after a
                       fix ships in March (step-down trend)
  3. Mixer Grinder  -> "blade durability" is a STABLE control theme
                       (no real trend either way)

Run:  python data/generate_data.py
Output: data/reviews.csv
"""

import csv
import random
from datetime import date, timedelta

random.seed(42)

PRODUCTS = {
    "Havells Fan X1": {
        "aspects": ["motor noise", "airflow", "build quality", "price"],
        "good_phrases": {
            "airflow": ["great airflow even on low speed", "cools the whole room fast"],
            "build quality": ["feels sturdy and well built", "solid plastic, no wobble"],
            "price": ["good value for the price", "reasonably priced for the features"],
        },
        "bad_phrases": {
            "motor noise": ["motor makes a loud humming noise", "very noisy after a few weeks",
                             "rattling sound from the motor at high speed", "noise got worse over time"],
        },
    },
    "Havells Instanio Water Heater": {
        "aspects": ["thermostat", "heating speed", "installation", "price"],
        "good_phrases": {
            "heating speed": ["heats water quickly", "hot water within minutes"],
            "installation": ["easy to install", "plumber had no issues fitting it"],
            "price": ["decent price for the capacity", "affordable compared to competitors"],
        },
        "bad_phrases": {
            "thermostat": ["thermostat doesn't cut off properly", "temperature control is unreliable",
                            "thermostat stopped working within a month", "inconsistent heating due to thermostat"],
        },
    },
    "Havells Mixer Grinder MG100": {
        "aspects": ["blade durability", "motor power", "jar quality", "price"],
        "good_phrases": {
            "motor power": ["powerful motor, grinds fast", "handles tough spices easily"],
            "jar quality": ["jars are leak proof", "good quality jars, no cracks"],
            "price": ["worth the money", "budget friendly option"],
        },
        "bad_phrases": {
            "blade durability": ["blade turned blunt quickly", "blades are not very durable",
                                  "blade edge chipped after a month"],
            "jar quality": ["jar lid doesn't seal well"],
        },
    },
}

START = date(2026, 1, 1)
MONTHS = 6  # Jan - Jun 2026


def month_bucket(d):
    return f"{d.year}-{d.month:02d}"


def motor_noise_prob(month_idx):
    # rises from 8% to ~55% of fan reviews mentioning it, month over month
    return 0.08 + month_idx * 0.09


def thermostat_prob(month_idx):
    # high before the fix (Jan/Feb), drops sharply from March onward
    return 0.55 if month_idx < 2 else 0.12


def random_date_in_month(month_idx):
    start = date(START.year, START.month, 1)
    for _ in range(month_idx):
        start = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    day = random.randint(1, 27)
    return start.replace(day=day)


def make_review(review_id, product):
    info = PRODUCTS[product]
    month_idx = random.randint(0, MONTHS - 1)
    d = random_date_in_month(month_idx)

    sentences = []
    aspects_hit = []

    # decide the "signature" defect for this product with seeded probability
    if product == "Havells Fan X1":
        if random.random() < motor_noise_prob(month_idx):
            sentences.append(random.choice(info["bad_phrases"]["motor noise"]))
            aspects_hit.append(("motor noise", "negative"))
    elif product == "Havells Instanio Water Heater":
        if random.random() < thermostat_prob(month_idx):
            sentences.append(random.choice(info["bad_phrases"]["thermostat"]))
            aspects_hit.append(("thermostat", "negative"))
    elif product == "Havells Mixer Grinder MG100":
        if random.random() < 0.15:  # stable, low, no trend
            sentences.append(random.choice(info["bad_phrases"]["blade durability"]))
            aspects_hit.append(("blade durability", "negative"))

    # sprinkle 1-2 generic positive aspects for realism
    good_aspects = list(info["good_phrases"].keys())
    for a in random.sample(good_aspects, k=min(2, len(good_aspects))):
        if random.random() < 0.6:
            sentences.append(random.choice(info["good_phrases"][a]))
            aspects_hit.append((a, "positive"))

    if not sentences:
        sentences.append("Overall an okay product, does the job.")

    random.shuffle(sentences)
    text = ". ".join(s.capitalize() for s in sentences) + "."

    neg_count = sum(1 for _, s in aspects_hit if s == "negative")
    rating = max(1, min(5, 5 - neg_count * 2 + random.choice([-1, 0, 0, 1])))

    return {
        "review_id": review_id,
        "product": product,
        "date": d.isoformat(),
        "month": month_bucket(d),
        "rating": rating,
        "review_text": text,
        # ground truth, kept ONLY for evaluation, not fed to the pipeline
        "_gt_aspects": aspects_hit,
    }


def generate(n_per_product=80):
    rows = []
    rid = 1
    for product in PRODUCTS:
        for _ in range(n_per_product):
            rows.append(make_review(rid, product))
            rid += 1
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = generate()
    out_path = "data/reviews.csv"
    fieldnames = ["review_id", "product", "date", "month", "rating", "review_text"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    # also dump ground truth separately for the eval script
    import json
    with open("data/ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"review_id": r["review_id"], "aspects": r["_gt_aspects"]} for r in rows],
            f, indent=2,
        )

    print(f"Wrote {len(rows)} reviews to {out_path}")
    print("Wrote ground-truth aspect labels to data/ground_truth.json")
