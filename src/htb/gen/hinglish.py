"""How Hinglish speakers say dates, times, amounts and counts.

The sampler picks an exact value (e.g. 19:30) and one of these surface forms
("shaam saade 7 baje"). The teacher model is asked to use that form, so the
dataset reliably covers the hard cases: kal = tomorrow/yesterday, saade/sava/paune,
dedh/dhai, hazaar/lakh, weekday names in Hindi, and so on.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

WEEKDAYS_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
WEEKDAYS_HI = ("somvaar", "mangalvaar", "budhvaar", "guruvaar", "shukravaar", "shanivaar",
               "ravivaar")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")

HINDI_NUMBERS = {1: "ek", 2: "do", 3: "teen", 4: "chaar", 5: "paanch", 6: "chhah", 7: "saat",
                 8: "aath", 9: "nau", 10: "das"}

# Round amounts and the ways people say them.
HINDI_AMOUNTS: dict[int, list[str]] = {
    100: ["sau rupaye", "100"],
    200: ["do sau"],
    250: ["dhai sau"],
    300: ["teen sau"],
    500: ["paanch sau", "500"],
    750: ["saade saat sau"],
    1000: ["ek hazaar", "1k", "hazaar rupaye"],
    1200: ["barah sau"],
    1500: ["dedh hazaar", "pandrah sau"],
    2000: ["do hazaar", "2k"],
    2500: ["dhai hazaar", "pachchees sau"],
    3000: ["teen hazaar"],
    3500: ["saade teen hazaar", "paintees sau"],
    4000: ["chaar hazaar", "4k"],
    5000: ["paanch hazaar", "5k"],
    7500: ["saade saat hazaar"],
    10000: ["das hazaar", "10k"],
    12000: ["barah hazaar", "12k"],
    15000: ["pandrah hazaar"],
    20000: ["bees hazaar", "20k"],
    25000: ["pachchees hazaar"],
    50000: ["pachaas hazaar"],
    100000: ["ek lakh", "1 lakh"],
    150000: ["dedh lakh"],
    200000: ["do lakh"],
    250000: ["dhai lakh"],
}


def indian_grouping(n: int) -> str:
    """1500000 -> '15,00,000'."""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def amount_phrase(rng: random.Random, amount: float) -> str:
    n = int(amount)
    options = [str(n), f"₹{indian_grouping(n)}", f"{indian_grouping(n)} rupaye"]
    if n in HINDI_AMOUNTS and rng.random() < 0.7:
        return rng.choice(HINDI_AMOUNTS[n])  # spoken forms are the point of the dataset
    return rng.choice(options)


def sample_amount(rng: random.Random, lo: int, hi: int) -> int:
    """Mostly round amounts people actually say, sometimes an exact odd value."""
    round_ones = [a for a in HINDI_AMOUNTS if lo <= a <= hi]
    if round_ones and rng.random() < 0.7:
        return rng.choice(round_ones)
    step = 50 if rng.random() < 0.6 else 1
    return max(lo, rng.randrange(lo, hi + 1, step))


def quantity_phrase(rng: random.Random, q: int) -> str:
    if q in HINDI_NUMBERS and rng.random() < 0.6:
        return HINDI_NUMBERS[q]
    return str(q)


def _period(hour: int) -> str:
    if 4 <= hour < 12:
        return "subah"
    if 12 <= hour < 16:
        return "dopahar"
    if 16 <= hour < 20:
        return "shaam"
    return "raat"


def time_phrase(rng: random.Random, hh: int, mm: int) -> str:
    """19:30 -> 'shaam saade 7 baje' / '7:30 pm' etc. Always unambiguous."""
    h12 = hh % 12 or 12
    nxt = (hh + 1) % 12 or 12
    if mm == 0:
        spoken = f"{h12} baje"
    elif mm == 30:
        spoken = {1: "dedh baje", 2: "dhai baje"}.get(h12, f"saade {h12} baje")
    elif mm == 15:
        spoken = f"sava {h12} baje"
    elif mm == 45:
        spoken = f"paune {nxt} baje"
    else:
        spoken = f"{h12}:{mm:02d} baje"
    hindi = f"{_period(hh)} {spoken}"
    suffix = "am" if hh < 12 else "pm"
    english = f"{h12}:{mm:02d} {suffix}" if mm else f"{h12} {suffix}"
    return rng.choice([hindi, hindi, hindi, english])


def _absolute(rng: random.Random, target: date, today: date) -> str:
    if target.month == today.month and target.year == today.year:
        return rng.choice([f"{target.day} tarikh ko", f"{target.day} {MONTHS[target.month - 1][:3]}"])
    return rng.choice([f"{target.day} {MONTHS[target.month - 1]}",
                       f"{target.day} {MONTHS[target.month - 1][:3]} ko"])


def future_date_phrase(rng: random.Random, target: date, today: date) -> str:
    """How a user refers to a day on or after today."""
    delta = (target - today).days
    if delta < 0:
        raise ValueError("target is in the past")
    wd = target.weekday()
    options: list[str] = []
    if delta == 0:
        options += ["aaj", "aaj", "today"]
    elif delta == 1:
        options += ["kal"] * 5 + ["tomorrow"]
    elif delta == 2:
        options += ["parso"] * 5 + ["day after tomorrow"]
    if 1 <= delta <= 6:
        options += [f"is {WEEKDAYS_EN[wd]}", f"{WEEKDAYS_HI[wd]} ko", f"coming {WEEKDAYS_EN[wd]}"]
    next_monday = today + timedelta(days=7 - today.weekday())
    if next_monday <= target <= next_monday + timedelta(days=6):
        options += [f"agle hafte {WEEKDAYS_EN[wd]}", f"next week {WEEKDAYS_EN[wd]}"]
    options.append(_absolute(rng, target, today))
    return rng.choice(options)


def past_range_phrase(rng: random.Random, today: date) -> tuple[date, date, str]:
    """A past date range for transaction queries: (start, end, phrase)."""
    kind = rng.choice(["yesterday", "today", "last_n", "last_n", "this_month", "last_month",
                       "named_month"])
    if kind == "yesterday":
        d = today - timedelta(days=1)
        return d, d, rng.choice(["kal ke", "kal ke", "yesterday ke"])
    if kind == "today":
        return today, today, rng.choice(["aaj ke", "today ke"])
    if kind == "last_n":
        n = rng.choice([3, 5, 7, 10, 15, 30])
        return today - timedelta(days=n), today, rng.choice(
            [f"pichle {n} din ke", f"last {n} days ke"])
    first_this = today.replace(day=1)
    if kind == "this_month":
        return first_this, today, rng.choice(["is mahine ke", "this month ke"])
    last_prev = first_this - timedelta(days=1)
    first_prev = last_prev.replace(day=1)
    if kind == "last_month":
        return first_prev, last_prev, rng.choice(["pichle mahine ke", "last month ke"])
    # named month: one of the previous 2-4 months, fully elapsed
    end = first_prev - timedelta(days=1)
    for _ in range(rng.randint(0, 2)):
        end = end.replace(day=1) - timedelta(days=1)
    start = end.replace(day=1)
    return start, end, f"{MONTHS[start.month - 1]} ke"
