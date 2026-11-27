"""Label-first scenario sampling.

Most synthetic tool-calling datasets ask an LLM to write a request *and* its label,
so label noise is baked in. Here the label comes first: we sample a valid call in
code, then ask the teacher only to phrase it. Labels are correct by construction;
what can go wrong is the phrasing, and the round-trip verifier catches that.

Each per-tool sampler returns a ``Draft``: the arguments plus, for every argument,
a phrasing hint ("say the date as 'parso'"). Clarification scenarios are built by
removing one required argument and its hint.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from .. import lexicon as lx
from ..schema import Context, Expected, ToolCall
from ..tools import TOOLS, canonicalize
from .hinglish import (
    WEEKDAYS_EN,
    amount_phrase,
    future_date_phrase,
    past_range_phrase,
    quantity_phrase,
    sample_amount,
    time_phrase,
)


@dataclass
class Draft:
    tool: str
    args: dict[str, Any]
    hints: dict[str, str] = field(default_factory=dict)  # argument -> phrasing hint
    extra: list[str] = field(default_factory=list)  # hints not tied to one argument

    def all_hints(self) -> list[str]:
        return list(self.hints.values()) + self.extra


def _city_hint(rng: random.Random, role: str, city: str) -> str:
    if city in lx.CITY_COLLOQUIAL and rng.random() < 0.35:
        return f'{role}: {city} (call it "{rng.choice(lx.CITY_COLLOQUIAL[city])}")'
    return f"{role}: {city}"


def _day_offset(rng: random.Random, lo: int, hi: int) -> int:
    """Days from today, weighted toward how people actually refer to days.

    A uniform spread over a month makes "kal" and "parso" rare, yet they are the
    commonest and the most ambiguous forms. So: 55% today / tomorrow / day after,
    25% within two weeks (weekday names, "agle hafte"), 20% anywhere in range.
    """
    r = rng.random()
    if r < 0.55:
        offset = rng.choice([0, 1, 1, 1, 2, 2, 2])
    elif r < 0.8:
        offset = rng.randint(3, 13)
    else:
        offset = rng.randint(lo, hi)
    return min(max(offset, lo), hi)


def _future_datetime(rng: random.Random, now: datetime, max_days: int = 14) -> datetime:
    while True:
        day = now.date() + timedelta(days=_day_offset(rng, 0, max_days))
        hh = rng.choice([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
        mm = rng.choice([0, 0, 0, 30, 30, 15, 45])
        dt = datetime(day.year, day.month, day.day, hh, mm)
        if dt > now + timedelta(minutes=30):
            return dt


def _datetime_hint(rng: random.Random, dt: datetime, now: datetime) -> tuple[str, str]:
    """Returns (iso, hint). Sometimes gives only the time (next occurrence rule)."""
    t = time_phrase(rng, dt.hour, dt.minute)
    next_occurrence = (dt.date() == now.date()) or (
        dt.date() == now.date() + timedelta(days=1) and dt.time() <= now.time()
    )
    if next_occurrence and rng.random() < 0.4:
        return dt.strftime("%Y-%m-%dT%H:%M"), f'Time: say only "{t}", no day'
    d = future_date_phrase(rng, dt.date(), now.date())
    return dt.strftime("%Y-%m-%dT%H:%M"), f'When: "{d}" at "{t}"'


def _phone(rng: random.Random) -> str:
    return str(rng.choice([6, 7, 8, 9])) + "".join(str(rng.randint(0, 9)) for _ in range(9))


# --------------------------------------------------------------------------- samplers


def s_check_balance(rng: random.Random, ctx: Context) -> Draft:
    account = rng.choices(["savings", "current", "credit_card"], [6, 2, 2])[0]
    d = Draft("check_balance", {"account": account})
    if account == "savings":
        if rng.random() < 0.5:
            d.hints["account"] = "Account: savings (say it)"
        else:
            d.extra.append("Do not name the account type")
    elif account == "credit_card":
        d.hints["account"] = "Account: credit card (ask how much is outstanding/due)"
    else:
        d.hints["account"] = "Account: current account"
    return d


def s_transfer_money(rng: random.Random, ctx: Context) -> Draft:
    to = rng.choice(lx.PEOPLE)
    big = rng.random() < 0.12  # rent, fees: the amounts people say in lakh
    amount = rng.choice([100000, 150000, 200000, 250000]) if big else sample_amount(rng, 100, 50000)
    d = Draft("transfer_money", {"recipient": to, "amount": amount})
    d.hints["recipient"] = f"Recipient: {to}"
    d.hints["amount"] = f'Amount: "{amount_phrase(rng, amount)}"'
    if rng.random() < 0.35:
        mode = rng.choice(["upi", "imps", "neft"])
        d.args["mode"] = mode
        d.hints["mode"] = f"Mode: {mode.upper()}"
    if rng.random() < 0.3:
        note, said = rng.choice(lx.TRANSFER_NOTES)
        d.args["note"] = note
        d.hints["note"] = f'Note for the payee: "{said}"'
    return d


def s_get_transactions(rng: random.Random, ctx: Context) -> Draft:
    d = Draft("get_transactions", {})
    if rng.random() < 0.25:
        d.args["account"] = "credit_card"
        d.hints["account"] = "Account: credit card"
    if rng.random() < 0.75:
        start, end, phrase = past_range_phrase(rng, ctx.dt.date())
        d.args["start_date"], d.args["end_date"] = start.isoformat(), end.isoformat()
        d.hints["start_date"] = f'Period: "{phrase}" transactions'
    if rng.random() < 0.35:
        cat = rng.choice(list(lx.CATEGORY_PHRASES))
        d.args["category"] = cat
        d.hints["category"] = f'Category: "{rng.choice(lx.CATEGORY_PHRASES[cat])}"'
    if rng.random() < 0.2:
        amt = rng.choice([500, 1000, 2000, 5000])
        d.args["min_amount"] = amt
        d.hints["min_amount"] = f'Only those of "{amount_phrase(rng, amt)}" or more'
    if len(d.args) == 0:
        d.extra.append("The user just wants to see recent transactions")
    return d


def s_pay_bill(rng: random.Random, ctx: Context) -> Draft:
    biller = rng.choice(list(lx.BILLER_PHRASES))
    d = Draft("pay_bill", {"biller": biller})
    d.hints["biller"] = f'Bill: "{rng.choice(lx.BILLER_PHRASES[biller])}"'
    if rng.random() < 0.4:
        amt = sample_amount(rng, 150, 5000)
        d.args["amount"] = amt
        d.hints["amount"] = f'Amount: "{amount_phrase(rng, amt)}"'
    else:
        d.extra.append("No amount: pay whatever is due")
    return d


def s_recharge_mobile(rng: random.Random, ctx: Context) -> Draft:
    phone = _phone(rng)
    amt = rng.choice([149, 199, 239, 299, 349, 399, 479, 599, 666, 719, 999])
    d = Draft("recharge_mobile", {"phone_number": phone, "amount": amt})
    style = rng.choice(["plain", "spaced", "plus91"])
    shown = {"plain": phone, "spaced": f"{phone[:5]} {phone[5:]}", "plus91": f"+91 {phone}"}[style]
    d.hints["phone_number"] = f"Number: {shown}"
    d.hints["amount"] = f'Pack: "{amt}"'
    return d


def s_search_trains(rng: random.Random, ctx: Context) -> Draft:
    src = ctx.city if rng.random() < 0.4 else rng.choice(lx.CITIES)
    dst = rng.choice([c for c in lx.CITIES if c != src])
    day = ctx.dt.date() + timedelta(days=_day_offset(rng, 0, 20))
    d = Draft("search_trains", {"source": src, "destination": dst, "date": day.isoformat()})
    if src == ctx.city and rng.random() < 0.7:
        d.hints["source"] = "Do not name the departure city (user leaves from where they are)"
    else:
        d.hints["source"] = _city_hint(rng, "From", src)
    d.hints["destination"] = _city_hint(rng, "To", dst)
    d.hints["date"] = f'Date: "{future_date_phrase(rng, day, ctx.dt.date())}"'
    if rng.random() < 0.5:
        cls, said = rng.choice([("SL", "sleeper"), ("3A", "3AC / third AC"), ("2A", "2AC"),
                                ("1A", "first AC"), ("CC", "chair car")])
        d.args["travel_class"] = cls
        d.hints["travel_class"] = f'Class: "{said}"'
    r = rng.random()
    if r < 0.3:
        hh, mm = rng.choice([5, 6, 7, 8, 9, 10, 14, 16, 18]), rng.choice([0, 0, 30])
        d.args["depart_after"] = f"{hh:02d}:{mm:02d}"
        d.hints["depart_after"] = f'Leaving after "{time_phrase(rng, hh, mm)}"'
    elif r < 0.45:
        hh = rng.choice([9, 10, 12, 14, 18, 20, 22])
        d.args["depart_before"] = f"{hh:02d}:00"
        d.hints["depart_before"] = f'Leaving before "{time_phrase(rng, hh, 0)}"'
    return d


def s_check_pnr_status(rng: random.Random, ctx: Context) -> Draft:
    pnr = "".join(str(rng.randint(0, 9)) for _ in range(10))
    pnr = str(rng.randint(1, 9)) + pnr[1:]
    return Draft("check_pnr_status", {"pnr": pnr}, {"pnr": f"PNR: {pnr}"})


def s_search_flights(rng: random.Random, ctx: Context) -> Draft:
    src = ctx.city if ctx.city in lx.AIRPORT_CITIES and rng.random() < 0.3 else rng.choice(
        lx.AIRPORT_CITIES)
    dst = rng.choice([c for c in lx.AIRPORT_CITIES if c != src])
    day = ctx.dt.date() + timedelta(days=_day_offset(rng, 1, 30))
    d = Draft("search_flights", {"source": src, "destination": dst, "date": day.isoformat()})
    if src == ctx.city and rng.random() < 0.6:
        d.hints["source"] = "Do not name the departure city (user flies from where they are)"
    else:
        d.hints["source"] = _city_hint(rng, "From", src)
    d.hints["destination"] = _city_hint(rng, "To", dst)
    d.hints["date"] = f'Date: "{future_date_phrase(rng, day, ctx.dt.date())}"'
    if rng.random() < 0.25:
        ret = day + timedelta(days=rng.randint(2, 10))
        d.args["return_date"] = ret.isoformat()
        d.hints["return_date"] = f'Return: "{future_date_phrase(rng, ret, ctx.dt.date())}"'
    if rng.random() < 0.3:
        n = rng.randint(2, 5)
        d.args["passengers"] = n
        d.hints["passengers"] = f'Travellers: "{quantity_phrase(rng, n)} log"'
    if rng.random() < 0.15:
        d.args["cabin"] = "business"
        d.hints["cabin"] = "Cabin: business class"
    return d


def s_book_cab(rng: random.Random, ctx: Context) -> Draft:
    drop = rng.choice(lx.LANDMARKS)
    d = Draft("book_cab", {"drop": drop})
    d.hints["drop"] = f"Drop: {drop}"
    if rng.random() < 0.4:
        pickup = rng.choice([x for x in lx.LANDMARKS if x != drop])
        d.args["pickup"] = pickup
        d.hints["pickup"] = f"Pickup: {pickup}"
    else:
        d.extra.append("Pickup is where the user is now (say 'yahan se' or nothing)")
    if rng.random() < 0.6:
        ride, said = rng.choice([("auto", "auto"), ("bike", "bike taxi"), ("mini", "mini cab"),
                                 ("sedan", "sedan"), ("suv", "SUV / badi gaadi")])
        d.args["ride_type"] = ride
        d.hints["ride_type"] = f'Vehicle: "{said}"'
    if rng.random() < 0.3:
        dt = _future_datetime(rng, ctx.dt, max_days=1)
        iso, hint = _datetime_hint(rng, dt, ctx.dt)
        d.args["schedule_at"] = iso
        d.hints["schedule_at"] = hint
    else:
        d.extra.append("The ride is needed right now")
    return d


def s_search_restaurants(rng: random.Random, ctx: Context) -> Draft:
    d = Draft("search_restaurants", {})
    while not d.args:
        if rng.random() < 0.3:
            area = rng.choice(lx.AREAS)
            d.args["location"] = area
            d.hints["location"] = f"Area: {area}"
        if rng.random() < 0.55:
            cuisine = rng.choice(list(lx.CUISINES))
            d.args["cuisine"] = cuisine
            d.hints["cuisine"] = f'Cuisine: "{rng.choice(lx.CUISINES[cuisine])}"'
        if rng.random() < 0.3:
            d.args["veg_only"] = True
            d.hints["veg_only"] = "Pure veg only"
        if rng.random() < 0.25:
            amt = rng.choice([300, 500, 800, 1000, 1500, 2000])
            d.args["max_price_for_two"] = amt
            d.hints["max_price_for_two"] = f'Budget for two: "{amount_phrase(rng, amt)}"'
        if rng.random() < 0.3:
            sort, said = rng.choice([("rating", "best rated first"), ("distance", "nearest first"),
                                     ("price_low_to_high", "cheapest first")])
            d.args["sort_by"] = sort
            d.hints["sort_by"] = f"Sort: {said}"
    if "location" not in d.args:
        d.extra.append("Near the user (do not name an area)")
    return d


def s_order_food(rng: random.Random, ctx: Context) -> Draft:
    rest = rng.choice(lx.RESTAURANTS)
    dishes = rng.sample(lx.DISHES, rng.choice([1, 1, 2, 2, 3]))
    items, said = [], []
    for dish in dishes:
        q = rng.choice([1, 1, 1, 2, 2, 3, 4])
        items.append({"name": dish, "quantity": q})
        said.append(f"{quantity_phrase(rng, q)} {dish}" if q > 1 or rng.random() < 0.5 else dish)
    d = Draft("order_food", {"restaurant": rest, "items": items})
    d.hints["restaurant"] = f"Restaurant: {rest}"
    d.hints["items"] = "Items: " + ", ".join(said)
    if rng.random() < 0.3:
        pay, phrase = rng.choice([("card", "card se"), ("cash_on_delivery", "cash on delivery / COD"),
                                  ("upi", "UPI se")])
        d.args["payment_mode"] = pay
        d.hints["payment_mode"] = f'Payment: "{phrase}"'
    return d


def s_track_order(rng: random.Random, ctx: Context) -> Draft:
    if rng.random() < 0.35:
        oid = f"OD{rng.randint(100000, 999999)}"
        return Draft("track_order", {"order_id": oid}, {"order_id": f"Order id: {oid}"})
    return Draft("track_order", {}, extra=["Asks about their latest food order, no id"])


def s_set_reminder(rng: random.Random, ctx: Context) -> Draft:
    title, said = rng.choice(lx.REMINDER_TITLES)
    dt = _future_datetime(rng, ctx.dt)
    repeat = rng.choices(["none", "daily", "weekly", "monthly"], [7, 2, 1, 1])[0]
    d = Draft("set_reminder", {"title": title})
    d.hints["title"] = f'About: "{said}"'
    if repeat == "daily":
        # "roz subah 8 baje": next occurrence of that time
        t = dt.time()
        first = datetime.combine(ctx.dt.date(), t)
        if first <= ctx.dt:
            first += timedelta(days=1)
        d.args["remind_at"] = first.strftime("%Y-%m-%dT%H:%M")
        d.hints["remind_at"] = f'Time: "{time_phrase(rng, t.hour, t.minute)}", every day (roz)'
        d.args["repeat"] = "daily"
    else:
        iso, hint = _datetime_hint(rng, dt, ctx.dt)
        d.args["remind_at"] = iso
        d.hints["remind_at"] = hint
        if repeat != "none":
            d.args["repeat"] = repeat
            d.hints["repeat"] = {"weekly": "Repeats every week (har hafte)",
                                 "monthly": "Repeats every month (har mahine)"}[repeat]
    return d


def s_set_alarm(rng: random.Random, ctx: Context) -> Draft:
    hh = rng.choice([4, 5, 5, 6, 6, 6, 7, 7, 8, 9, 14, 22])
    mm = rng.choice([0, 0, 15, 30, 30, 45])
    d = Draft("set_alarm", {"time": f"{hh:02d}:{mm:02d}"})
    d.hints["time"] = f'Time: "{time_phrase(rng, hh, mm)}"'
    if rng.random() < 0.3:
        label, said = rng.choice(lx.ALARM_LABELS)
        d.args["label"] = label
        d.hints["label"] = f'Label: "{said}"'
    if rng.random() < 0.35:
        days, said = rng.choice([
            (["mon", "tue", "wed", "thu", "fri"], "weekdays (Monday se Friday)"),
            (["sat", "sun"], "weekends"),
            (["mon", "tue", "wed", "thu", "fri", "sat", "sun"], "roz / every day"),
            (["mon", "wed", "fri"], "Monday, Wednesday, Friday"),
        ])
        d.args["repeat_days"] = days
        d.hints["repeat_days"] = f"Repeat: {said}"
    return d


def s_create_calendar_event(rng: random.Random, ctx: Context) -> Draft:
    title, said = rng.choice(lx.EVENT_TITLES)
    dt = _future_datetime(rng, ctx.dt)
    iso, hint = _datetime_hint(rng, dt, ctx.dt)
    d = Draft("create_calendar_event", {"title": title, "start": iso})
    d.hints["title"] = f'Event: "{said}"'
    d.hints["start"] = hint
    if rng.random() < 0.4:
        mins, phrase = rng.choice([(30, "aadha ghanta"), (45, "45 minute"), (90, "dedh ghanta"),
                                   (120, "do ghante"), (15, "15 minute")])
        d.args["duration_minutes"] = mins
        d.hints["duration_minutes"] = f'Length: "{phrase}"'
    if rng.random() < 0.4:
        people = rng.sample([p for p in lx.PEOPLE if p[0].isupper() and " " not in p][:16],
                            rng.randint(1, 3))
        d.args["attendees"] = people
        d.hints["attendees"] = "Invite: " + ", ".join(people)
    return d


def s_get_weather(rng: random.Random, ctx: Context) -> Draft:
    here = rng.random() < 0.4
    city = ctx.city if here else rng.choice(lx.CITIES)
    d = Draft("get_weather", {"city": city})
    d.hints["city"] = ("Do not name the city: the user means where they are ('yahan')"
                       if here else _city_hint(rng, "City", city))
    if rng.random() < 0.5:
        day = ctx.dt.date() + timedelta(days=_day_offset(rng, 0, 5))
        d.args["date"] = day.isoformat()
        d.hints["date"] = f'Day: "{future_date_phrase(rng, day, ctx.dt.date())}"'
    else:
        d.extra.append("Asks about the weather right now")
    return d


def s_convert_currency(rng: random.Random, ctx: Context) -> Draft:
    codes = list(lx.CURRENCY_PHRASES)
    if rng.random() < 0.6:
        frm, to = rng.choice([c for c in codes if c != "INR"]), "INR"
    else:
        frm, to = "INR", rng.choice([c for c in codes if c != "INR"])
    amount = rng.choice([1, 10, 50, 100, 250, 500, 1000, 5000, 20000, 100000])
    d = Draft("convert_currency", {"amount": amount, "from_currency": frm, "to_currency": to})
    d.hints["amount"] = f"Amount: {amount}"
    d.hints["from_currency"] = f'From: "{rng.choice(lx.CURRENCY_PHRASES[frm])}"'
    d.hints["to_currency"] = f'To: "{rng.choice(lx.CURRENCY_PHRASES[to])}"'
    return d


SAMPLERS: dict[str, Callable[[random.Random, Context], Draft]] = {
    "check_balance": s_check_balance,
    "transfer_money": s_transfer_money,
    "get_transactions": s_get_transactions,
    "pay_bill": s_pay_bill,
    "recharge_mobile": s_recharge_mobile,
    "search_trains": s_search_trains,
    "check_pnr_status": s_check_pnr_status,
    "search_flights": s_search_flights,
    "book_cab": s_book_cab,
    "search_restaurants": s_search_restaurants,
    "order_food": s_order_food,
    "track_order": s_track_order,
    "set_reminder": s_set_reminder,
    "set_alarm": s_set_alarm,
    "create_calendar_event": s_create_calendar_event,
    "get_weather": s_get_weather,
    "convert_currency": s_convert_currency,
}
assert set(SAMPLERS) == set(TOOLS), "every tool needs a sampler"

# Required arguments a user can plausibly forget, and arguments that make no sense
# once that one is gone (e.g. a departure time without a date).
CLARIFY_FIELDS: dict[str, list[str]] = {
    "transfer_money": ["amount", "recipient"],
    "recharge_mobile": ["phone_number", "amount"],
    "search_trains": ["destination", "date"],
    "search_flights": ["destination", "date"],
    "book_cab": ["drop"],
    "order_food": ["restaurant", "items"],
    "check_pnr_status": ["pnr"],
    "set_reminder": ["remind_at"],
    "set_alarm": ["time"],
    "create_calendar_event": ["start"],
}
DEPENDENTS: dict[tuple[str, str], list[str]] = {
    ("search_trains", "date"): ["depart_after", "depart_before"],
    ("search_flights", "date"): ["return_date"],
}
FIELD_NAMES: dict[str, str] = {
    "amount": "the amount", "recipient": "who to send it to", "phone_number": "the phone number",
    "destination": "where they are going", "date": "the date", "drop": "where they want to go",
    "restaurant": "which restaurant", "items": "what to order", "pnr": "the PNR number",
    "remind_at": "when to remind", "time": "the alarm time", "start": "when the event is",
}


# --------------------------------------------------------------------------- scenarios


class Scenario(BaseModel):
    id: str
    category: str
    script: str
    style: list[str]
    context: Context
    turns: list[list[str]]  # phrasing hints for each user turn
    expected: Expected
    meta: dict[str, Any] = Field(default_factory=dict)


CATEGORY_WEIGHTS = {"single": 45, "parallel": 15, "multi_turn": 15, "clarify": 12,
                    "irrelevant": 13}
SCRIPT_WEIGHTS = {"roman": 60, "devanagari": 20, "english": 20}
STYLE_OPTIONS = [
    ("casual texting: lowercase, few or no punctuation marks, short forms like 'kr do', 'plz'", 0.35),
    ("polite: uses 'please', 'zara', 'ji'", 0.25),
    ("adds a little irrelevant small talk (e.g. 'bahut busy din hai'), which must not read "
     "as a note, title, reason for payment or any other detail of the request", 0.2),
    ("one or two small typos", 0.15),
]


def sample_context(rng: random.Random) -> Context:
    start = datetime(2026, 11, 1)
    day = start + timedelta(days=rng.randint(0, 150))  # Nov 2026 - Mar 2027, crosses a year
    now = day.replace(hour=rng.randint(7, 22), minute=rng.choice([0, 10, 20, 30, 40, 50]))
    return Context(now=now.strftime("%Y-%m-%dT%H:%M"), city=rng.choice(lx.CITIES))


def _call(d: Draft) -> ToolCall:
    canonicalize(d.tool, d.args)  # fail loudly if a sampler ever emits an invalid call
    return ToolCall(name=d.tool, arguments=d.args)


def _weighted(rng: random.Random, weights: dict[str, int]) -> str:
    return rng.choices(list(weights), list(weights.values()))[0]


def build_scenario(rng: random.Random, sid: str, category: str | None = None) -> Scenario:
    ctx = sample_context(rng)
    category = category or _weighted(rng, CATEGORY_WEIGHTS)
    script = _weighted(rng, SCRIPT_WEIGHTS)
    style = [s for s, p in STYLE_OPTIONS if rng.random() < p]
    meta: dict[str, Any] = {}

    if category == "single":
        d = SAMPLERS[rng.choice(list(SAMPLERS))](rng, ctx)
        turns = [d.all_hints()]
        expected = Expected(action="call", tool_calls=[_call(d)])
        meta["tools"] = [d.tool]

    elif category == "parallel":
        t1, t2 = rng.sample(list(SAMPLERS), 2)
        d1, d2 = SAMPLERS[t1](rng, ctx), SAMPLERS[t2](rng, ctx)
        turns = [["Request 1 -> " + h for h in d1.all_hints()]
                 + ["Request 2 -> " + h for h in d2.all_hints()]
                 + ["Ask for both things in one message"]]
        expected = Expected(action="call", tool_calls=[_call(d1), _call(d2)])
        meta["tools"] = [t1, t2]

    elif category in ("clarify", "multi_turn"):
        tool = rng.choice(list(CLARIFY_FIELDS))
        d = SAMPLERS[tool](rng, ctx)
        missing = rng.choice(CLARIFY_FIELDS[tool])
        for dep in DEPENDENTS.get((tool, missing), []):
            d.args.pop(dep, None)
            d.hints.pop(dep, None)
        missing_hint = d.hints.pop(missing)
        first = d.all_hints() + [
            f"The user forgets to say {FIELD_NAMES[missing]}; do not mention or imply it"]
        meta.update(tools=[tool], missing=missing)
        if category == "clarify":
            d.args.pop(missing)
            turns = [first]
            expected = Expected(action="clarify")
        else:
            turns = [first, [missing_hint]]
            expected = Expected(action="call", tool_calls=[_call(d)])

    elif category == "irrelevant":
        ask, hard = rng.choice(lx.OUT_OF_SCOPE)
        turns = [[f"The user {ask}"]]
        expected = Expected(action="respond")
        meta.update(hard_negative=hard, topic=ask)

    else:
        raise ValueError(f"unknown category {category!r}")

    return Scenario(id=sid, category=category, script=script, style=style, context=ctx,
                    turns=turns, expected=expected, meta=meta)


def build_scenarios(n: int, seed: int, prefix: str = "s") -> list[Scenario]:
    rng = random.Random(seed)
    width = len(str(n))
    return [build_scenario(rng, f"{prefix}{i:0{width}d}") for i in range(n)]


__all__ = ["Draft", "Scenario", "SAMPLERS", "build_scenario", "build_scenarios", "WEEKDAYS_EN"]
