"""Fake implementations of the 17 tools, for the demo.

Results are made up but deterministic (seeded by the call), so the demo can show the
whole loop (request, tool call, result) without touching any real bank or booking
system. Nothing here is used by the benchmark.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta
from typing import Any

from .tools import TOOLS


def _rng(name: str, args: dict) -> random.Random:
    key = name + json.dumps(args, sort_keys=True, ensure_ascii=False)
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))


def _inr(x: float) -> str:
    whole = int(round(x))
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        s = ",".join(([head] if head else []) + groups) + "," + tail
    return f"₹{s}"


def execute(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Return ``{"summary": str, "data": ...}`` for a validated call."""
    if name not in TOOLS:
        raise KeyError(name)
    full = TOOLS[name].params.model_validate(args).model_dump(mode="json")
    r = _rng(name, args)
    fn = _HANDLERS[name]
    summary, data = fn(r, full)
    return {"tool": name, "summary": summary, "data": data}


def _balance(r, a):
    amt = r.randint(2_000, 250_000)
    what = "outstanding" if a["account"] == "credit_card" else "available"
    return f"{a['account'].replace('_', ' ').title()}: {_inr(amt)} {what}", {"balance": amt}


def _transfer(r, a):
    ref = f"UPI{r.randint(10**11, 10**12 - 1)}"
    note = f" ({a['note']})" if a.get("note") else ""
    return (f"Sent {_inr(a['amount'])} to {a['recipient']} via {a['mode'].upper()}{note}. "
            f"Ref {ref}", {"reference": ref, "status": "success"})


def _transactions(r, a):
    merchants = ["Swiggy", "BigBasket", "Amazon", "IRCTC", "Jio", "BookMyShow", "Zomato"]
    rows = [{"merchant": r.choice(merchants), "amount": r.randint(80, 4_000)} for _ in range(4)]
    span = f"{a.get('start_date') or 'last 30 days'} to {a.get('end_date') or 'today'}"
    return f"{len(rows)} transactions, {span}", rows


def _bill(r, a):
    amt = a.get("amount") or r.randint(300, 3_500)
    return f"Paid {a['biller'].replace('_', ' ')} bill: {_inr(amt)}", {"paid": amt}


def _recharge(r, a):
    return (f"Recharged {a['phone_number']} with {_inr(a['amount'])}",
            {"status": "success"})


def _trains(r, a):
    names = ["Shatabdi Express", "Rajdhani Express", "Vande Bharat", "Superfast Express",
             "Mail Express", "Intercity Express"]
    lo = int((a.get("depart_after") or "05:00")[:2])
    hi = int((a.get("depart_before") or "23:00")[:2])
    rows = []
    for _ in range(3):
        h = r.randint(min(lo, hi), max(lo, hi))
        rows.append({"train": f"{r.randint(12001, 22999)} {r.choice(names)}",
                     "departs": f"{h:02d}:{r.choice(['05', '15', '30', '45'])}",
                     "class": a.get("travel_class") or r.choice(["SL", "3A", "2A"]),
                     "seats": r.choice(["AVAILABLE 42", "AVAILABLE 8", "RAC 12", "WL 23"])})
    rows.sort(key=lambda x: x["departs"])
    return f"{len(rows)} trains {a['source']} → {a['destination']} on {a['date']}", rows


def _pnr(r, a):
    status = r.choice(["CNF B2 34", "CNF S5 61", "RAC 7", "WL 12"])
    return f"PNR {a['pnr']}: {status}", {"status": status}


def _flights(r, a):
    airlines = ["IndiGo", "Air India", "Akasa Air", "SpiceJet"]
    rows = [{"airline": r.choice(airlines), "departs": f"{r.randint(5, 22):02d}:{r.choice(['00', '20', '40'])}",
             "fare": r.randint(3_000, 14_000) * a["passengers"]} for _ in range(3)]
    rows.sort(key=lambda x: x["fare"])
    trip = f", back {a['return_date']}" if a.get("return_date") else ""
    return (f"{len(rows)} flights {a['source']} → {a['destination']} on {a['date']}{trip}, "
            f"from {_inr(rows[0]['fare'])}", rows)


def _cab(r, a):
    pickup = "your location" if a["pickup"] == "current_location" else a["pickup"]
    when = f" at {a['schedule_at'].replace('T', ' ')}" if a.get("schedule_at") else ", arriving in " \
        f"{r.randint(2, 9)} min"
    fare = r.randint(60, 600)
    return (f"{a['ride_type'].title()} booked: {pickup} → {a['drop']}{when}. Est. {_inr(fare)}",
            {"fare_estimate": fare})


def _restaurants(r, a):
    adj = ["Spice", "Royal", "Green", "Urban", "Old", "Tandoor"]
    noun = ["Kitchen", "Dhaba", "Bistro", "House", "Cafe", "Junction"]
    rows = [{"name": f"{r.choice(adj)} {r.choice(noun)}", "rating": round(r.uniform(3.6, 4.8), 1),
             "for_two": r.randint(3, 15) * 100} for _ in range(3)]
    where = a.get("location") or "near you"
    what = f"{a['cuisine']} " if a.get("cuisine") else ""
    return f"3 {what}restaurants {where}", rows


def _order(r, a):
    items = ", ".join(f"{i['quantity']}× {i['name']}" for i in a["items"])
    oid = f"FD{r.randint(100000, 999999)}"
    return (f"Ordered {items} from {a['restaurant']}. Order {oid}, ETA {r.randint(25, 50)} min",
            {"order_id": oid})


def _track(r, a):
    status = r.choice(["being prepared", "picked up", "4 min away"])
    which = a.get("order_id") or "your latest order"
    return f"Order {which}: {status}", {"status": status}


def _reminder(r, a):
    rep = "" if a["repeat"] == "none" else f", {a['repeat']}"
    return f"Reminder '{a['title']}' at {a['remind_at'].replace('T', ' ')}{rep}", {"id": r.randint(1, 999)}


def _alarm(r, a):
    days = f" on {', '.join(a['repeat_days'])}" if a.get("repeat_days") else ""
    label = f" ({a['label']})" if a.get("label") else ""
    return f"Alarm set for {a['time']}{days}{label}", {"id": r.randint(1, 999)}


def _event(r, a):
    start = datetime.strptime(a["start"], "%Y-%m-%dT%H:%M")
    end = start + timedelta(minutes=a["duration_minutes"])
    who = f" with {', '.join(a['attendees'])}" if a.get("attendees") else ""
    return (f"'{a['title']}' {start:%a %d %b, %H:%M}–{end:%H:%M}{who}", {"id": r.randint(1, 999)})


def _weather(r, a):
    temp = r.randint(8, 34)
    sky = r.choice(["clear", "hazy", "partly cloudy", "light rain"])
    when = a.get("date") or "now"
    return f"{a['city']}, {when}: {temp}°C, {sky}", {"temp_c": temp, "sky": sky}


_RATES = {"INR": 1.0, "USD": 88.0, "EUR": 102.0, "GBP": 117.0, "AED": 24.0, "SGD": 68.0,
          "JPY": 0.58, "AUD": 57.0, "CAD": 63.0, "THB": 2.7}


def _currency(r, a):
    src = _RATES.get(a["from_currency"], 1.0)
    dst = _RATES.get(a["to_currency"], 1.0)
    out = a["amount"] * src / dst
    return (f"{a['amount']:g} {a['from_currency']} ≈ {out:,.2f} {a['to_currency']} "
            "(illustrative rate)", {"converted": round(out, 2)})


_HANDLERS = {
    "check_balance": _balance, "transfer_money": _transfer, "get_transactions": _transactions,
    "pay_bill": _bill, "recharge_mobile": _recharge, "search_trains": _trains,
    "check_pnr_status": _pnr, "search_flights": _flights, "book_cab": _cab,
    "search_restaurants": _restaurants, "order_food": _order, "track_order": _track,
    "set_reminder": _reminder, "set_alarm": _alarm, "create_calendar_event": _event,
    "get_weather": _weather, "convert_currency": _currency,
}
assert set(_HANDLERS) == set(TOOLS), set(TOOLS) ^ set(_HANDLERS)
