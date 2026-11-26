"""Tool registry: the fictional APIs the assistant can call.

Each tool's parameters are a pydantic model. One definition gives us three things:

* the JSON schema shown to the model in the system prompt,
* strict validation of predicted calls (wrong types, unknown keys, bad formats), and
* the per-field matching policy the evaluator uses (``x-match``).

The APIs are deliberately generic (no real banks, airlines or delivery apps).
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Matching policies, read by htb.eval.matching.
EXACT = "exact"  # enums, numbers, booleans, dates, ids
ENTITY = "entity"  # cities, people, restaurants: case/spacing/known aliases ignored
FUZZY = "fuzzy"  # free text such as titles, notes and dish names


def arg(default: Any = ..., *, description: str, match: str = EXACT, **kw: Any) -> Any:
    return Field(default, description=description, json_schema_extra={"x-match": match}, **kw)


ISODate = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
Time24 = Annotated[str, Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
DateTime = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T([01]\d|2[0-3]):[0-5]\d$")]
Rupees = Annotated[float, Field(gt=0)]
Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
Currency = Literal["INR", "USD", "EUR", "GBP", "AED", "JPY", "SGD", "CAD", "AUD"]


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- banking


class CheckBalance(Params):
    account: Literal["savings", "current", "credit_card"] = arg(
        "savings", description="Which account. For credit_card this is the outstanding amount."
    )


class TransferMoney(Params):
    recipient: str = arg(description="Name of the person or saved payee.", match=ENTITY)
    amount: Rupees = arg(description="Amount in INR.")
    mode: Literal["upi", "imps", "neft"] = arg("upi", description="Transfer rail.")
    note: str | None = arg(None, description="Short note for the payee, in English.", match=FUZZY)


class GetTransactions(Params):
    account: Literal["savings", "current", "credit_card"] = arg("savings", description="Account.")
    start_date: ISODate | None = arg(None, description="First day of the range, YYYY-MM-DD.")
    end_date: ISODate | None = arg(None, description="Last day of the range, YYYY-MM-DD.")
    category: (
        Literal["food", "groceries", "shopping", "travel", "bills", "entertainment"] | None
    ) = arg(None, description="Spending category filter.")
    min_amount: Rupees | None = arg(None, description="Only transactions at or above this amount.")


class PayBill(Params):
    biller: Literal["electricity", "mobile_postpaid", "broadband", "gas", "water", "dth"] = arg(
        description="Type of bill."
    )
    amount: Rupees | None = arg(None, description="Amount in INR. Omit to pay the full amount due.")


class RechargeMobile(Params):
    phone_number: Annotated[str, Field(pattern=r"^[6-9]\d{9}$")] = arg(
        description="10-digit Indian mobile number, digits only."
    )
    amount: Rupees = arg(description="Recharge amount in INR.")


# --------------------------------------------------------------------------- travel


class SearchTrains(Params):
    source: str = arg(description="Departure city.", match=ENTITY)
    destination: str = arg(description="Arrival city.", match=ENTITY)
    date: ISODate = arg(description="Travel date, YYYY-MM-DD.")
    travel_class: Literal["SL", "3A", "2A", "1A", "CC"] | None = arg(
        None, description="SL=sleeper, 3A/2A/1A=AC tiers, CC=AC chair car."
    )
    depart_after: Time24 | None = arg(None, description="Earliest departure, HH:MM 24-hour.")
    depart_before: Time24 | None = arg(None, description="Latest departure, HH:MM 24-hour.")


class CheckPnrStatus(Params):
    pnr: Annotated[str, Field(pattern=r"^\d{10}$")] = arg(description="10-digit PNR number.")


class SearchFlights(Params):
    source: str = arg(description="Departure city.", match=ENTITY)
    destination: str = arg(description="Arrival city.", match=ENTITY)
    date: ISODate = arg(description="Departure date, YYYY-MM-DD.")
    return_date: ISODate | None = arg(None, description="Return date for a round trip.")
    passengers: int = arg(1, description="Number of travellers.", ge=1, le=9)
    cabin: Literal["economy", "business"] = arg("economy", description="Cabin class.")


class BookCab(Params):
    drop: str = arg(description="Drop location.", match=FUZZY)
    pickup: str = arg(
        "current_location", description="Pickup location. Default is the user's current location.",
        match=FUZZY,
    )
    ride_type: Literal["auto", "bike", "mini", "sedan", "suv"] = arg("mini", description="Vehicle.")
    schedule_at: DateTime | None = arg(
        None, description="Pickup time YYYY-MM-DDTHH:MM. Omit for an immediate ride."
    )


# --------------------------------------------------------------------------- food


class SearchRestaurants(Params):
    location: str | None = arg(
        None, description="Area to search in. Omit to search near the user.", match=FUZZY
    )
    cuisine: str | None = arg(None, description="Cuisine, in English.", match=FUZZY)
    veg_only: bool = arg(False, description="Only pure vegetarian restaurants.")
    max_price_for_two: Rupees | None = arg(None, description="Budget for two people in INR.")
    sort_by: Literal["relevance", "rating", "distance", "price_low_to_high"] = arg(
        "relevance", description="Result ordering."
    )


class OrderItem(Params):
    name: str = arg(description="Dish name, in English/romanized form.", match=FUZZY)
    quantity: int = arg(1, description="How many.", ge=1, le=20)


class OrderFood(Params):
    restaurant: str = arg(description="Restaurant name.", match=ENTITY)
    items: list[OrderItem] = arg(description="Dishes to order.", min_length=1)
    payment_mode: Literal["upi", "card", "cash_on_delivery"] = arg("upi", description="Payment.")


class TrackOrder(Params):
    order_id: str | None = arg(None, description="Order id. Omit for the most recent order.")


# --------------------------------------------------------------------------- productivity


class SetReminder(Params):
    title: str = arg(description="What to remind about, short, in English.", match=FUZZY)
    remind_at: DateTime = arg(description="When, YYYY-MM-DDTHH:MM.")
    repeat: Literal["none", "daily", "weekly", "monthly"] = arg("none", description="Recurrence.")


class SetAlarm(Params):
    time: Time24 = arg(description="Alarm time, HH:MM 24-hour.")
    label: str | None = arg(None, description="Short label, in English.", match=FUZZY)
    repeat_days: list[Weekday] | None = arg(
        None, description="Days the alarm repeats on. Omit for a one-time alarm."
    )


class CreateCalendarEvent(Params):
    title: str = arg(description="Event title, short, in English.", match=FUZZY)
    start: DateTime = arg(description="Start, YYYY-MM-DDTHH:MM.")
    duration_minutes: int = arg(60, description="Length in minutes.", ge=5, le=1440)
    attendees: list[str] | None = arg(None, description="People to invite.", match=ENTITY)


# --------------------------------------------------------------------------- utility


class GetWeather(Params):
    city: str = arg(description="City name.", match=ENTITY)
    date: ISODate | None = arg(None, description="Forecast date. Omit for current weather.")


class ConvertCurrency(Params):
    amount: float = arg(description="Amount to convert.", gt=0)
    from_currency: Currency = arg(description="ISO code of the source currency.")
    to_currency: Currency = arg(description="ISO code of the target currency.")


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Tool:
    name: str
    domain: str
    description: str
    params: type[Params]

    def json_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": clean_schema(self.params.model_json_schema()),
        }


_TOOL_LIST = [
    Tool("check_balance", "banking", "Get the balance of a bank account.", CheckBalance),
    Tool("transfer_money", "banking", "Send money to a person.", TransferMoney),
    Tool("get_transactions", "banking", "List past transactions.", GetTransactions),
    Tool("pay_bill", "banking", "Pay a utility bill.", PayBill),
    Tool("recharge_mobile", "banking", "Prepaid mobile recharge.", RechargeMobile),
    Tool("search_trains", "travel", "Search trains between two cities.", SearchTrains),
    Tool("check_pnr_status", "travel", "Check status of a train booking.", CheckPnrStatus),
    Tool("search_flights", "travel", "Search flights between two cities.", SearchFlights),
    Tool("book_cab", "travel", "Book a cab, auto or bike ride.", BookCab),
    Tool("search_restaurants", "food", "Find restaurants.", SearchRestaurants),
    Tool("order_food", "food", "Order food from a restaurant.", OrderFood),
    Tool("track_order", "food", "Track a food order.", TrackOrder),
    Tool("set_reminder", "productivity", "Create a reminder.", SetReminder),
    Tool("set_alarm", "productivity", "Set an alarm.", SetAlarm),
    Tool("create_calendar_event", "productivity", "Add a calendar event.", CreateCalendarEvent),
    Tool("get_weather", "utility", "Weather for a city.", GetWeather),
    Tool("convert_currency", "utility", "Convert between currencies.", ConvertCurrency),
]

TOOLS: dict[str, Tool] = {t.name: t for t in _TOOL_LIST}


class ToolCallError(ValueError):
    """Raised when a call names an unknown tool or has invalid arguments."""


def canonicalize(name: str, arguments: Any) -> dict[str, Any]:
    """Validate a call strictly and return its canonical arguments.

    Canonical form drops every argument equal to its default, so ``{"mode": "upi"}``
    and ``{}`` are the same transfer. Validation goes through JSON in strict mode,
    which rejects things like ``"amount": "2000"``.
    """
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolCallError(f"unknown tool: {name!r}")
    if not isinstance(arguments, dict):
        raise ToolCallError("arguments must be an object")
    try:
        model = tool.params.model_validate_json(json.dumps(arguments), strict=True)
    except ValidationError as exc:
        raise ToolCallError(str(exc)) from exc
    return model.model_dump(mode="json", exclude_defaults=True)


def field_policy(model: type[BaseModel], key: str) -> str:
    info = model.model_fields.get(key)
    extra = info.json_schema_extra if info is not None else None
    if isinstance(extra, dict):
        return str(extra.get("x-match", EXACT))
    return EXACT


def item_model(model: type[BaseModel], key: str) -> type[BaseModel] | None:
    """For a ``list[SomeModel]`` field (possibly optional), return ``SomeModel``."""
    info = model.model_fields.get(key)
    if info is None:
        return None
    ann = info.annotation
    if get_origin(ann) in (Union, types.UnionType):
        ann = next((a for a in get_args(ann) if a is not type(None)), ann)
    if get_origin(ann) is list:
        (inner,) = get_args(ann) or (None,)
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return inner
    return None


def clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make pydantic's JSON schema compact and readable for a small model.

    Inlines ``$ref``s, collapses ``anyOf[X, null]`` to ``X``, and drops titles,
    null defaults, regex patterns (descriptions already state formats),
    ``additionalProperties`` and our private ``x-`` keys. Validation still uses the
    full pydantic models, so nothing is lost on the checking side.
    """
    dropped = {"title", "$defs", "pattern", "additionalProperties"}
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = walk(defs[node["$ref"].rsplit("/", 1)[-1]])
            rest = walk({k: v for k, v in node.items() if k != "$ref"})
            return {**target, **rest}
        if "anyOf" in node:
            branches = [b for b in node["anyOf"] if b.get("type") != "null"]
            if len(branches) == 1:
                rest = {k: v for k, v in node.items() if k != "anyOf"}
                return walk({**branches[0], **rest})
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in dropped or k.startswith("x-"):
                continue
            if k == "default" and v is None:
                continue
            if k == "properties":
                out[k] = {pk: walk(pv) for pk, pv in v.items()}
            else:
                out[k] = walk(v)
        return out

    return walk(schema)
