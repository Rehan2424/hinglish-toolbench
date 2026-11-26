import json

import pytest

from htb.prompting import tools_block
from htb.tools import (
    FUZZY,
    TOOLS,
    OrderFood,
    SetReminder,
    ToolCallError,
    canonicalize,
    field_policy,
    item_model,
)


def test_registry_size_and_domains():
    assert len(TOOLS) == 17
    assert {t.domain for t in TOOLS.values()} == {"banking", "travel", "food", "productivity",
                                                  "utility"}


def test_defaults_are_dropped_in_canonical_form():
    a = canonicalize("transfer_money", {"recipient": "Rahul", "amount": 2000, "mode": "upi"})
    b = canonicalize("transfer_money", {"recipient": "Rahul", "amount": 2000.0})
    assert a == b == {"recipient": "Rahul", "amount": 2000.0}


def test_nested_defaults_are_dropped():
    out = canonicalize("order_food", {"restaurant": "X", "items": [{"name": "raita", "quantity": 1}]})
    assert out == {"restaurant": "X", "items": [{"name": "raita"}]}


@pytest.mark.parametrize("name,args", [
    ("transfer_money", {"recipient": "R", "amount": "2000"}),  # string number
    ("transfer_money", {"recipient": "R", "amount": 5, "upi_id": "x"}),  # unknown key
    ("transfer_money", {"recipient": "R", "amount": -5}),  # negative
    ("search_trains", {"source": "A", "destination": "B", "date": "27-11-2026"}),  # bad date
    ("set_alarm", {"time": "7:30"}),  # not HH:MM
    ("recharge_mobile", {"phone_number": "12345", "amount": 199}),  # bad phone
    ("pay_bill", {"biller": "netflix"}),  # not an enum value
    ("not_a_tool", {}),
])
def test_invalid_calls_are_rejected(name, args):
    with pytest.raises(ToolCallError):
        canonicalize(name, args)


def test_policy_lookup():
    assert field_policy(SetReminder, "title") == FUZZY
    assert item_model(OrderFood, "items").__name__ == "OrderItem"
    assert item_model(SetReminder, "title") is None


def test_prompt_schema_is_compact_and_complete():
    block = tools_block()
    lines = block.splitlines()
    assert len(lines) == len(TOOLS)
    for line in lines:
        schema = json.loads(line)
        assert "x-match" not in line and "$ref" not in line and '"title":"' not in line
        assert schema["parameters"]["type"] == "object"
    # the 'title' *field* must survive cleaning even though 'title' keys are dropped
    reminder = json.loads(lines[list(TOOLS).index("set_reminder")])
    assert "title" in reminder["parameters"]["properties"]
