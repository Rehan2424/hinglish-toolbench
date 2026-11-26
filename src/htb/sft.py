"""Render examples as chat-format supervised fine-tuning data.

Each row is ``{"id", "messages": [system, user, (assistant, user)..., assistant]}``
where the final assistant turn is the target in the exact output format. The
training script only needs this file, not the htb package.

``message`` text for clarify/respond targets is not scored by the benchmark, but the
model still has to say something sensible, so we template it in the user's script.
"""

from __future__ import annotations

import random

from .prompting import agent_messages, render_target
from .schema import Example

CLARIFY_QUESTIONS: dict[str, dict[str, str]] = {
    "amount": {"roman": "Kitne paise?", "devanagari": "कितने पैसे?",
               "english": "How much should I send?"},
    "recipient": {"roman": "Kisko bhejna hai?", "devanagari": "किसको भेजना है?",
                  "english": "Who should I send it to?"},
    "phone_number": {"roman": "Kis number pe recharge karna hai?",
                     "devanagari": "किस नंबर पर रिचार्ज करना है?",
                     "english": "Which number should I recharge?"},
    "destination": {"roman": "Kahan jaana hai?", "devanagari": "कहाँ जाना है?",
                    "english": "Where are you travelling to?"},
    "date": {"roman": "Kis din ke liye?", "devanagari": "किस दिन के लिए?",
             "english": "For which date?"},
    "drop": {"roman": "Kahan jaana hai?", "devanagari": "कहाँ जाना है?",
             "english": "Where do you want to go?"},
    "restaurant": {"roman": "Kis restaurant se mangwana hai?",
                   "devanagari": "किस रेस्टोरेंट से मंगवाना है?",
                   "english": "Which restaurant should I order from?"},
    "items": {"roman": "Kya order karna hai?", "devanagari": "क्या ऑर्डर करना है?",
              "english": "What would you like to order?"},
    "pnr": {"roman": "PNR number batayein?", "devanagari": "PNR नंबर बताइए?",
            "english": "What is the PNR number?"},
    "remind_at": {"roman": "Kab yaad dilana hai?", "devanagari": "कब याद दिलाना है?",
                  "english": "When should I remind you?"},
    "time": {"roman": "Alarm kitne baje ka lagana hai?",
             "devanagari": "अलार्म कितने बजे का लगाना है?",
             "english": "What time should the alarm be?"},
    "start": {"roman": "Kab rakhna hai?", "devanagari": "कब रखना है?",
              "english": "When is it?"},
}
GENERIC_CLARIFY = {"roman": "Thoda aur detail batayenge?", "devanagari": "थोड़ा और बताइए?",
                   "english": "Could you share a bit more detail?"}
DECLINES = {
    "roman": ["Sorry, isme main madad nahi kar sakta. Main payments, bills, travel, khana "
              "order, reminders aur weather mein help kar sakta hoon.",
              "Yeh abhi mere paas available nahi hai, sorry."],
    "devanagari": ["माफ़ कीजिए, इसमें मैं मदद नहीं कर सकता। मैं पेमेंट, बिल, ट्रैवल, खाना ऑर्डर, "
                   "रिमाइंडर और मौसम में मदद कर सकता हूँ।"],
    "english": ["Sorry, I can't help with that. I can handle payments, bills, travel, food "
                "orders, reminders and weather.",
                "That isn't something I can do yet, sorry."],
}


def target_message(ex: Example, rng: random.Random) -> str | None:
    if ex.expected.action == "clarify":
        q = CLARIFY_QUESTIONS.get(ex.meta.get("missing", ""), GENERIC_CLARIFY)
        return q[ex.script]
    if ex.expected.action == "respond":
        return rng.choice(DECLINES[ex.script])
    return None


def to_sft(ex: Example, rng: random.Random) -> dict:
    target = render_target(ex.expected, target_message(ex, rng))
    return {"id": ex.id, "messages": agent_messages(ex) + [{"role": "assistant",
                                                           "content": target}]}
