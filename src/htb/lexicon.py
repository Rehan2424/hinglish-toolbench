"""Vocabulary shared by the scenario sampler and the evaluator.

Aliases matter for evaluation: "Bombay", "मुंबई" and "Mumbai" are the same city,
and a model should not lose points for using one over another.
"""

from __future__ import annotations

# canonical city -> aliases (colloquial, old names, Devanagari)
CITY_ALIASES: dict[str, list[str]] = {
    "Delhi": ["Dilli", "New Delhi", "दिल्ली", "नई दिल्ली"],
    "Mumbai": ["Bombay", "मुंबई", "बंबई"],
    "Bengaluru": ["Bangalore", "बेंगलुरु", "बैंगलोर"],
    "Kolkata": ["Calcutta", "कोलकाता"],
    "Chennai": ["Madras", "चेन्नई"],
    "Hyderabad": ["हैदराबाद"],
    "Pune": ["Poona", "पुणे"],
    "Ahmedabad": ["Amdavad", "अहमदाबाद"],
    "Jaipur": ["जयपुर"],
    "Lucknow": ["लखनऊ"],
    "Varanasi": ["Banaras", "Benares", "Kashi", "वाराणसी", "बनारस"],
    "Chandigarh": ["चंडीगढ़"],
    "Dehradun": ["देहरादून"],
    "Roorkee": ["रुड़की", "रूड़की"],
    "Haridwar": ["हरिद्वार"],
    "Patna": ["पटना"],
    "Bhopal": ["भोपाल"],
    "Indore": ["इंदौर"],
    "Amritsar": ["अमृतसर"],
    "Agra": ["आगरा"],
    "Kanpur": ["कानपुर"],
    "Gurugram": ["Gurgaon", "गुड़गांव", "गुरुग्राम"],
    "Noida": ["नोएडा"],
    "Shimla": ["शिमला"],
    "Goa": ["गोवा"],
    "Kochi": ["Cochin", "कोच्चि"],
    "Guwahati": ["गुवाहाटी"],
    "Nagpur": ["नागपुर"],
}

# Colloquial names a user might type instead of the official one.
CITY_COLLOQUIAL: dict[str, list[str]] = {
    "Delhi": ["Dilli"],
    "Mumbai": ["Bombay"],
    "Bengaluru": ["Bangalore"],
    "Kolkata": ["Calcutta"],
    "Varanasi": ["Banaras", "Kashi"],
    "Gurugram": ["Gurgaon"],
    "Chennai": ["Madras"],
}

CITIES: list[str] = list(CITY_ALIASES)

# Cities with an airport, for flight scenarios.
AIRPORT_CITIES: list[str] = [
    "Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai", "Hyderabad", "Pune", "Ahmedabad",
    "Jaipur", "Lucknow", "Varanasi", "Chandigarh", "Dehradun", "Patna", "Bhopal", "Indore",
    "Amritsar", "Goa", "Kochi", "Guwahati", "Nagpur",
]

# canonical person -> aliases
PEOPLE_ALIASES: dict[str, list[str]] = {
    "Rahul": ["राहुल"], "Priya": ["प्रिया"], "Amit": ["अमित"], "Neha": ["नेहा"],
    "Rohan": ["रोहन"], "Sneha": ["स्नेहा"], "Vikram": ["विक्रम"], "Anjali": ["अंजलि"],
    "Arjun": ["अर्जुन"], "Pooja": ["पूजा"], "Karan": ["करण"], "Simran": ["सिमरन"],
    "Aditya": ["आदित्य"], "Kavya": ["काव्या"], "Ishaan": ["ईशान"], "Meera": ["मीरा"],
    "Sharma ji": ["शर्मा जी", "Sharmaji"], "Gupta uncle": ["गुप्ता अंकल"],
    "Mummy": ["Mom", "Maa", "Mumma", "Mummy ji", "मम्मी", "माँ", "मां"],
    "Papa": ["Dad", "Papa ji", "Pitaji", "पापा", "पिताजी"],
    "Bhaiya": ["Bhai", "Bhaiyya", "भैया", "भाई"],
    "Didi": ["Di", "दीदी"],
}
PEOPLE: list[str] = list(PEOPLE_ALIASES)

# Generic places for cab rides.
LANDMARKS: list[str] = [
    "railway station", "airport", "bus stand", "City Mall", "civil hospital", "main market",
    "university gate", "Clock Tower", "metro station", "old city", "tech park", "stadium",
]

# Neighbourhoods for restaurant search.
AREAS: list[str] = [
    "Connaught Place", "Civil Lines", "MG Road", "Lajpat Nagar", "Koramangala", "Bandra",
    "Sector 29", "Hazratganj", "Park Street", "Model Town", "Rajpur Road", "Banjara Hills",
]

# Invented restaurant names.
RESTAURANTS: list[str] = [
    "Annapurna Bhojnalaya", "Spice Garden", "Tandoor Express", "Udupi Sagar", "Kathi Roll Corner",
    "Green Leaf Cafe", "Royal Biryani House", "Momo Junction", "Punjab Rasoi", "Chaat Chowk",
    "Coastal Curry Co", "Pind Da Dhaba", "Little Italy Kitchen", "Dragon Bowl",
]

DISHES: list[str] = [
    "paneer butter masala", "dal makhani", "butter naan", "masala dosa", "chicken biryani",
    "veg biryani", "chole bhature", "pav bhaji", "veg momos", "kadhai paneer", "rajma chawal",
    "aloo paratha", "gulab jamun", "cold coffee", "masala chai", "tandoori roti", "egg roll",
    "veg thali", "raita", "idli sambar", "chicken tikka", "hakka noodles", "margherita pizza",
]

# canonical cuisine -> ways a user might say it
CUISINES: dict[str, list[str]] = {
    "north indian": ["North Indian", "Punjabi khana"],
    "south indian": ["South Indian", "dosa-idli wala"],
    "chinese": ["Chinese"],
    "italian": ["Italian", "pizza-pasta"],
    "mughlai": ["Mughlai"],
    "street food": ["street food", "chaat"],
    "bengali": ["Bengali"],
    "continental": ["continental"],
    "desserts": ["dessert", "meetha"],
}

# (canonical English label, how a user might say it in Hinglish)
REMINDER_TITLES: list[tuple[str, str]] = [
    ("Call mummy", "mummy ko call karna"),
    ("Take medicine", "dawai leni hai"),
    ("Pay rent", "rent bharna"),
    ("Submit assignment", "assignment submit karna"),
    ("Water the plants", "paudhon ko paani dena"),
    ("Buy milk", "doodh lana"),
    ("Go to the gym", "gym jaana"),
    ("Book train tickets", "train ki ticket book karni"),
    ("Renew insurance", "insurance renew karna"),
    ("Wish Priya happy birthday", "Priya ko birthday wish karna"),
    ("Pick up laundry", "laundry uthani"),
    ("Pay credit card bill", "credit card ka bill bharna"),
    ("Meet the professor", "professor se milna"),
    ("Pack bags", "bag pack karna"),
    ("Recharge metro card", "metro card recharge karna"),
]

EVENT_TITLES: list[tuple[str, str]] = [
    ("Team sync", "team sync meeting"),
    ("Project review", "project review"),
    ("Doctor appointment", "doctor ka appointment"),
    ("Interview prep", "interview ki taiyari"),
    ("Lunch with Rohan", "Rohan ke saath lunch"),
    ("Family video call", "ghar walon ke saath video call"),
    ("Study group", "study group"),
    ("Client call", "client call"),
]

ALARM_LABELS: list[tuple[str, str]] = [
    ("Gym", "gym"), ("Wake up", "uthna hai"), ("Medicine", "dawai"),
    ("Office", "office"), ("Class", "class"), ("Morning walk", "morning walk"),
]

TRANSFER_NOTES: list[tuple[str, str]] = [
    ("rent", "kiraya"), ("dinner split", "dinner ka hissa"), ("birthday gift", "birthday gift"),
    ("electricity share", "bijli ka share"), ("loan return", "udhaar wapas"),
    ("trip expenses", "trip ka kharcha"),
]

BILLER_PHRASES: dict[str, list[str]] = {
    "electricity": ["bijli ka bill", "light ka bill", "electricity bill"],
    "mobile_postpaid": ["postpaid bill", "mobile ka bill"],
    "broadband": ["wifi ka bill", "broadband bill", "internet ka bill"],
    "gas": ["gas ka bill", "pipeline gas ka bill"],
    "water": ["paani ka bill", "water bill"],
    "dth": ["DTH recharge", "TV ka recharge", "dish TV ka bill"],
}

CATEGORY_PHRASES: dict[str, list[str]] = {
    "food": ["khane-peene", "food", "Zomato-type khana orders"],
    "groceries": ["grocery", "ration"],
    "shopping": ["shopping"],
    "travel": ["travel", "ghoomne-phirne"],
    "bills": ["bills"],
    "entertainment": ["entertainment", "movies-shows"],
}

CURRENCY_PHRASES: dict[str, list[str]] = {
    "INR": ["rupaye", "rupees", "INR"],
    "USD": ["dollar", "USD", "US dollar"],
    "EUR": ["euro"],
    "GBP": ["pound", "British pound"],
    "AED": ["dirham", "UAE dirham"],
    "JPY": ["yen", "Japanese yen"],
    "SGD": ["Singapore dollar"],
    "CAD": ["Canadian dollar"],
    "AUD": ["Australian dollar"],
}

# Accepted alternatives for free-text labels. Free text is scored on meaning: the
# canonical English label, common English paraphrases, and the user's own phrasing
# (the Hinglish form) all count. Same idea as multiple ground truths in BFCL.
_EXTRA_PARAPHRASES: dict[str, list[str]] = {
    "dinner split": ["dinner share", "share of dinner", "dinner bill"],
    "electricity share": ["electricity bill share", "share of electricity bill"],
    "loan return": ["returning loan", "repayment", "loan repayment", "money returned"],
    "trip expenses": ["trip cost", "trip share", "travel expenses"],
    "Take medicine": ["medicine", "take medicines", "medication"],
    "Go to the gym": ["gym"],
    "Buy milk": ["get milk", "milk"],
    "Water the plants": ["water plants"],
    "Pick up laundry": ["collect laundry", "laundry"],
    "Meet the professor": ["meeting with professor", "professor meeting"],
    "Pack bags": ["pack bag", "packing"],
    "Family video call": ["video call with family", "family call"],
    "Interview prep": ["interview preparation"],
    "Wake up": ["wake-up alarm", "alarm"],
    "Morning walk": ["walk"],
}


def _free_text_alternates() -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    for pairs in (REMINDER_TITLES, EVENT_TITLES, ALARM_LABELS, TRANSFER_NOTES):
        for label, said in pairs:
            table.setdefault(label, []).append(said)
    for label, extra in _EXTRA_PARAPHRASES.items():
        table.setdefault(label, []).extend(extra)
    return table


FREE_TEXT_ALTERNATES: dict[str, list[str]] = _free_text_alternates()

# (what the user asks, is_hard_negative). Hard negatives sound like a supported
# tool but are not: the assistant must say it cannot help instead of forcing a call.
OUT_OF_SCOPE: list[tuple[str, bool]] = [
    ("asks for a joke", False),
    ("asks how to reverse a list in Python", False),
    ("asks for the live cricket score", False),
    ("asks for a rajma chawal recipe", False),
    ("says thanks and goodbye", False),
    ("asks for tips to prepare for a job interview", False),
    ("asks to translate a Hindi sentence into English", False),
    ("asks for good movie recommendations", False),
    ("asks the meaning of a Hindi word", False),
    ("asks to book a hotel room in Goa for three nights", True),
    ("asks to book movie tickets for tonight", True),
    ("asks to cancel an already booked flight", True),
    ("asks to book (not search) a confirmed train ticket and pay for it", True),
    ("asks to block their lost debit card", True),
    ("asks to apply for a personal loan", True),
    ("asks to send a WhatsApp message to a friend", True),
    ("asks to order groceries like atta and milk from a supermarket", True),
    ("asks to delete an alarm they set earlier", True),
]
