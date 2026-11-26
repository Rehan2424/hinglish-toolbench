# Hinglish ToolBench

**Can a 1B-parameter model, running offline on a laptop CPU, turn Hinglish requests into correct API calls?**

Most Indian users don't type in clean English. They write things like:

> kal subah 7 baje ke baad Delhi se Jaipur ki train dikhao, 3AC chahiye

An assistant has to turn that into a structured call:

```json
{"name": "search_trains",
 "arguments": {"source": "Delhi", "destination": "Jaipur", "date": "2026-11-27",
               "travel_class": "3A", "depart_after": "07:00"}}
```

Large hosted models do this reasonably well, but they are expensive per request, need a network connection, and send user data to a third party. Small open models are cheap and private, but they struggle with code-mixed input: *kal* means tomorrow or yesterday depending on context, *saade 7* is 7:30, *dedh hazaar* is 1,500, and the same sentence can arrive in Latin or Devanagari script.

This project has three parts:

1. **A benchmark** for Hinglish tool calling: 17 tools across banking, travel, food, productivity and utilities; five task types; a human-verified test set.
2. **A label-first synthetic data pipeline** that produces training data with correct-by-construction labels.
3. **A distilled small model**, fine-tuned with LoRA and quantized for CPU inference, compared against hosted APIs on accuracy, latency and cost.

> **Status:** benchmark design, data pipeline and evaluation harness are complete and tested. Data generation, fine-tuning and results are in progress (see [Roadmap](#roadmap)).

---

## Task

The model receives a system prompt (current date and time, the user's city, tool schemas) and a conversation. It must reply with one JSON object:

| action | when | example |
|---|---|---|
| `call` | the request maps to one or more tools | *"balance batao aur Priya ko 500 bhej do"* → two calls |
| `clarify` | a required argument is missing | *"Amit ko paise bhejne hain"* → ask how much |
| `respond` | no tool fits | *"Goa mein hotel book kar do"* → there is no hotel tool |

Examples come in five categories: `single`, `parallel` (several calls in one message), `multi_turn` (the missing detail arrives after a follow-up question), `clarify`, and `irrelevant`. Irrelevant requests include *hard negatives* that sound like a supported tool but are not, such as cancelling a flight when only flight search exists.

Inputs come in three scripts: romanized Hinglish, Devanagari, and Indian English.

## Data pipeline

```mermaid
flowchart LR
    A[Sample a valid call in code<br/>+ Hinglish phrasing hints] --> B[Teacher LLM writes<br/>the user message]
    B --> C[Lint: script, leaks,<br/>ids present]
    C --> D[Round-trip check:<br/>second model recovers<br/>the label blind]
    D -->|match| E[(train / dev)]
    D -->|mismatch| F[(rejects, for analysis)]
    E --> G[Test candidates<br/>hand-verified]
```

**Label first, text second.** Most synthetic tool-calling datasets ask an LLM to write a request and its label together, so label errors are baked in. Here the call is sampled in code from typed tool definitions, so every label is valid by construction. The teacher model only writes the user's message, following phrasing hints such as `Date: "parso"` or `Amount: "dedh hazaar"`.

**Round-trip verification.** A second model that never sees the label must recover it from the message alone. If it can't, the message is ambiguous or unfaithful, and the example is dropped. Rejected examples are kept for error analysis.

**Hinglish coverage by design.** The sampler controls which surface forms appear, so the data reliably covers the hard cases:

| phenomenon | examples |
|---|---|
| relative dates | *aaj, kal, parso, is Friday, shukravaar ko, agle hafte Monday* |
| *kal* ambiguity | tomorrow for plans, yesterday for past transactions |
| spoken times | *shaam saade 7* (19:30), *sava 8* (8:15), *paune 9* (8:45), *dopahar dedh baje* (13:30) |
| spoken amounts | *dedh hazaar* (1,500), *dhai sau* (250), *ek lakh*, *2k*, *₹15,00,000* |
| implicit context | *"yahan ka mausam"* → user's city; no departure city → user's city |
| aliases | *Bombay, Dilli, Banaras, Gurgaon*, Devanagari city names |
| next occurrence | *"roz subah 8 baje"* at 10:00 → tomorrow 08:00, daily |

## Evaluation

The headline metric is **task accuracy**: an example counts as solved only if the action is right and, for calls, the set of calls matches exactly. Order doesn't matter, and arguments equal to their defaults are ignored. Diagnostics break failures down further:

| metric | what it catches |
|---|---|
| parse rate | output is not usable JSON |
| schema-valid rate | wrong types, unknown keys, bad date formats (strict validation) |
| hallucination rate | calls to tools that don't exist |
| tool selection accuracy | right intent, regardless of arguments |
| argument precision / recall / F1 | partial credit at the slot level |
| false-call rate | calling a tool when it should clarify or decline |

Each argument is compared according to a policy declared in its tool definition:

- **exact** for enums, numbers, dates and ids.
- **entity** for cities and people, ignoring case and resolving aliases, so *Bombay* = *Mumbai* = *मुंबई*.
- **fuzzy** for free text: F1 over content words with function words removed, so *"Call mummy"* matches *"Mummy ko call"* but not *"Call papa"*.

Results are reported overall, by category, and by script.

## Results

*Coming soon:* hosted API baselines, zero-shot and few-shot small models, and the fine-tuned model (full precision and 4-bit), with CPU latency and cost per 1,000 requests.

## Quickstart

```bash
pip install -e ".[dev]"
pytest

htb tools                                   # print the 17 tool schemas
htb scenarios --n 5000 --seed 0 --out data/scenarios.jsonl
htb eval --examples data/seed/examples.jsonl --oracle   # sanity check: 100%
```

Generation and verification work with any OpenAI-compatible endpoint: Gemini, Groq, OpenRouter, OpenAI, or a local llama.cpp, vLLM or Ollama server. Put keys in `.env` (see `.env.example`):

```bash
htb generate --scenarios data/scenarios.jsonl --out data/generated.jsonl \
    --rejects data/rejects_gen.jsonl --provider gemini --model gemini-2.5-flash --rpm 10
htb verify --examples data/generated.jsonl --out data/verified.jsonl \
    --rejects data/rejects_verify.jsonl --provider groq --model <model>
htb predict --examples data/test.jsonl --out preds/<name>.jsonl --provider <p> --model <m>
htb eval --examples data/test.jsonl --predictions preds/<name>.jsonl --report reports/<name>.json
```

Every stage is resumable. Rerunning a command skips examples it has already processed, which matters on rate-limited free tiers.

## Repository layout

```
src/htb/
  tools.py        17 tool definitions (pydantic): schema, strict validation, match policies
  prompting.py    the system prompt and output format, shared by every stage
  lexicon.py      cities, aliases, names, dishes, out-of-scope requests
  gen/
    hinglish.py   spoken forms of dates, times, amounts and counts
    sampler.py    label-first scenario sampling for all five categories
    prompts.py    teacher prompt
    pipeline.py   generate -> lint -> verify, resumable and parallel
  eval/
    parse.py      robust JSON extraction from model output
    matching.py   per-field matching, call alignment
    metrics.py    scoring and reports
  llm.py          OpenAI-compatible client with retries and rate limiting
  cli.py          `htb` command
data/seed/        hand-written examples (style references and test fixtures)
tests/            unit and pipeline tests
```

## Roadmap

- [x] Tool definitions, strict validation, matching policies
- [x] Label-first scenario sampler with Hinglish surface forms
- [x] Generation, lint and round-trip verification pipeline
- [x] Evaluation harness and reports
- [ ] Generate and verify the training set
- [ ] Hand-verify the test set; publish the dataset with a datasheet
- [ ] Hosted API and zero-shot baselines
- [ ] LoRA fine-tuning of 0.5B and 1.5B models (ablations: data size, filtering, script, constrained decoding)
- [ ] 4-bit quantization, CPU latency benchmark, cost comparison
- [ ] FastAPI service, MCP server, Docker image, live demo

## License

Code is MIT-licensed. The dataset will be released under CC BY 4.0.
