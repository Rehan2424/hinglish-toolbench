"""Per-example scoring and aggregate reports.

Headline metric is **task accuracy**: the example counts as solved only if the
model takes the right action and, for tool calls, produces exactly the right set
of calls (order-insensitive, after removing default-valued arguments).

Diagnostics:

* parse rate         - output was a usable JSON response
* schema-valid rate  - every predicted call passed strict validation
* hallucination rate - a predicted call named a tool that does not exist
* tool selection     - predicted tool names equal gold names (as a multiset)
* argument P/R/F1    - micro-averaged over (tool, key, value) slots of aligned calls
* false-call rate    - on clarify/respond examples, the model called a tool anyway
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from ..schema import Example
from ..tools import TOOLS, ToolCallError, canonicalize
from .matching import align_calls, calls_equal, matching_keys, multiset_match
from .parse import parse_prediction


@dataclass
class ExampleScore:
    id: str
    category: str
    script: str
    gold_action: str
    parsed: bool
    pred_action: str | None
    action_correct: bool
    correct: bool
    schema_valid: bool | None = None
    hallucinated_tool: bool = False
    names_match: bool | None = None
    arg_tp: int = 0
    arg_fp: int = 0
    arg_fn: int = 0
    error: str | None = None


def _canonical_gold(example: Example) -> list[dict]:
    return [
        {"name": c.name, "arguments": canonicalize(c.name, c.arguments)}
        for c in example.expected.tool_calls
    ]


def score_example(example: Example, output: str) -> ExampleScore:
    gold_action = example.expected.action
    base = dict(id=example.id, category=example.category, script=example.script,
                gold_action=gold_action)
    pred = parse_prediction(output)
    if pred is None:
        gold_calls = _canonical_gold(example)
        fn = sum(len(c["arguments"]) for c in gold_calls)
        return ExampleScore(**base, parsed=False, pred_action=None, action_correct=False,
                            correct=False, arg_fn=fn, error="unparseable output")

    action_correct = pred.action == gold_action
    score = ExampleScore(**base, parsed=True, pred_action=pred.action,
                         action_correct=action_correct, correct=False)

    # Canonicalize predicted calls; keep raw args for partial credit when invalid.
    pred_calls: list[dict] = []
    all_valid = True
    errors = []
    for c in pred.tool_calls:
        if c["name"] not in TOOLS:
            score.hallucinated_tool = True
        try:
            args = canonicalize(c["name"], c["arguments"])
        except ToolCallError as exc:
            all_valid = False
            errors.append(str(exc).splitlines()[0])
            args = {k: v for k, v in c["arguments"].items() if v is not None}
        pred_calls.append({"name": c["name"], "arguments": args})
    if pred_calls:
        score.schema_valid = all_valid
    if errors:
        score.error = "; ".join(errors)

    gold_calls = _canonical_gold(example)
    if gold_action == "call":
        score.names_match = Counter(c["name"] for c in gold_calls) == Counter(
            c["name"] for c in pred_calls
        )
        pairs = align_calls(gold_calls, pred_calls)
        paired_g = {g for g, _ in pairs}
        paired_p = {p for _, p in pairs}
        for gi, pi in pairs:
            g, p = gold_calls[gi], pred_calls[pi]
            tp = matching_keys(g["name"], g["arguments"], p["arguments"])
            score.arg_tp += tp
            score.arg_fn += len(g["arguments"]) - tp
            score.arg_fp += len(p["arguments"]) - tp
        score.arg_fn += sum(len(gold_calls[i]["arguments"]) for i in range(len(gold_calls))
                            if i not in paired_g)
        score.arg_fp += sum(len(pred_calls[i]["arguments"]) for i in range(len(pred_calls))
                            if i not in paired_p)
        score.correct = (
            action_correct and all_valid and multiset_match(gold_calls, pred_calls, calls_equal)
        )
    else:
        score.arg_fp += sum(len(c["arguments"]) for c in pred_calls)
        score.correct = action_correct
    return score


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _summary(scores: list[ExampleScore]) -> dict:
    n = len(scores)
    with_calls = [s for s in scores if s.schema_valid is not None]
    call_gold = [s for s in scores if s.gold_action == "call"]
    no_call_gold = [s for s in scores if s.gold_action != "call"]
    tp = sum(s.arg_tp for s in scores)
    fp = sum(s.arg_fp for s in scores)
    fn = sum(s.arg_fn for s in scores)
    p, r = _rate(tp, tp + fp), _rate(tp, tp + fn)
    f1 = round(2 * p * r / (p + r), 4) if p and r else 0.0
    return {
        "n": n,
        "task_accuracy": _rate(sum(s.correct for s in scores), n),
        "parse_rate": _rate(sum(s.parsed for s in scores), n),
        "action_accuracy": _rate(sum(s.action_correct for s in scores), n),
        "schema_valid_rate": _rate(sum(bool(s.schema_valid) for s in with_calls), len(with_calls)),
        "hallucination_rate": _rate(sum(s.hallucinated_tool for s in with_calls), len(with_calls)),
        "tool_selection_accuracy": _rate(sum(bool(s.names_match) for s in call_gold),
                                         len(call_gold)),
        "arg_precision": p,
        "arg_recall": r,
        "arg_f1": f1,
        "false_call_rate": _rate(sum(s.pred_action == "call" for s in no_call_gold),
                                 len(no_call_gold)),
    }


def aggregate(scores: list[ExampleScore]) -> dict:
    by_cat: dict[str, list[ExampleScore]] = defaultdict(list)
    by_script: dict[str, list[ExampleScore]] = defaultdict(list)
    for s in scores:
        by_cat[s.category].append(s)
        by_script[s.script].append(s)
    return {
        "overall": _summary(scores),
        "by_category": {k: _summary(v) for k, v in sorted(by_cat.items())},
        "by_script": {k: _summary(v) for k, v in sorted(by_script.items())},
    }


def bootstrap_ci(flags: list[bool], *, n_boot: int = 2000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean of 0/1 outcomes."""
    n = len(flags)
    if not n:
        return (0.0, 0.0)
    rng = random.Random(seed)
    vals = [int(f) for f in flags]
    stats = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = stats[int((1 - level) / 2 * n_boot)]
    hi = stats[int((1 + level) / 2 * n_boot) - 1]
    return (round(lo, 4), round(hi, 4))


def evaluate(examples: list[Example], outputs: dict[str, str]) -> tuple[dict, list[dict]]:
    """Score every example. Missing predictions count as unparseable."""
    scores = [score_example(ex, outputs.get(ex.id, "")) for ex in examples]
    report = aggregate(scores)
    report["overall"]["task_accuracy_ci95"] = list(bootstrap_ci([s.correct for s in scores]))
    return report, [asdict(s) for s in scores]


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def to_markdown(report: dict, title: str = "Results") -> str:
    o = report["overall"]
    lines = [f"### {title}", "", "| metric | value |", "|---|---|"]
    for k in ("task_accuracy", "parse_rate", "action_accuracy", "schema_valid_rate",
              "hallucination_rate", "tool_selection_accuracy", "arg_precision", "arg_recall",
              "arg_f1", "false_call_rate"):
        lines.append(f"| {k} | {_fmt(o[k])} |")
    if "task_accuracy_ci95" in o:
        lo, hi = o["task_accuracy_ci95"]
        lines.insert(5, f"| task_accuracy 95% CI | {_fmt(lo)} to {_fmt(hi)} |")
    if "latency_s" in report:
        lat = report["latency_s"]
        lines.append(f"| latency median / p90 (s) | {lat['median']:.2f} / {lat['p90']:.2f} |")
    for group in ("by_category", "by_script"):
        lines += ["", f"| {group.split('_')[1]} | n | task acc | arg F1 |", "|---|---|---|---|"]
        for name, s in report[group].items():
            lines.append(f"| {name} | {s['n']} | {_fmt(s['task_accuracy'])} | {_fmt(s['arg_f1'])} |")
    return "\n".join(lines)
