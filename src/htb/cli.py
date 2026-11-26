"""Command line entry point: ``htb <command>``."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from .io import read_jsonl, write_jsonl
from .schema import Example


def _client(args: argparse.Namespace):
    from .llm import ChatClient, load_dotenv

    load_dotenv()
    return ChatClient(args.provider, args.model, base_url=args.base_url,
                      requests_per_minute=args.rpm,
                      extra_body=json.loads(args.extra) if args.extra else None)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", required=True, help="gemini | groq | openrouter | openai | local")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default=None, help="override the provider URL")
    p.add_argument("--rpm", type=float, default=None, help="requests-per-minute cap")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--extra", default=None,
                   help='extra JSON fields for the request, e.g. \'{"reasoning_effort": "low"}\'')


def _load_examples(path: str) -> list[Example]:
    return [Example.model_validate(row) for row in read_jsonl(path)]


def _style_examples(path: str | None) -> dict[str, list[str]]:
    if not path or not Path(path).exists():
        return {}
    by_script: dict[str, list[str]] = {}
    for ex in _load_examples(path):
        by_script.setdefault(ex.script, []).append(ex.messages[0].content)
    rng = random.Random(0)
    return {k: rng.sample(v, min(4, len(v))) for k, v in by_script.items()}


def cmd_tools(args: argparse.Namespace) -> None:
    from .tools import TOOLS

    for t in TOOLS.values():
        print(json.dumps(t.json_schema(), ensure_ascii=False))


def cmd_scenarios(args: argparse.Namespace) -> None:
    from .gen.sampler import build_scenarios

    scenarios = build_scenarios(args.n, args.seed, prefix=args.prefix)
    n = write_jsonl(args.out, (s.model_dump() for s in scenarios))
    print(f"wrote {n} scenarios to {args.out}")
    print(dict(Counter(s.category for s in scenarios)))


def cmd_generate(args: argparse.Namespace) -> None:
    from .gen.pipeline import generate
    from .gen.sampler import Scenario

    scenarios = [Scenario.model_validate(r) for r in read_jsonl(args.scenarios)]
    if args.limit:
        scenarios = scenarios[: args.limit]
    stats = generate(scenarios, _client(args), args.out, args.rejects, workers=args.workers,
                     temperature=args.temperature,
                     style_examples=_style_examples(args.style_examples),
                     teacher_name=f"{args.provider}/{args.model}")
    print(dict(stats))


def cmd_verify(args: argparse.Namespace) -> None:
    from .gen.pipeline import verify

    verifiers = [(f"{args.provider}/{args.model}", _client(args))]
    if args.fallback_model:
        fb = argparse.Namespace(**vars(args))
        fb.provider = args.fallback_provider or args.provider
        fb.model = args.fallback_model
        fb.extra = args.fallback_extra
        verifiers.append((f"{fb.provider}/{fb.model}", _client(fb)))
    stats = verify(_load_examples(args.examples), verifiers, args.out, args.rejects,
                   workers=args.workers)
    print(dict(stats))


def cmd_predict(args: argparse.Namespace) -> None:
    from .gen.pipeline import predict

    stats = predict(_load_examples(args.examples), _client(args), args.out, workers=args.workers)
    print(dict(stats))


def cmd_eval(args: argparse.Namespace) -> None:
    from .eval.metrics import evaluate, to_markdown

    examples = _load_examples(args.examples)
    if args.oracle:
        from .prompting import render_target

        outputs = {e.id: render_target(e.expected) for e in examples}
    else:
        outputs = {r["id"]: r["output"] for r in read_jsonl(args.predictions)}
    report, per_example = evaluate(examples, outputs)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.details:
        write_jsonl(args.details, per_example)
    print(to_markdown(report, title=args.title or Path(args.predictions or "oracle").stem))


def cmd_stats(args: argparse.Namespace) -> None:
    examples = _load_examples(args.examples)
    print(f"{len(examples)} examples")
    for key in ("category", "script"):
        print(key, dict(Counter(getattr(e, key) for e in examples)))
    tools = Counter(c.name for e in examples for c in e.expected.tool_calls)
    print("tools", dict(tools.most_common()))


def cmd_split(args: argparse.Namespace) -> None:
    from .splits import dedupe, dedupe_key, split_train_dev

    pool = [ex for path in args.examples for ex in _load_examples(path)]
    exclude = {dedupe_key(e) for p in (args.exclude or []) for e in _load_examples(p)}
    unique = dedupe(pool, exclude)
    train, dev = split_train_dev(unique, args.dev_frac, args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    write_jsonl(Path(args.out_dir) / "train.jsonl", (e.model_dump() for e in train))
    write_jsonl(Path(args.out_dir) / "dev.jsonl", (e.model_dump() for e in dev))
    print(f"{len(pool)} in, {len(pool) - len(unique)} duplicates/overlaps dropped, "
          f"train {len(train)}, dev {len(dev)}")


def cmd_review(args: argparse.Namespace) -> None:
    from .splits import dedupe, render_review_html

    examples = dedupe([ex for path in args.examples for ex in _load_examples(path)])
    if args.limit:
        examples = examples[: args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render_review_html(examples, args.title), encoding="utf-8")
    write_jsonl(Path(args.out).with_suffix(".candidates.jsonl"), (e.model_dump() for e in examples))
    print(f"review page with {len(examples)} examples: {args.out}")


def cmd_apply_review(args: argparse.Namespace) -> None:
    from .splits import apply_review

    candidates = _load_examples(args.candidates)
    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    accepted, stats = apply_review(candidates, decisions)
    write_jsonl(args.out, (e.model_dump() for e in accepted))
    print(stats)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="htb", description="Hinglish tool-calling benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tools", help="print tool schemas")
    p.set_defaults(fn=cmd_tools)

    p = sub.add_parser("scenarios", help="sample labelled scenarios (no LLM needed)")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="s")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_scenarios)

    p = sub.add_parser("generate", help="teacher model phrases scenarios as user messages")
    p.add_argument("--scenarios", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rejects", required=True)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--style-examples", default="data/seed/examples.jsonl")
    _add_model_args(p)
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("verify", help="round-trip check with a second model")
    p.add_argument("--examples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rejects", required=True)
    _add_model_args(p)
    p.add_argument("--fallback-model", help="second verifier, tried only on mismatches")
    p.add_argument("--fallback-provider", help="defaults to --provider")
    p.add_argument("--fallback-extra", default=None)
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("predict", help="run a model on examples")
    p.add_argument("--examples", required=True)
    p.add_argument("--out", required=True)
    _add_model_args(p)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("eval", help="score predictions")
    p.add_argument("--examples", required=True)
    p.add_argument("--predictions")
    p.add_argument("--oracle", action="store_true", help="score the gold answers (sanity check)")
    p.add_argument("--report", help="write the JSON report here")
    p.add_argument("--details", help="write per-example scores (JSONL) here")
    p.add_argument("--title")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("split", help="dedupe and split verified data into train/dev")
    p.add_argument("--examples", nargs="+", required=True)
    p.add_argument("--exclude", nargs="*", help="files whose messages must not appear (test)")
    p.add_argument("--dev-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_split)

    p = sub.add_parser("review", help="build an offline HTML page for hand-checking examples")
    p.add_argument("--examples", nargs="+", required=True)
    p.add_argument("--out", required=True, help="HTML file to write")
    p.add_argument("--title", default="Hinglish ToolBench: test set review")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("apply-review", help="build the test set from exported review decisions")
    p.add_argument("--candidates", required=True)
    p.add_argument("--decisions", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_apply_review)

    p = sub.add_parser("stats", help="dataset composition")
    p.add_argument("--examples", required=True)
    p.set_defaults(fn=cmd_stats)

    args = parser.parse_args(argv)
    if args.command == "eval" and not (args.oracle or args.predictions):
        parser.error("eval needs --predictions or --oracle")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args.fn(args)


if __name__ == "__main__":
    main()
