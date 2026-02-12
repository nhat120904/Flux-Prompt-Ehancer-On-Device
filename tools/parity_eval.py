from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

from prompt_enhancer import EnhanceOptions, enhance_prompt, is_complete_sentence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parity evaluation against Python baseline.")
    parser.add_argument("--prompts", type=Path, required=True, help="Path to prompts JSONL.")
    parser.add_argument("--baseline-out", type=Path, required=True)
    parser.add_argument(
        "--candidate-out",
        type=Path,
        default=None,
        help="Optional candidate output JSONL from mobile runtime to compare.",
    )
    parser.add_argument("--model-id", default="imranali291/flux-prompt-enhancer")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        prompts.append(record["prompt"])
    return prompts


def run_baseline(prompts: list[str], model_id: str, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, prompt in enumerate(prompts):
        result = enhance_prompt(
            prompt,
            options=EnhanceOptions(seed=seed + idx),
            model_id=model_id,
        )
        rows.append({"prompt": prompt, **asdict(result)})
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compare_rows(
    baseline_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> dict[str, float]:
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("baseline and candidate outputs have different number of rows")

    similarities: list[float] = []
    complete_matches = 0
    for base, cand in zip(baseline_rows, candidate_rows, strict=True):
        btxt = str(base["text"])
        ctxt = str(cand["text"])
        similarities.append(SequenceMatcher(a=btxt, b=ctxt).ratio())
        if is_complete_sentence(ctxt):
            complete_matches += 1

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    completion_rate = complete_matches / len(similarities) if similarities else 0.0
    return {
        "count": float(len(similarities)),
        "avg_similarity": avg_similarity,
        "completion_rate": completion_rate,
    }


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    baseline_rows = run_baseline(prompts, model_id=args.model_id, seed=args.seed)
    write_jsonl(args.baseline_out, baseline_rows)
    print(f"Wrote baseline outputs to {args.baseline_out}")

    if args.candidate_out:
        candidate_rows = [json.loads(line) for line in args.candidate_out.read_text("utf-8").splitlines() if line]
        metrics = compare_rows(baseline_rows, candidate_rows)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
