from __future__ import annotations

import argparse

from prompt_enhancer import DEFAULT_MODEL_ID, EnhanceOptions, enhance_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local baseline for flux-prompt-enhancer.")
    parser.add_argument("prompt", help="Input prompt to enhance.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--retries", default=3, type=int)
    parser.add_argument("--top-p", default=0.92, type=float)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--repetition-penalty", default=1.2, type=float)
    parser.add_argument("--no-repeat-ngram-size", default=3, type=int)
    parser.add_argument("--min-new-tokens", default=60, type=int)
    parser.add_argument("--max-new-tokens", default=180, type=int)
    parser.add_argument("--timeout-ms", default=8_000, type=int)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = EnhanceOptions(
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        retries=args.retries,
        seed=args.seed,
        timeout_ms=args.timeout_ms,
    )
    result = enhance_prompt(args.prompt, options=options, model_id=args.model_id)
    print(result.text)


if __name__ == "__main__":
    main()
