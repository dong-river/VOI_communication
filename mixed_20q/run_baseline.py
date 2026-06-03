#!/usr/bin/env python3
"""
Generate baseline game transcripts (one JSONL file per game).

These are plain question-asking games that log a forced guess + confidence each
turn. Both the **Confidence** and **Round** baselines are derived from these
transcripts; the stopping rule is applied offline by ``evaluate.py``.

Examples (200 games each, OpenRouter / gpt-5.4):

  export OPENROUTER_API_KEY=sk-or-...
  python run_baseline.py --domain 20q     --num_games 200 --out_dir runs/baseline_20q
  python run_baseline.py --domain medical --num_games 200 --out_dir runs/baseline_medical \\
      --medical_data data/MedDG.json
"""
import argparse
import os
import random

from baseline_strategy import ConfidenceBaselineStrategy
from environment import TwentyQuestionsEnvironment
from utils import load_json_data


def parse_args():
    p = argparse.ArgumentParser(description="Confidence/Round baseline transcript generator")
    p.add_argument("--domain", choices=["20q", "medical"], default="20q")
    p.add_argument("--num_games", type=int, default=200)
    p.add_argument("--start_exp_id", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="runs/baseline_20q",
                   help="Directory to write per-game JSONL transcripts into.")
    p.add_argument("--model", type=str, default="openai/gpt-5.4")
    p.add_argument("--provider", type=str, default="openrouter", choices=["openrouter", "openai"])
    p.add_argument("--question_type", type=str, default="yes_no")
    p.add_argument("--max_questions", type=int, default=None,
                   help="Default: 20 for 20q, 10 for medical.")
    p.add_argument("--candidate_set_size", type=int, default=None,
                   help="Default: 100 for 20q, 15 for medical.")
    p.add_argument("--medical_data", type=str, default="data/MedDG.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.max_questions is None:
        args.max_questions = 20 if args.domain == "20q" else 10
    if args.candidate_set_size is None:
        args.candidate_set_size = 100 if args.domain == "20q" else 15
    args.animal_set = "animal_set" if args.domain == "20q" else "default"
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = None
    if args.domain == "medical":
        data = load_json_data(args.medical_data)
        assert data is not None, f"Failed to load medical data from {args.medical_data}"
        random.seed(0)
        random.shuffle(data)

    n_correct = 0
    for i, game_id in enumerate(range(args.start_exp_id, args.start_exp_id + args.num_games)):
        print(f"\n=== Game {i + 1}/{args.num_games} (id={game_id}) ===")
        env = TwentyQuestionsEnvironment(
            max_questions=args.max_questions,
            model=args.model,
            verbose=args.verbose,
            question_type=args.question_type,
            domain=args.domain,
            exp_id=game_id,
            data=data,
            provider=args.provider,
            log_dir=args.out_dir,
            run_id=0,
            args=args,
        )
        strategy = ConfidenceBaselineStrategy(env)
        result = strategy.play_game()
        n_correct += bool(result and result.get("success"))
        print(f"  subject={result['subject']} success={result['success']} "
              f"#Q={result['question_count']}")

    print(f"\nDONE: {args.num_games} games | final-guess accuracy "
          f"(at max questions) = {n_correct / args.num_games:.3f}")
    print(f"Transcripts written under: {env.log_subdir}")


if __name__ == "__main__":
    main()
