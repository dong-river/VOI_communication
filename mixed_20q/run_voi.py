#!/usr/bin/env python3
"""
Generate VoI game transcripts (one JSONL file per game).

Each turn logs the per-turn VoI estimate under the ``utility`` key; the decision
of *when to stop* is made offline by ``evaluate.py``.

Examples (200 games each, OpenRouter / gpt-5.4):

  export OPENROUTER_API_KEY=sk-or-...
  python run_voi.py --domain 20q     --num_games 200 --out_dir runs/voi_20q
  python run_voi.py --domain medical --num_games 200 --out_dir runs/voi_medical \\
      --medical_data data/MedDG.json
"""
import argparse
import os
import random

from environment import TwentyQuestionsEnvironment
from utils import load_json_data
from voi_strategy import ValueOfInformationStrategy


def parse_args():
    p = argparse.ArgumentParser(description="VoI full-set + best-of-N transcript generator")
    p.add_argument("--domain", choices=["20q", "medical"], default="20q")
    p.add_argument("--num_games", type=int, default=200)
    p.add_argument("--start_exp_id", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="runs/voi_20q",
                   help="Directory to write per-game JSONL transcripts into.")
    # Model / provider.
    p.add_argument("--model", type=str, default="openai/gpt-5.4")
    p.add_argument("--provider", type=str, default="openrouter", choices=["openrouter", "openai"])
    # Game configuration.
    p.add_argument("--question_type", type=str, default="yes_no")
    p.add_argument("--max_questions", type=int, default=None,
                   help="Default: 20 for 20q, 10 for medical.")
    p.add_argument("--candidate_set_size", type=int, default=None,
                   help="Default: 100 for 20q, 15 for medical.")
    # Strategy hyper-parameters.
    p.add_argument("--top_question_candidates", type=int, default=3)
    p.add_argument("--lookahead_k", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--communication_cost_base", type=float, default=0.0)
    p.add_argument("--medical_data", type=str, default="data/MedDG.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Domain-specific defaults.
    if args.domain == "20q" 
        args.max_questions = 20 
    else:
        args.max_questions = 10
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
        strategy = ValueOfInformationStrategy(
            env,
            batch_size=args.batch_size,
            communication_cost_base=args.communication_cost_base,
            lookahead_k=args.lookahead_k,
            top_question_candidates=args.top_question_candidates,
        )
        result = strategy.play_game()
        n_correct += bool(result and result.get("success"))
        print(f"  subject={result['subject']} success={result['success']} "
              f"#Q={result['question_count']}")

    print(f"\nDONE: {args.num_games} games | final-guess accuracy "
          f"(at max questions) = {n_correct / args.num_games:.3f}")
    print(f"Transcripts written under: {env.log_subdir}")


if __name__ == "__main__":
    main()
