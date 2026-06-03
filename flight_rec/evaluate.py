#!/usr/bin/env python3
"""
Build the flight-recommendation comparison table OFFLINE from per-turn transcripts
(same design as mixed_20q/evaluate.py).

Each game is rolled out once to ``max_questions`` by ``flight_rec.py --log_per_turn``
and every turn is logged. Stopping is decided here, post-hoc, so all methods/thresholds
share a single trajectory per game:

  * NoQuestion : commit at turn 0 (no questions).
  * Fixed (NQ) : commit after exactly NQ questions.
  * Confidence : commit once the belief confidence >= threshold.
  * VOI        : parameter-free -- ask the next question only while its logged
                 VOI >= cost, so the stopping threshold equals the per-question cost
                 and exactly ONE VOI row is shown per cost.

Utility = reward - cost * (#questions).

VOI rows come from the ``voi`` rollout; the NoQuestion/Fixed/Confidence rows come from
the ``confidence`` rollout (its questioner matches the baselines).
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.realpath(__file__))

COSTS = [0.01, 0.05, 0.1]
CONF_THRESHOLDS = [0.5, 0.7, 0.9]
FIXED_NQ = [1, 2, 3]


def load_games(folder):
    """Return a list of games; each game is a list of per-turn dicts (turn order)."""
    games = []
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        with open(path) as f:
            turns = [json.loads(line) for line in f if line.strip()]
        if turns:
            games.append(sorted(turns, key=lambda t: t["turn"]))
    return games


def _aggregate(stops):
    """stops: list of per-game chosen turn dicts -> (avg_q, avg_reward)."""
    n = len(stops)
    if n == 0:
        return 0.0, 0.0
    q = sum(s["questions_asked"] for s in stops) / n
    rew = sum(s["predicted_reward"] for s in stops) / n
    return q, rew


def voi_metrics(games, cost):
    """Ask the next question only while its logged VOI >= cost."""
    stops = []
    for turns in games:
        d = 0
        while d < len(turns) - 1 and turns[d]["voi_next"] is not None and turns[d]["voi_next"] >= cost:
            d += 1
        stops.append(turns[d])
    return _aggregate(stops)


def confidence_metrics(games, threshold):
    """Keep asking while belief confidence < threshold."""
    stops = []
    for turns in games:
        d = 0
        while d < len(turns) - 1 and turns[d]["confidence"] < threshold:
            d += 1
        stops.append(turns[d])
    return _aggregate(stops)


def fixed_metrics(games, nq):
    """Commit after exactly nq questions (clamped to the logged trajectory)."""
    stops = [turns[min(nq, len(turns) - 1)] for turns in games]
    return _aggregate(stops)


def noquestion_metrics(games):
    """Commit at turn 0."""
    return _aggregate([turns[0] for turns in games])


def main():
    p = argparse.ArgumentParser(description="Offline flight-rec comparison table from per-turn logs")
    p.add_argument("model", nargs="?", default="openai/gpt-5.4",
                   help="Model tag; transcripts are read from <root>/voi_<model>/ and "
                        "<root>/confidence_<model>/ (slashes become dashes).")
    p.add_argument("--root", default=os.path.join(HERE, "results"),
                   help="Root dir of per-turn transcripts (use results_reproduce for re-runs).")
    args = p.parse_args()

    safe = args.model.replace("/", "-")
    voi_dir = os.path.join(args.root, f"voi_{safe}")
    conf_dir = os.path.join(args.root, f"confidence_{safe}")
    voi_games = load_games(voi_dir)
    conf_games = load_games(conf_dir)
    print(f"Loaded transcripts -> VOI:{len(voi_games)} ({voi_dir})  "
          f"Baselines:{len(conf_games)} ({conf_dir})")
    if not voi_games and not conf_games:
        raise SystemExit("No per-turn transcripts found. Run flight_rec.py --log_per_turn first.")

    W = 66
    for cost in COSTS:
        rows = []  # (order, method, label, q, rew)
        if conf_games:
            q, rew = noquestion_metrics(conf_games)
            rows.append((0, "NoQuestion", "ask=0", q, rew))
            for ct in CONF_THRESHOLDS:
                q, rew = confidence_metrics(conf_games, ct)
                rows.append((1, "Confidence", f"conf>={ct}", q, rew))
            for nq in FIXED_NQ:
                q, rew = fixed_metrics(conf_games, nq)
                rows.append((2, "Fixed", f"max={nq}Q", q, rew))
        if voi_games:
            q, rew = voi_metrics(voi_games, cost)
            rows.append((3, "VOI", f"voi>={cost}", q, rew))

        scored = []
        for order, meth, lbl, q, rew in rows:
            util = rew - cost * q
            scored.append((order, meth, lbl, q, rew, util))
        if not scored:
            continue
        scored.sort(key=lambda s: (s[0], s[2]))
        best_g = max(s[5] for s in scored)
        bym = {}
        for s in scored:
            bym.setdefault(s[1], []).append(s[5])
        best_m = {k: max(v) for k, v in bym.items()}

        print(f"\n{'=' * W}")
        print(f"  Flight-Rec   model={args.model}   Cost per question = {cost}   (Utility = reward - cost*#Q)")
        print(f"{'=' * W}")
        print(f"  {'Method':<12} {'Threshold':<11} {'#Q':>6} {'Reward':>8} {'Utility':>9}")
        print(f"  {'-'*12} {'-'*11} {'-'*6} {'-'*8} {'-'*9}")
        prev = None
        for order, meth, lbl, q, rew, util in scored:
            if prev is not None and meth != prev:
                print(f"  {'.'*12} {'.'*11} {'.'*6} {'.'*8} {'.'*9}")
            mark = ' *' if util == best_g else (' +' if util == best_m[meth] else '  ')
            print(f"  {meth:<12} {lbl:<11} {q:>6.2f} {rew:>8.4f} {util:>9.4f}{mark}")
            prev = meth

        br = max(scored, key=lambda s: s[5])
        print(f"\n  * Best overall : [{br[1]}] {br[2]}  ->  Utility={br[5]:.4f}  (reward={br[4]:.4f} #Q={br[3]:.2f})")
        for meth in ["NoQuestion", "Confidence", "Fixed", "VOI"]:
            cand = [s for s in scored if s[1] == meth]
            if not cand:
                continue
            b = max(cand, key=lambda s: s[5])
            print(f"  + Best {meth:<11}: [{b[2]}]  ->  Utility={b[5]:.4f}  (reward={b[4]:.4f} #Q={b[3]:.2f})")
    print()


if __name__ == "__main__":
    main()
