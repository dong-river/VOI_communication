#!/usr/bin/env python3
"""
Build the cost-vs-accuracy comparison table from logged game transcripts.

For every per-question cost we report the best-tuned baselines and the
parameter-free VoI stopping rule (threshold == cost):

  * Confidence : stop once the running guess confidence >= threshold.
  * Round      : stop after a fixed number of questions.
  * VoI        : stop the first time the logged VoI estimate < cost.

The two tasks are combined as a single objective:
  Combined = 10 * Util_20Q + Util_Med,   with Reward(20Q)=1, Reward(Medical)=10
where Util = accuracy * reward - cost * (avg #questions).
We make 10 * Util_20Q to make the total reward balanced with Util_Med.

By default it reads the bundled transcripts in ``results/`` so the published
table reproduces exactly with no API calls.
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.realpath(__file__))

REWARD_20Q = 1
REWARD_MED = 10
COSTS = [0.01, 0.05, 0.1]
CONF_THRESHOLDS = [50, 70, 90]
ROUND_THRESHOLDS = [5, 8, 10, 15, 20]


# ------------------------------------------------------------------- loading
def load_games(folder):
    """Return a list of games; each game is a list of per-turn dicts."""
    games = []
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        with open(path) as f:
            turns = [json.loads(line) for line in f if line.strip()]
        if turns:
            games.append(turns)
    return games


# --------------------------------------------------- VoI offline stopping
def voi_raw_metrics(games, cost):
    """Stop the first time (after >=2 turns) the raw VoI estimate < cost."""
    acc = rounds = 0
    for turns in games:
        n = len(turns)
        stop = n - 1
        for i in range(n):
            u = turns[i].get("utility")
            if i > 1 and u is not None and u < cost:
                stop = i
                break
        acc += bool(turns[stop].get("success"))
        rounds += stop + 1
    n = len(games)
    return acc / n, rounds / n


# ----------------------------------------------------- baseline stopping
def baseline_metrics(games, stop_by, conf_threshold=None, max_round=None):
    """Confidence- or round-based stopping over baseline transcripts."""
    acc = rounds = 0
    used = 0
    for turns in games:
        selected = []
        for i, d in enumerate(turns):
            if stop_by == "round" and max_round is not None and i >= max_round:
                break
            selected.append(d)
            if stop_by == "conf":
                q = d.get("question", "").lower()
                if "my guess is" in q or (
                    conf_threshold is not None
                    and d.get("confidence", -float("inf")) >= conf_threshold
                ):
                    break
        if not selected:
            continue
        used += 1
        acc += bool(selected[-1].get("success"))
        rounds += len(selected)
    if used == 0:
        return 0.0, 0.0
    return acc / used, rounds / used


# ------------------------------------------------------------------ scoring
def util_20q(acc, rounds, cost):
    return REWARD_MED * (acc * REWARD_20Q - cost * rounds)


def util_med(acc, rounds, cost):
    return acc * REWARD_MED - cost * rounds


def main():
    p = argparse.ArgumentParser(description="VoI vs. Confidence/Round comparison table")
    p.add_argument("--voi_20q", default=os.path.join(HERE, "results/voi/20q"))
    p.add_argument("--voi_medical", default=os.path.join(HERE, "results/voi/medical"))
    p.add_argument("--conf_20q", default=os.path.join(HERE, "results/baselines/confidence_20q"))
    p.add_argument("--conf_medical", default=os.path.join(HERE, "results/baselines/confidence_medical"))
    args = p.parse_args()

    voi_20q = load_games(args.voi_20q)
    voi_med = load_games(args.voi_medical)
    conf_20q = load_games(args.conf_20q)
    conf_med = load_games(args.conf_medical)
    print(f"Loaded games -> VoI 20Q:{len(voi_20q)}  VoI Med:{len(voi_med)}  "
          f"Conf 20Q:{len(conf_20q)}  Conf Med:{len(conf_med)}")

    missing = [name for name, games, path in [
        ("VoI 20Q", voi_20q, args.voi_20q),
        ("VoI Medical", voi_med, args.voi_medical),
        ("Confidence 20Q", conf_20q, args.conf_20q),
        ("Confidence Medical", conf_med, args.conf_medical),
    ] if not games]
    if missing:
        raise SystemExit(
            "No transcripts found for: " + ", ".join(missing) + ".\n"
            "Generate them with run_voi.py / run_baseline.py, or point --voi_* / "
            "--conf_* at the right folders (each must contain *.jsonl games)."
        )

    W = 108
    for cost in COSTS:
        rows = []  # (method, label, a20, r20, am, rm)
        for thr in CONF_THRESHOLDS:
            a20, r20 = baseline_metrics(conf_20q, "conf", conf_threshold=thr)
            am, rm = baseline_metrics(conf_med, "conf", conf_threshold=thr)
            rows.append(("Confidence", f"conf>={thr}%", a20, r20, am, rm))
        for thr in ROUND_THRESHOLDS:
            a20, r20 = baseline_metrics(conf_20q, "round", max_round=thr)
            am, rm = baseline_metrics(conf_med, "round", max_round=thr)
            rows.append(("Round", f"max={thr}Q", a20, r20, am, rm))
        a20, r20 = voi_raw_metrics(voi_20q, cost)
        am, rm = voi_raw_metrics(voi_med, cost)
        rows.append(("VOI", f"voi>={cost}", a20, r20, am, rm))

        scored = []
        for m, lbl, a20, r20, am, rm in rows:
            u20 = util_20q(a20, r20, cost)
            um = util_med(am, rm, cost)
            scored.append((m, lbl, a20, r20, u20, am, rm, um, u20 + um))
        best_g = max(s[8] for s in scored)
        by_method = {}
        for s in scored:
            by_method.setdefault(s[0], []).append(s[8])
        best_m = {k: max(v) for k, v in by_method.items()}

        print(f"\n{'=' * W}")
        print(f"  Cost per question = {cost}   "
              f"(Reward: 20Q=1, Medical=10  |  Combined = 10xUtil_20Q + Util_Med)")
        print(f"{'=' * W}")
        print(f"  {'Method':<12} {'Threshold':<11} {'#Q 20Q':>7} {'Acc 20Q':>8} {'Util 20Q':>10}"
              f"  |  {'#Q Med':>7} {'Acc Med':>8} {'Util Med':>10}  |  {'Combined':>10}")
        print(f"  {'-'*12} {'-'*11} {'-'*7} {'-'*8} {'-'*10}  |  "
              f"{'-'*7} {'-'*8} {'-'*10}  |  {'-'*10}")
        prev = None
        for m, lbl, a20, r20, u20, am, rm, um, cb in scored:
            if prev and m != prev:
                print(f"  {'.'*12} {'.'*11} {'.'*7} {'.'*8} {'.'*10}  |  "
                      f"{'.'*7} {'.'*8} {'.'*10}  |  {'.'*10}")
            mark = " *" if cb == best_g else (" +" if cb == best_m[m] else "  ")
            print(f"  {m:<12} {lbl:<11} {r20:>7.2f} {a20:>8.3f} {u20:>10.3f}  |  "
                  f"{rm:>7.2f} {am:>8.3f} {um:>10.3f}  |  {cb:>8.3f}{mark}")
            prev = m

        br = max(scored, key=lambda s: s[8])
        print(f"\n  * Best overall : [{br[0]}] {br[1]}  ->  Combined={br[8]:.3f}")
        for method in ["Confidence", "Round", "VOI"]:
            cands = [s for s in scored if s[0] == method]
            if not cands:
                continue
            b = max(cands, key=lambda s: s[8])
            print(f"  + Best {method:<11}: [{b[1]}]  ->  Combined={b[8]:.3f}  "
                  f"(20Q acc={b[2]:.3f} #Q={b[3]:.1f} | Med acc={b[5]:.3f} #Q={b[6]:.1f})")
    print()


if __name__ == "__main__":
    main()
