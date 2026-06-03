# Flight Recommendation — Value-of-Information (VOI) Clarification

An LLM assistant must recommend one of three flights to a user whose preferences
are hidden. Before committing, it may ask **clarifying questions**. The question
is *when to ask vs. when to commit*. We compare a **Value-of-Information (VOI)**
policy against standard baselines under a per-question cost.

**Headline result (100 games, `openai/gpt-5.4`):** VOI has the best cost-adjusted
utility at every question cost while scaling its question budget with the cost
(2.11 → 0.64 → 0.09 questions as cost rises 0.01 → 0.05 → 0.10).

---

## 1. Task & reward

Each flight is an 8-D feature vector (normalized to `[0,1]`), in this order:

```
[arrival, american, delta, jetblue, southwest, layover, stops, price]
```

A user has a hidden 8-D **preference weight vector θ** (per user, values can be
negative). A flight's reward is the dot product:

```
reward(flight) = θ · features
```

---

## 2. Methods compared

| Method | When it stops asking |
|---|---|
| `no_question` | never asks; commits immediately |
| `confidence` (`ct`) | asks until belief-distribution confidence ≥ `ct` (0.5 / 0.7 / 0.9) |
| `fixed_num_questions` (`NQ`) | always asks exactly `NQ` questions (1 / 2 / 3) |
| `voi` (`question_cost`) | each turn scores all 5 candidate questions (price, stops, layover, arrival, airline) and asks the best while its estimated VOI ≥ `question_cost`; else commits (0.01 / 0.05 / 0.10) |

The cost-adjusted objective is `Utility = reward − cost · #Q`.

---

## 3. Visualize the results (no LLM calls — instant)

The repository ships the exact per-turn transcripts under `results/`, so the
comparison table reproduces deterministically and for free. Each game is rolled
out once to `max_questions` and every turn is logged (`flight_rec.py
--log_per_turn`); `evaluate.py` then derives every method/threshold **offline**
from those transcripts, so all VOI costs (and all confidence/fixed thresholds)
share a single trajectory per game.

```bash
cd flight_rec
python evaluate.py openai/gpt-5.4
```

---

## 4. Reproduce from scratch (runs the LLM)

1. Set the API key in your environment (never stored in code):

   ```bash
   export OPENROUTER_API_KEY=sk-or-...   # provider="openrouter" (default)
   # or
   export OPENAI_API_KEY=sk-...          # provider="openai"
   ```

2. Run the two per-turn rollouts and build the table offline:

   ```bash
   cd flight_rec
   bash run_experiments.sh
   python evaluate.py openai/gpt-5.4 --root results_reproduce
   ```

`run_experiments.sh` rolls out each game once to `max_questions` and shards the
games across parallel workers. Reproductions are written to `results_reproduce/`
so they never clobber the shipped `results/`. Override defaults via env vars, e.g.
`MODELS="openai/gpt-5.4" NUM_GAMES=100 WORKERS=10 bash run_experiments.sh`.

It launches just **two** rollouts (decoupled from cost/threshold):

| Rollout | Questioner | Per-turn signal logged | Methods derived offline |
|---|---|---|---|
| `voi` (cot) | best-VOI question each turn | `voi_next` (VOI of the next question) | VOI at every cost |
| `confidence` (belief_dist) | baseline questioner each turn | `confidence` (belief max-prob) | NoQuestion, Fixed(NQ), Confidence(ct) |

Per-game transcripts are written to `<root>/<rollout>_<model>/<game>.jsonl`, one
JSONL line per turn (prediction, reward, confidence, and `voi_next`). Because
thresholds are applied **after** the rollout, two costs that don't change the
stopping point produce **identical** results, instead of differing due to
independent stochastic runs.
