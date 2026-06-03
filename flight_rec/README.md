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

### Result — `openai/gpt-5.4` (100 games, offline thresholding)

Best configuration per method (VOI is parameter-free: threshold = cost):

| Method | thresh | #Q | Reward | Util@0.01 | Util@0.05 | Util@0.10 |
|---|---|---|---|---|---|---|
| no_question | — | 0.00 | 0.2221 | 0.2221 | 0.2221 | 0.2221 |
| Confidence | ct=0.9 | 2.44 | 0.3327 | 0.3084 | 0.2107 | 0.0887 |
| Fixed | best NQ | 1–3 | — | 0.3119 | 0.2531 | 0.2031 |
| **VOI** | **voi≥cost** | 2.11 / 0.64 / 0.09 | — | **0.4094** | **0.2616** | **0.2440** |

- **cost 0.01**: VOI wins (0.4094, ~2.1 questions) vs. best baseline Fixed=3Q (0.3119).
- **cost 0.05**: VOI wins (0.2616, ~0.6 questions) vs. best baseline Fixed=1Q (0.2531).
- **cost 0.10**: VOI wins (0.2440, ~0.1 questions) vs. NoQuestion (0.2221).

VOI is the most question-efficient policy: it scales its question budget with the
cost, whereas the fixed/confidence baselines pay a flat penalty.

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

---

## 5. Code layout

| File | Purpose |
|------|---------|
| `flight_rec.py` | Per-turn rollout runner (`--log_per_turn`). |
| `evaluate.py` | Apply offline stopping rules and print the comparison table. |
| `voi_utils.py` | VOI candidate questions, belief/answer estimation, and scoring. |
| `utils.py` | Game construction, prompts, option/answer parsing helpers. |
| `api_utils.py` | LLM client dispatch (env-var keys; OpenAI / OpenRouter). |
| `constants.py` | Feature order, de-normalization constants, random seed. |
| `run_experiments.sh` | Launch the two sharded per-turn rollouts. |
