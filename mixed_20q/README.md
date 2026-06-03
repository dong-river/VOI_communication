
## Installation

```bash
pip install -r requirements.txt    # just the openai SDK
```

Set the API key for your provider (keys are read from the environment, never
stored in code):

```bash
export OPENROUTER_API_KEY=sk-or-...     # provider="openrouter" (default)
# or
export OPENAI_API_KEY=sk-...            # provider="openai"
```

All released results were produced with `openai/gpt-5.4` via OpenRouter.

## Reproduce the table (no API calls)

The repository ships the exact transcripts used in the paper under `results/`,
so the comparison table reproduces deterministically and for free:

```bash
python evaluate.py
```

To evaluate your own runs, point it at different transcript folders:

```bash
python evaluate.py \
    --voi_20q runs/voi_20q/yes_no_20q_openai_gpt-5.4_set_large \
    --voi_medical runs/voi_medical/yes_no_medical_openai_gpt-5.4_set_default
```

## Regenerate transcripts (requires API access)

```bash
# 200 animal games (20 questions, 100 candidates)
python run_voi.py --domain 20q --num_games 200 --out_dir runs/voi_20q

# 200 medical games (10 questions, 15 candidates)
python run_voi.py --domain medical --num_games 200 --out_dir runs/voi_medical \
    --medical_data data/MedDG.json
```

Each game writes one JSONL transcript; every line is a turn that records the
asked question, the oracle answer, the running forced guess, and the per-turn
VoI estimate under the `utility` key. Generation is independent per game, so you
can shard `--start_exp_id` / `--num_games` across parallel processes to speed
things up.

### Baselines (Confidence and Round)

The two baselines are derived from the *same* plain question-asking transcripts;
only the stopping rule differs (applied offline in `evaluate.py`):

- **Confidence** — stop once the running guess confidence ≥ threshold.
- **Round** — stop after a fixed number of questions.

Generate the baseline transcripts with `run_baseline.py` (each turn logs a
forced guess + confidence, played to the question budget):

```bash
python run_baseline.py --domain 20q     --num_games 200 --out_dir runs/baseline_20q
python run_baseline.py --domain medical --num_games 200 --out_dir runs/baseline_medical \
    --medical_data data/MedDG.json
```

Then evaluate your own VoI and baseline runs together:

```bash
python evaluate.py \
    --voi_20q       runs/voi_20q/yes_no_20q_openai_gpt-5.4_set_animal_set \
    --voi_medical   runs/voi_medical/yes_no_medical_openai_gpt-5.4_set_default \
    --conf_20q      runs/baseline_20q/yes_no_20q_openai_gpt-5.4_set_animal_set \
    --conf_medical  runs/baseline_medical/yes_no_medical_openai_gpt-5.4_set_default
```

`evaluate.py` sweeps the Confidence thresholds and Round limits internally and
reports the best one per cost, for a fair comparison against the parameter-free
VoI rule.

## Code layout

| File | Purpose |
|------|---------|
| `run_voi.py` | Generate VoI transcripts (full-set belief + best-of-N). |
| `run_baseline.py` | Generate Confidence/Round baseline transcripts. |
| `evaluate.py` | Apply offline stopping rules and print the comparison table. |
| `voi_strategy.py` | The VoI stopping strategy. |
| `baseline_strategy.py` | The plain question-asking baseline. |
| `strategy_base.py` | Shared play-loop base class. |
| `environment.py` | Game mechanics: oracle, forced guess, logging. |
| `prompts.py` | Prompts + candidate sets for both domains. |
| `utils.py` | LLM client (env-var keys) + parsing helpers. |
