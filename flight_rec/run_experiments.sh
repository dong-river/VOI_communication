#!/usr/bin/env bash
# Reproduce the flight-recommendation comparison table (mixed_20q-style pipeline).
#
# Instead of one online run per method/threshold, we roll out each game ONCE to
# max_questions and log every turn (flight_rec.py --log_per_turn). Two rollouts:
#   * voi        -> asks the best-VOI question each turn; logs per-turn VOI.
#   * confidence -> asks the baseline questioner each turn; logs per-turn confidence.
# evaluate.py then derives EVERY method/threshold offline from those transcripts,
# so all VOI costs (and all confidence/fixed thresholds) share one trajectory.
#
# All runs use the category-equal feature-weight setting (the only supported mode):
#   the agent's internal value model weights arrival/layover/stops/price = 0.2 each;
#   final scoring always uses the user's true per-game reward vector from the dataset.
#
# Every model is run on the SAME data file + seed, so the per-game trajectories line
# up across models (e.g. gpt-5.4 vs gpt-5.4-mini are directly comparable).
#
# Requires an OpenRouter key:  export OPENROUTER_API_KEY=sk-or-...
set -euo pipefail
cd "$(dirname "$0")"

# Space-separated list of models (OpenRouter ids). Override with: MODELS="openai/gpt-5.4" bash run_experiments.sh
MODELS="${MODELS:-openai/gpt-5.4}"
PROVIDER="${PROVIDER:-openrouter}"
NUM_GAMES="${NUM_GAMES:-100}"
DATA="${DATA:-data/eval.json}"
MAX_Q="${MAX_Q:-3}"
# VOI considers ALL candidate questions (incl. airline). -1 = use all 5 features.
VOI_NUM_Q="${VOI_NUM_Q:--1}"
# Parallel workers PER strategy. Games are split into WORKERS contiguous chunks;
# each worker processes one chunk (e.g. worker 0 -> games 0..9, worker 1 -> 10..19).
# Sharding is RNG-safe: skipped games are still drawn to advance the shared seed.
WORKERS="${WORKERS:-10}"
# Reproductions write here so they never clobber the canonical results/ dir.
OUT_ROOT="${OUT_ROOT:-results_reproduce}"

if [[ "${PROVIDER}" == "openrouter" && -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set. Run: export OPENROUTER_API_KEY=sk-or-..." >&2
  exit 1
fi

mkdir -p run_logs

# chunk size = ceil(NUM_GAMES / WORKERS)
CHUNK=$(( (NUM_GAMES + WORKERS - 1) / WORKERS ))

# launch_sharded <model> <strategy> <prompt_method> <safe_tag>
launch_sharded() {
  local MODEL="$1" STRATEGY="$2" PROMPT="$3" SAFE="$4"
  local COMMON="--model $MODEL --provider $PROVIDER --num_games $NUM_GAMES --data $DATA --max_questions $MAX_Q --log_per_turn --no-voi_debug --voi_num_questions $VOI_NUM_Q --out_root $OUT_ROOT"
  # fresh output dir so stale games never linger
  rm -rf "${OUT_ROOT}/${STRATEGY}_${SAFE}"
  echo "  [$STRATEGY] launching $WORKERS workers x ~$CHUNK games each"
  local w start end
  for (( w=0; w<WORKERS; w++ )); do
    start=$(( w * CHUNK ))
    end=$(( start + CHUNK ))
    (( start >= NUM_GAMES )) && break
    (( end > NUM_GAMES )) && end=$NUM_GAMES
    python flight_rec.py $COMMON --strategy "$STRATEGY" --prompt_method "$PROMPT" \
          --game_start "$start" --game_end "$end" \
          > "run_logs/${STRATEGY}_rollout_${SAFE}_w${w}.log" 2>&1 &
  done
  wait
}

for MODEL in $MODELS; do
  SAFE="${MODEL//\//-}"
  echo "=== Per-turn rollouts: model=$MODEL provider=$PROVIDER games=$NUM_GAMES data=$DATA workers=$WORKERS voi_num_q=$VOI_NUM_Q out_root=$OUT_ROOT ==="

  # VOI rollout (cot): logs per-turn VOI; all costs derived offline.
  launch_sharded "$MODEL" "voi" "cot" "$SAFE"

  # Baseline rollout (belief_dist): NoQuestion/Fixed/Confidence derived offline.
  launch_sharded "$MODEL" "confidence" "belief_dist" "$SAFE"

  echo "--- Rollouts complete for $MODEL. Transcripts in ${OUT_ROOT}/{voi,confidence}_${SAFE}/"
done

echo "All rollouts complete. Build the tables with:"
for MODEL in $MODELS; do
  echo "    python evaluate.py $MODEL --root $OUT_ROOT"
done
