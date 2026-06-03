# main.py
# Flight recommendation experiment runner with VOI strategy integrated.

import json, random, argparse, os
from typing import List, Dict, Any, Tuple

from api_utils import get_llm_response
from constants import (
    AIRLINE_NAMES, AIRLINES,
    d_price, d_arr, d_lay, d_stops,
    PRICE_MAX_USD, ARRIVAL_MAX_MIN, LAYOVER_MAX_MIN, S_MAX_STOPS,
    RANDOM_SEED, FEATURE_ORDER
)
from utils import (
    choose_game, build_cot_prompt, build_direct_prompt, build_belief_prompt, build_belief_dist_cot_prompt,
    extract_reasoning_answer, extract_direct_answer, extract_belief_distribution, extract_belief_dist_cot,
    gold_from_theta, humanize_option, vec_fmt
)

# VOI helpers
from voi_utils import choose_best_voi_question

# ------------------------------------------------------------------------------------
# Confidence strategy turn-by-turn logging
# ------------------------------------------------------------------------------------

def log_confidence_turn_data(game_idx: int, turn_number: int, strategy: str, 
                           parsed_output: Dict[str, Any], gold_idx: int, 
                           scores: List[float], query_round: Dict[str, Any], 
                           questions_asked: int, need_question: bool, 
                           question_text: str = "", user_response: str = "",
                           output_dir: str = "./confidence_logs") -> None:
    """
    Log turn-by-turn data for confidence strategy analysis.
    
    Args:
        game_idx: Index of the current game
        turn_number: Turn number within the game (0 = initial, 1+ = after questions)
        strategy: The strategy being used
        parsed_output: Model's parsed output containing choice_probs, best_index, etc.
        gold_idx: Ground truth optimal choice index
        scores: Reward scores for each option [A, B, C]
        query_round: Current query round data with options
        questions_asked: Total questions asked so far
        need_question: Whether a question was determined to be needed
        question_text: The question asked (if any)
        user_response: User's response to the question (if any)
        output_dir: Directory to save the log files
    """
    if strategy != "confidence":
        return  # Only log for confidence strategy
    
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"game_{game_idx:04d}_confidence_turns.jsonl")
    
    # Get choice probabilities and confidence
    choice_probs = parsed_output["choice_probs"]
    max_confidence = max(choice_probs.values())
    predicted_choice = parsed_output["answer_letter"]
    predicted_index = parsed_output["best_index"]
    
    # Calculate if prediction is correct
    is_correct = (predicted_index == gold_idx)
    
    # Calculate rewards
    best_possible_reward = max(scores)
    predicted_reward = scores[predicted_index] if predicted_index is not None else min(scores)
    reward_regret = best_possible_reward - predicted_reward
    
    # Humanize options for readability
    labels = ["A", "B", "C"]
    query_options_human = {lbl: humanize_option(opt) for lbl, opt in zip(labels, query_round["options"])}
    
    turn_data = {
        "game_idx": game_idx,
        "turn_number": turn_number,
        "strategy": strategy,
        "questions_asked_so_far": questions_asked,
        "choice_probabilities": choice_probs,
        "max_confidence": max_confidence,
        "predicted_choice": predicted_choice,
        "predicted_index": predicted_index,
        "ground_truth_index": gold_idx,
        "ground_truth_scores": [round(s, 4) for s in scores],
        "is_correct": is_correct,
        "predicted_reward": round(predicted_reward, 4),
        "best_possible_reward": round(best_possible_reward, 4),
        "reward_regret": round(reward_regret, 4),
        "need_question": need_question,
        "question_asked": question_text,
        "user_response": user_response,
        "query_options": query_options_human,
        "reasoning": parsed_output["reasoning"],
        "feature_explanation": parsed_output["feature_explanation"]
    }
    
    # Append turn data to the game's log file
    with open(log_file, "a") as f:
        f.write(json.dumps(turn_data) + "\n")
    
    print(f"📝 Logged turn {turn_number} for game {game_idx} (confidence: {max_confidence:.3f}, correct: {is_correct})")

# ------------------------------------------------------------------------------------
# Questioning policy
# ------------------------------------------------------------------------------------

def determine_need_question(strategy: str, game: Dict[str, Any], parsed_output: Dict[str, Any], 
                          questions_asked: int = 0, max_questions: int = 3, fixed_max_questions = 0,
                          confidence_threshold: float = 0.7, **kwargs) -> Tuple[bool, str]:
    """
    Determine whether to ask a clarifying question based on the strategy.
    
    Args:
        strategy: Strategy type ("direct", "no_question", "confidence", "adaptive", "voi")
        game: Game data containing rounds and reward function
        parsed_output: Parsed model output from current prediction
        questions_asked: Number of questions already asked
        max_questions: Maximum questions allowed
        confidence_threshold: Minimum confidence for confidence strategy
        **kwargs: Additional parameters. Expected:
            - model, temperature, max_tokens, system_prompt
            - query_round
            - interaction_history (for VOI priors)
            - question_cost (for VOI stopping)
    
    Returns:
        Tuple of (need_question, question_text)
        - need_question: True if should ask a question
        - question_text: The question to ask (empty if need_question is False)
    """
    
    if strategy in ("no_question", "direct"):
        return False, ""
    
    elif strategy == "confidence":
        if questions_asked >= max_questions:
            return False, ""
        
        choice_probs = parsed_output["choice_probs"]
        max_confidence = max(choice_probs.values())
        print(choice_probs)
        # import pdb; pdb.set_trace()
        
        if max_confidence < confidence_threshold:
            question = _generate_clarifying_question(
                questions_asked, game,
                kwargs['model'], kwargs['provider'],
                kwargs['temperature'],
                kwargs['max_tokens'],
                kwargs['system_prompt'],
                kwargs['query_round'],
                kwargs['interaction_history']
            )
            return True, question
        
        return False, ""
    
    elif strategy == "adaptive":
        if questions_asked >= max_questions:
            return False, ""
        
        choice_probs = parsed_output["choice_probs"]
        vals_sorted = sorted(choice_probs.values(), reverse=True)
        max_confidence = vals_sorted[0]
        second = vals_sorted[1] if len(vals_sorted) > 1 else 0.0
        confidence_gap = max_confidence - second
        
        should_ask = (max_confidence < 0.8 and confidence_gap < 0.3 and questions_asked < 2)
        
        if should_ask:
            question = _generate_adaptive_question(
                choice_probs, questions_asked, game,
                kwargs['model'], kwargs['provider'],
                kwargs['temperature'],
                kwargs['max_tokens'],
                kwargs['system_prompt'],
                kwargs['query_round']
            )
            return True, question
        
        return False, ""
    
    elif strategy == 'fixed_num_questions':
        # import pdb; pdb.set_trace()
        if questions_asked >= max_questions:
            return False, ""
        if questions_asked < fixed_max_questions:
            question = _generate_clarifying_question(
                questions_asked, game,
                kwargs['model'], kwargs['provider'],
                kwargs['temperature'],
                kwargs['max_tokens'],
                kwargs['system_prompt'],
                kwargs['query_round'],
                kwargs['interaction_history']
            )
            return True, question
        return False, ""

    elif strategy == "voi":
        if questions_asked >= max_questions:
            return False, ""
        query_round = kwargs['query_round']
        if not query_round or "options" not in query_round:
            return False, ""

        best_q, best_voi = choose_best_voi_question(
            game=game,
            query_round=query_round,
            model=kwargs['model'],
            system_prompt=kwargs['system_prompt'],
            provider=kwargs['provider'],
            interaction_history=kwargs['interaction_history'],
            question_cost=kwargs['question_cost'],
            debug=kwargs['voi_debug'],
            num_candidate_questions=kwargs['voi_num_questions'],
        )
        if best_q and best_voi > 0.0:
            return True, best_q
        return False, ""
    
    else:
        # Unknown strategy -> default to no questions
        return False, ""


def _generate_clarifying_question(questions_asked: int, 
                                game: Dict[str, Any], model: str, provider: str, temperature: float, 
                                max_tokens: int, system_prompt: str, query_round: Dict[str, Any] = None,
                                interaction_history: List[Dict[str, Any]] = None) -> str:
    """Generate a clarifying question using LLM for confidence strategy."""
    
    flight_context = ""
    if query_round and "options" in query_round:
        options = query_round["options"]
        flight_context = f"""
Current flight options being compared:
A) {humanize_option(options[0])}
B) {humanize_option(options[1])}
C) {humanize_option(options[2])}
"""
    
    # Add support history context to understand user preferences
    support_context = ""
    if game and "rounds" in game:
        rounds = game["rounds"][:5]  # Take first 5 support rounds
        support_examples = []
        for i, round_data in enumerate(rounds, 1):
            if "options" in round_data and "optimal_index" in round_data:
                opts = round_data["options"]
                optimal_idx = round_data["optimal_index"]
                chosen_option = humanize_option(opts[optimal_idx])
                support_examples.append(f"Round {i}: User chose {chosen_option}")
        
        if support_examples:
            support_context = f"""
Support history showing user's past preferences:
{chr(10).join(support_examples)}
"""

    # Add previous Q&A context to avoid repeating questions
    qa_context = ""
    if interaction_history:
        qa_examples = []
        for interaction in interaction_history:
            qa_examples.append(f"Q: {interaction['question']}")
            qa_examples.append(f"User: {interaction['response']}")
        
        if qa_examples:
            qa_context = f"""
Previous clarifying questions already asked:
{chr(10).join(qa_examples)}
"""
    
    prompt = f"""You are helping a user choose between flight options A, B, and C. Based on the support examples shown below, you're uncertain about the user's preferences.

{support_context}

{flight_context}

{qa_context}

Choose a multiple-choice question below that would help you best determine the user's preference for the current flight:
1. Which airline do you prefer? (A) American, (B) Delta, (C) JetBlue, (C) Southwest (D) No preference
2. Do you prefer a flight with stops or a direct flight? (A) Direct flight, (B) Flight with stops (C) No preference
3. Do you prefer a flight with a short layover or a long layover? (A) Short layover, (B) Long layover (C) No preference
4. Do you prefer a flight with a lower price or a higher price? (A) Lower price, (B) Higher price (C) No preference
5. Do you prefer a flight that arrives a little before your meeting or a lot before your meeting? (A) A little before, (B) A lot before (C) No preference

IMPORTANT: Do not repeat questions that have already been asked above. Choose a different question that hasn't been asked yet.

Now choose the question you want to ask below. Don't include other text other than the question and options.
Question:"""

    resp = get_llm_response(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_strs=None
    )
    
    # Extract the question from response
    return resp.strip().replace("Question:", "").strip()


def _generate_adaptive_question(choice_probs: Dict[str, float], questions_asked: int,
                               game: Dict[str, Any], model: str, provider: str, temperature: float,
                               max_tokens: int, system_prompt: str, query_round: Dict[str, Any] = None) -> str:
    """Generate a question for adaptive strategy using LLM based on model uncertainty."""
    
    sorted_choices = sorted(choice_probs.items(), key=lambda x: x[1], reverse=True)
    top_choice, top_prob = sorted_choices[0]
    second_choice, second_prob = sorted_choices[1]
    
    uncertainty_context = f"I'm most confident in option {top_choice} ({top_prob:.2f}), but option {second_choice} is close ({second_prob:.2f})."
    
    # Get the actual flight options to understand what features are available
    flight_context = ""
    if query_round and "options" in query_round:
        options = query_round["options"]
        flight_context = f"""
Current flight options being compared:
A) {humanize_option(options[0])}
B) {humanize_option(options[1])}
C) {humanize_option(options[2])}
"""
    
    # Add support history context to understand user preferences
    support_context = ""
    if game and "rounds" in game:
        rounds = game["rounds"][:5]  # Take first 5 support rounds
        support_examples = []
        for i, round_data in enumerate(rounds, 1):
            if "options" in round_data and "optimal_index" in round_data:
                opts = round_data["options"]
                optimal_idx = round_data["optimal_index"]
                chosen_option = humanize_option(opts[optimal_idx])
                support_examples.append(f"Round {i}: User chose {chosen_option}")
        
        if support_examples:
            support_context = f"""
Support history showing user's past preferences:
{chr(10).join(support_examples)}
"""
    
    # Add Q&A context (placeholder since adaptive doesn't track interaction history)
    qa_context = ""
    
    prompt = f"""You are an AI assistant helping a user choose between flight options A, B, and C. You've analyzed the support examples but still have some uncertainty.

Current situation: {uncertainty_context}

{support_context}

{flight_context}

{qa_context}

Choose a multiple-choice question below that would help you best determine the user's preference for the current flight:
1. Which airline do you prefer? (A) American, (B) Delta, (C) JetBlue, (C) Southwest (D) No preference
2. Do you prefer a flight with stops or a direct flight? (A) Direct flight, (B) Flight with stops (C) No preference
3. Do you prefer a flight with a short layover or a long layover? (A) Short layover, (B) Long layover (C) No preference
4. Do you prefer a flight with a lower price or a higher price? (A) Lower price, (B) Higher price (C) No preference
5. Do you prefer a flight that arrives a little before your meeting or a lot before your meeting? (A) A little before, (B) A lot before (C) No preference

IMPORTANT: Do not repeat questions that have already been asked above. Choose a different question that hasn't been asked yet.

Now choose the question you want to ask below. Don't include other text other than the question and options.
Question:"""

    resp = get_llm_response(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_strs=None
    )
    
    return resp.strip().replace("Question:", "").strip()


def _simulate_question_response(question: str, game: Dict[str, Any], model: str, provider: str, temperature: float,
                               max_tokens: int, system_prompt: str) -> str:
    """Simulate user response to a clarifying question based on the user's preference profile."""
    
    reward_function = game["reward_function"]
    
    # Create user profile description
    profile_parts = []
    
    # Arrival time preference (index 0)
    if reward_function[0] > 0.1:
        profile_parts.append("prefers to arrive well before meetings")
    elif reward_function[0] < -0.1:
        profile_parts.append("prefers to arrive closer to meeting time")
    else:
        profile_parts.append("has no preference for arrival timing")
    
    # Airline preferences (indices 1-4: American, Delta, JetBlue, Southwest)
    airlines = ["American", "Delta", "JetBlue", "Southwest"]
    airline_prefs = reward_function[1:5]
    max_airline_idx = max(range(4), key=lambda i: airline_prefs[i])
    min_airline_idx = min(range(4), key=lambda i: airline_prefs[i])
    
    if airline_prefs[max_airline_idx] > 0.1:
        profile_parts.append(f"strongly prefers {airlines[max_airline_idx]} airline")
    elif airline_prefs[min_airline_idx] < -0.1:
        profile_parts.append(f"dislikes {airlines[min_airline_idx]} airline")
    else:
        profile_parts.append("has no preference for any specific airline")
    
    # Layover preference (index 5)
    if reward_function[5] > 0.1:
        profile_parts.append("doesn't mind longer layovers")
    elif reward_function[5] < -0.1:
        profile_parts.append("strongly prefers shorter layovers")
    else:
        profile_parts.append("has no preference for layover duration")
    
    # Number of stops preference (index 6)
    if reward_function[6] > 0.1:
        profile_parts.append("actually prefers flights with more stops")
    elif reward_function[6] < -0.1:
        profile_parts.append("strongly prefers direct flights with fewer stops")
    else:
        profile_parts.append("has no preference for number of stops")
    
    # Price preference (index 7)
    if reward_function[7] > 0.1:
        profile_parts.append("willing to pay more for flights")
    elif reward_function[7] < -0.1:
        profile_parts.append("very price-conscious and prefers cheaper flights")
    else:
        profile_parts.append("has no preference for flight price")
    
    user_profile = "This user " + ", ".join(profile_parts) + "."
    
    prompt = f"""You are simulating a user's response to a flight booking question. Here is the user's preference profile:

{user_profile}

The assistant asked a multiple-choice question: "{question}"

Respond as this user would, based on their specific preferences. You should directly choose from the options without other text.

User response:"""

    resp = get_llm_response(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_strs=None
    )
    resp = resp.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    
    return resp.strip().replace("User response:", "").strip()


# ------------------------------------------------------------------------------------
# Per-turn rollout logging (mixed_20q style: roll out to max_questions, log every turn,
# decide stopping OFFLINE from the logged signal)
# ------------------------------------------------------------------------------------

def _select_prompt_builder(prompt_method: str):
    if prompt_method == "cot":
        return build_cot_prompt, extract_reasoning_answer
    if prompt_method == "direct":
        return build_direct_prompt, extract_direct_answer
    if prompt_method == "belief_dist":
        return build_belief_prompt, extract_belief_distribution
    if prompt_method == "belief_dist_cot":
        return build_belief_dist_cot_prompt, extract_belief_dist_cot
    raise ValueError(f"Unknown prompt_method: {prompt_method}")


def run_per_turn_rollout(args, games, rng) -> None:
    """Play each game to `max_questions`, logging one JSONL line per turn.

    Stopping is NOT applied here. Each turn records the prediction/reward at that
    state plus the signal used to decide stopping offline:
      * voi rollout        -> ``voi_next`` (VOI of the question asked to advance)
      * confidence rollout -> ``confidence`` (max choice probability)

    evaluate.py then derives every method/threshold from these transcripts, so all
    VOI costs (and all confidence/fixed thresholds) share a single trajectory.
    """
    build_prompt, parser_func = _select_prompt_builder(args.prompt_method)
    safe_model = args.model.replace("/", "-")
    out_dir = os.path.join(args.out_root, f"{args.strategy}_{safe_model}")
    os.makedirs(out_dir, exist_ok=True)

    # Sharding: PROCESS only [game_start, game_end); earlier games are still drawn
    # (choose_game + build_prompt) to advance the shared RNG so every shard sees the
    # exact same game sequence a single process would.
    g_start = max(0, args.game_start)
    g_end = args.num_games if args.game_end is None or args.game_end < 0 else min(args.game_end, args.num_games)
    print(f"[per-turn rollout] strategy={args.strategy} games=[{g_start},{g_end}) of {args.num_games} "
          f"max_questions={args.max_questions} -> {out_dir}")

    for game_idx in range(args.num_games):
        if game_idx >= g_end:
            break
        try:
            game = choose_game(games, k_support=args.support_k, rng=rng)
            prompt, query_round, gold_idx = build_prompt(
                game, support_k=args.support_k,
                simulate_assistant_errors=args.simulate_errors, rng=rng)
            # Skip LLM work for games outside this shard (RNG already advanced above).
            if game_idx < g_start:
                continue
            gold_idx, scores = gold_from_theta(game, query_round)
            best_possible = max(scores)

            interaction_history: List[Dict[str, Any]] = []
            final_prompt = prompt
            asked_q = None          # question asked to reach the current state
            asked_resp = None
            turns: List[Dict[str, Any]] = []

            for d in range(args.max_questions + 1):
                # ---- prediction at the current state (d questions asked) ----
                resp = get_llm_response(
                    prompt=final_prompt, system_prompt=args.system_prompt,
                    model=args.model, provider=args.provider,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    stop_strs=None)
                try:
                    out = parser_func(resp)
                except Exception as e:
                    print(f"  game {game_idx} turn {d}: parse error: {e}")
                    break

                pred_idx = out["best_index"]
                pred_reward = scores[pred_idx] if pred_idx is not None else min(scores)
                choice_probs = out.get("choice_probs", {})
                confidence = max(choice_probs.values()) if choice_probs else 1.0

                # ---- pick the question that would advance to the next state ----
                voi_next, q_next = None, None
                if d < args.max_questions:
                    if args.strategy == "voi":
                        q_next, voi_next = choose_best_voi_question(
                            game=game, query_round=query_round, model=args.model,
                            system_prompt=args.system_prompt, provider=args.provider,
                            interaction_history=interaction_history,
                            question_cost=0.0,          # cost-free: keep the full trajectory
                            debug=False,
                            num_candidate_questions=args.voi_num_questions)
                    else:  # confidence / baseline questioner
                        q_next = _generate_clarifying_question(
                            d, game, args.model, args.provider, args.temperature,
                            args.max_tokens, args.system_prompt, query_round,
                            interaction_history)

                turns.append({
                    "game_idx": game_idx,
                    "turn": d,
                    "questions_asked": d,
                    "strategy": args.strategy,
                    "predicted_index": pred_idx,
                    "predicted_letter": out.get("answer_letter"),
                    "predicted_reward": round(pred_reward, 4),
                    "best_possible_reward": round(best_possible, 4),
                    "reward_regret": round(best_possible - pred_reward, 4),
                    "prediction_correct": bool(pred_idx == gold_idx),
                    "confidence": round(float(confidence), 4),
                    "voi_next": (round(float(voi_next), 6) if voi_next is not None else None),
                    "question_next": q_next,
                    "question_asked_to_reach": asked_q,
                    "user_response_to_reach": asked_resp,
                    "ground_truth_index": gold_idx,
                    "ground_truth_scores": [round(s, 4) for s in scores],
                })

                # stop the rollout if there is no further question to ask
                if d >= args.max_questions or q_next is None:
                    break

                # ---- ask the chosen question, fold answer into the prompt ----
                user_response = _simulate_question_response(
                    q_next, game, args.model, args.provider, args.temperature,
                    args.max_tokens, args.system_prompt)
                interaction_history.append({
                    "question": q_next, "response": user_response,
                    "question_number": d + 1})
                qa_history = "\n\nAdditional clarifying questions with user:\n"
                for it in interaction_history:
                    qa_history += f"Q: {it['question']}\nUser: {it['response']}\n\n"
                qa_history += "Based on the support history and the clarifying questions above, please make your final prediction."
                final_prompt = prompt + qa_history
                asked_q, asked_resp = q_next, user_response

            with open(os.path.join(out_dir, f"{game_idx:04d}.jsonl"), "w") as f:
                for t in turns:
                    f.write(json.dumps(t) + "\n")
            if turns:
                last = turns[-1]
                print(f"  game {game_idx+1}/{args.num_games}: logged {len(turns)} turns "
                      f"(final correct={last['prediction_correct']} reward={last['predicted_reward']})")
        except Exception as e:
            print(f"  game {game_idx}: error: {e}")

    print(f"[per-turn rollout] done -> {out_dir}")


# ------------------------------------------------------------------------------------
# Main runner
# ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval.json", help="Path to grouped JSON (eval.json or train.json)")
    ap.add_argument("--support_k", type=int, default=5, help="Number of support rounds (k) before the query round")
    ap.add_argument("--num_games", type=int, default=50, help="How many random games to evaluate")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI chat model for get_openai_gen/get_gemini_gen")
    ap.add_argument("--provider", default='openai')
    ap.add_argument("--system_prompt", default="", help="Optional system prompt")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--print_prompt", action="store_true", help="Print the first prompt sent to the LLM")
    ap.add_argument("--simulate_errors", action="store_true", help="Simulate assistant mistakes in support rounds")
    ap.add_argument("--prompt_method", choices=["cot","direct","belief_dist","belief_dist_cot"], default="cot")
    ap.add_argument("--shuffle_seed", type=int, default=RANDOM_SEED, help="Seed for option shuffling")
    ap.add_argument("--strategy", choices=["direct", "no_question", "confidence", "adaptive", "voi", "fixed_num_questions"], default="direct", 
                   help="Strategy for determining when to ask questions")
    ap.add_argument("--confidence_threshold", type=float, default=0.7, 
                   help="Confidence threshold for confidence strategy")
    ap.add_argument("--max_questions", type=int, default=3, 
                   help="Maximum number of clarifying questions to ask")  ## This applies to all questions
    ap.add_argument("--fixed_max_question", type=int, default=0)                               ## This is only for fixed_num_questions strategy
    ap.add_argument("--question_cost", type=float, default=0.0,
                   help="Per-question cost for VOI strategy (stop when VOI <= cost)")
    # NOTE: argparse `type=bool` is not what you want (bool("False") is True). Use BooleanOptionalAction when available.
    _bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if _bool_action is not None:
        ap.add_argument(
            "--voi_debug",
            action=_bool_action,
            default=True,
            help="Print detailed VOI debug info (priors, posteriors, VOI components). Use --no-voi_debug to disable.",
        )
    else:
        def _str2bool(v: str) -> bool:
            s = str(v).strip().lower()
            if s in ("1", "true", "t", "yes", "y", "on"):
                return True
            if s in ("0", "false", "f", "no", "n", "off"):
                return False
            raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {v!r}")

        ap.add_argument(
            "--voi_debug",
            type=_str2bool,
            default=True,
            help="Print detailed VOI debug info (priors, posteriors, VOI components).",
        )
    ap.add_argument("--voi_num_questions", type=int, default=4, help="Number of candidate questions to evaluate for VOI strategy (-1 for all available)")
    ap.add_argument("--game_start", type=int, default=0,
                    help="(per-turn rollout sharding) First game index to PROCESS. Games before this "
                         "are still drawn to advance the shared RNG, so shards stay reproducible.")
    ap.add_argument("--game_end", type=int, default=-1,
                    help="(per-turn rollout sharding) One past the last game index to PROCESS (-1 = num_games).")
    ap.add_argument("--out_root", default="results",
                    help="(per-turn rollout) Root dir for per-turn transcripts; each strategy writes "
                         "<out_root>/<strategy>_<model>/. Use results_reproduce for re-runs.")
    ap.add_argument("--confidence_log_dir", default="./confidence_logs", help="Directory to save confidence strategy turn logs (automatically enabled for confidence strategy)")
    ap.add_argument("--log_per_turn", action="store_true",
                    help="Roll out to max_questions and log per-turn transcripts (mixed_20q style); "
                         "stopping/thresholds are then applied offline by evaluate.py.")
    args = ap.parse_args()

    # Single RNG for ALL shuffles in this run
    rng = random.Random(args.shuffle_seed)

    with open(args.data, "r") as f:
        games = json.load(f)

    # mixed_20q-style per-turn logging: roll out once, threshold offline.
    if args.log_per_turn:
        run_per_turn_rollout(args, games, rng)
        return

    os.makedirs("./data", exist_ok=True)
    # Replace forward slashes in model name to avoid creating subdirectories
    safe_model_name = args.model.replace("/", "-")
    output_file = f"./data/flight_rec_{args.strategy}_{args.prompt_method}_{args.num_games}_{args.strategy}_True_{safe_model_name}.jsonl"
    
    # if 'confidence' in args.strategy or 'voi' == args.strategy:
    #     assert args.prompt_method in ['belief_dist', 'belief_dist_cot']

    correct_predictions = 0
    total_games = 0
    results = []
    total_questions_asked = 0
    total_predicted_reward = 0.0
    total_best_possible_reward = 0.0
    total_reward_regret = 0.0

    print(f"Running evaluation with {args.strategy} strategy and {args.prompt_method} method on {args.num_games} games...")
    if args.strategy not in ("direct", "no_question"):
        print(f"Strategy parameters: confidence_threshold={args.confidence_threshold}, max_questions={args.max_questions}, question_cost={args.question_cost}")
    print(f"Saving JSONL to: {output_file}")
    
    # Special message for confidence strategy logging
    if args.strategy == "confidence":
        print(f"🔍 Confidence strategy turn-by-turn logging enabled!")
        print(f"📁 Turn logs will be saved to: {args.confidence_log_dir}/")
        print(f"📊 Each game will generate: game_XXXX_confidence_turns.jsonl")
    
    for game_idx in range(args.num_games):
        try:
            print(f"Game {game_idx+1} of {args.num_games}")
            game = choose_game(games, k_support=args.support_k, rng=rng)
            
            # Print detailed game information
            print(f"\n{'='*60}")
            print(f"GAME {game_idx+1} DETAILS")
            print(f"{'='*60}")
            
            # Print reward function (user preferences)
            reward_function = game["reward_function"]
            print(f"User Reward Function: {[round(r, 3) for r in reward_function]}")
            print(f"Feature weights: [arrival_time, american, delta, jetblue, southwest, layover, stops, price]")
            
            # Print support rounds (user history)
            print(f"\nSUPPORT ROUNDS (k={args.support_k}):")
            print("-" * 40)
            for i, round_data in enumerate(game["rounds"][:args.support_k], 1):
                if "options" in round_data and "optimal_index" in round_data:
                    opts = round_data["options"]
                    optimal_idx = round_data["optimal_index"]
                    print(f"Round {i}:")
                    for j, opt in enumerate(opts):
                        marker = ">>> CHOSEN" if j == optimal_idx else "   "
                        print(f"  {chr(65+j)}) {humanize_option(opt)} {marker}")
                    print(f"  User chose: {chr(65+optimal_idx)} - {humanize_option(opts[optimal_idx])}")
                    print()
            print("-" * 40)

            # Build prompt (all builders accept rng now)
            if args.prompt_method == "cot":
                prompt, query_round, gold_idx = build_cot_prompt(game, support_k=args.support_k,
                                                       simulate_assistant_errors=args.simulate_errors, rng=rng)
                parser_func = extract_reasoning_answer
            elif args.prompt_method == "direct":
                prompt, query_round, gold_idx = build_direct_prompt(game, support_k=args.support_k,
                                                          simulate_assistant_errors=args.simulate_errors, rng=rng)
                parser_func = extract_direct_answer
            elif args.prompt_method == "belief_dist":
                prompt, query_round, gold_idx = build_belief_prompt(game, support_k=args.support_k,
                                                          simulate_assistant_errors=args.simulate_errors, rng=rng)
                parser_func = extract_belief_distribution
            elif args.prompt_method == 'belief_dist_cot':
                prompt, query_round, gold_idx = build_belief_dist_cot_prompt(game, support_k=args.support_k,
                                                                   simulate_assistant_errors=args.simulate_errors, rng=rng)
                parser_func = extract_belief_dist_cot

            if args.print_prompt and game_idx == 0:
                print("\n" + "="*80 + f"\nFIRST PROMPT SENT TO LLM ({args.prompt_method.upper()})\n" + "="*80)
                print(prompt)
                print("="*80 + "\n")

            # Initialize questioning loop
            questions_asked = 0
            interaction_history: List[Dict[str, Any]] = []
            final_prompt = prompt
            final_resp = None
            final_out = None
            
            # Calculate ground truth early for turn-by-turn logging
            gold_idx, scores = gold_from_theta(game, query_round)  # uses SHUFFLED query options
            
            # Main interaction loop with questioning strategy
            num_question = 0
            turn_number = 0
            while True:
                # if num_question > 0:
                #     import pdb; pdb.set_trace()
                resp = get_llm_response(
                    prompt=final_prompt,
                    system_prompt=args.system_prompt,
                    model=args.model,
                    provider=args.provider,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stop_strs=None
                )

                try:
                    out = parser_func(resp)
                    print(out)
                except Exception as e:
                    print(f"Game {game_idx+1}: Parse error: {e}")
                    results.append({"game_idx": game_idx, "prompt": final_prompt, "raw_response": resp,
                                    "prompt_method": args.prompt_method, "strategy": args.strategy, "error": str(e)})
                    break

                # Decide whether to ask a question
                need_question, question_text = determine_need_question(
                    strategy=args.strategy,
                    game=game,
                    parsed_output=out,
                    questions_asked=questions_asked,
                    max_questions=args.max_questions,
                    fixed_max_questions=args.fixed_max_question,
                    confidence_threshold=args.confidence_threshold,
                    model=args.model,
                    provider=args.provider,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    system_prompt=args.system_prompt,
                    query_round=query_round,
                    interaction_history=interaction_history,   # use Q&A so far for VOI priors
                    question_cost=args.question_cost,          # cost for VOI
                    voi_debug=args.voi_debug,
                    voi_num_questions=args.voi_num_questions,  # number of candidate questions
                )
                print("Need question: ", need_question)
                print("Question text: ", question_text)

                # Log turn data for confidence strategy (always enabled for confidence strategy)
                if args.strategy == "confidence":
                    log_confidence_turn_data(
                        game_idx=game_idx,
                        turn_number=turn_number,
                        strategy=args.strategy,
                        parsed_output=out,
                        gold_idx=gold_idx,
                        scores=scores,
                        query_round=query_round,
                        questions_asked=questions_asked,
                        need_question=need_question,
                        question_text=question_text,
                        user_response="",  # Will be filled in later if question is asked
                        output_dir=args.confidence_log_dir
                    )

                if need_question:
                    num_question += 1
                    # Ask clarifying question and get simulated user response
                    user_response = _simulate_question_response(
                        question_text, game, args.model, args.provider, args.temperature, 
                        args.max_tokens, args.system_prompt
                    )
                    print("User response: ", user_response)
                    questions_asked += 1
                    total_questions_asked += 1

                    # Log user response for confidence strategy (always enabled for confidence strategy)
                    if args.strategy == "confidence":
                        log_confidence_turn_data(
                            game_idx=game_idx,
                            turn_number=turn_number,
                            strategy=args.strategy,
                            parsed_output=out,
                            gold_idx=gold_idx,
                            scores=scores,
                            query_round=query_round,
                            questions_asked=questions_asked,
                            need_question=need_question,
                            question_text=question_text,
                            user_response=user_response,
                            output_dir=args.confidence_log_dir
                        )
                    
                    interaction_history.append({
                        "question": question_text,
                        "response": user_response,
                        "question_number": questions_asked
                    })

                    # Build new prompt incorporating the Q&A
                    qa_history = "\n\nAdditional clarifying questions with user:\n"
                    for interaction in interaction_history:
                        qa_history += f"Q: {interaction['question']}\n"
                        qa_history += f"User: {interaction['response']}\n\n"
                    
                    qa_history += "Based on the support history and the clarifying questions above, please make your final prediction."
                    final_prompt = prompt + qa_history

                    if args.print_prompt and game_idx == 0 and questions_asked == 1:
                        print("\n" + "="*80 + f"\nPROMPT WITH CLARIFYING QUESTIONS\n" + "="*80)
                        print(final_prompt)
                        print("="*80 + "\n")
                    
                    # Increment turn counter for next iteration
                    turn_number += 1
                    
                    # Continue the loop to make prediction with updated prompt
                    continue
                else:
                    # No more questions needed, finalize prediction
                    final_resp = resp
                    final_out = out
                    break
                        
            # If we exited the loop without a final prediction, use the last one
            if final_out is None:
                continue  # Skip this game due to error

            # gold_idx and scores already calculated earlier for turn-by-turn logging
            predicted_best = final_out["best_index"]
            prediction_correct = (predicted_best == gold_idx)
            print("Prediction correct: ", prediction_correct)

            # Calculate rewards
            best_possible_reward = max(scores)  # reward of optimal choice
            predicted_reward = scores[predicted_best] if predicted_best is not None else min(scores)
            reward_regret = best_possible_reward - predicted_reward  # how much reward we lost
            
            print(f"Best possible reward: {best_possible_reward:.4f}")
            print(f"Predicted reward: {predicted_reward:.4f}")
            print(f"Reward regret: {reward_regret:.4f}")

            if prediction_correct:
                correct_predictions += 1
            total_games += 1
            
            # Accumulate reward metrics
            total_predicted_reward += predicted_reward
            total_best_possible_reward += best_possible_reward
            total_reward_regret += reward_regret

            labels = ["A", "B", "C"]
            query_options_human = {lbl: humanize_option(opt) for lbl, opt in zip(labels, query_round["options"])}

            # Calculate final confidence
            choice_probs = final_out["choice_probs"]
            final_confidence = max(choice_probs.values())

            # Calibration analysis logging (for belief_dist methods)
            if args.prompt_method in ["belief_dist", "belief_dist_cot"]:
                # Log calibration data in a structured format for easy parsing
                calibration_data = {
                    "game_idx": game_idx,
                    "choice_probs": choice_probs,
                    "max_confidence": final_confidence,
                    "predicted_choice": final_out["answer_letter"],
                    "predicted_index": predicted_best,
                    "is_correct": prediction_correct,
                    "predicted_reward": predicted_reward,
                    "best_possible_reward": best_possible_reward,
                    "reward_regret": reward_regret
                }
                print(f"CALIBRATION_DATA: {json.dumps(calibration_data)}")

            results.append({
                "game_idx": game_idx,
                "prompt": final_prompt,
                "raw_response": final_resp,
                "parsed_output": final_out,
                "prompt_method": args.prompt_method,
                "strategy": args.strategy,
                "ground_truth_index": gold_idx,
                "ground_truth_scores": [round(s, 4) for s in scores],
                "predicted_index": predicted_best,
                "predicted_letter": final_out["answer_letter"],
                "prediction_correct": prediction_correct,
                "reasoning": final_out["reasoning"],
                "choice_probabilities": choice_probs,
                "feature_explanation": final_out["feature_explanation"],
                "query_options_human": query_options_human,
                "query_options_raw": [vec_fmt(opt) for opt in query_round["options"]],
                "questions_asked": questions_asked,
                "final_confidence": final_confidence,
                "interaction_history": interaction_history,
                "confidence_threshold": args.confidence_threshold if args.strategy == "confidence" else None,
                "num_question": num_question,
                "best_possible_reward": round(best_possible_reward, 4),
                "predicted_reward": round(predicted_reward, 4),
                "reward_regret": round(reward_regret, 4)
            })

            if (game_idx + 1) % 10 == 0:
                acc = correct_predictions / max(1, total_games)
                avg_questions = total_questions_asked / max(1, game_idx + 1)
                avg_predicted_reward = total_predicted_reward / max(1, total_games)
                avg_regret = total_reward_regret / max(1, total_games)
                print(f"Progress: {game_idx+1}/{args.num_games} games; accuracy: {acc:.3f}; avg questions: {avg_questions:.1f}; avg reward: {avg_predicted_reward:.3f}; avg regret: {avg_regret:.3f}")

        except Exception as e:
            print(f"Game {game_idx+1}: Unexpected error: {e}")
            results.append({"game_idx": game_idx, "prompt_method": args.prompt_method, "strategy": args.strategy, "error": str(e)})

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    final_acc = correct_predictions / max(1, total_games)
    avg_questions_per_game = total_questions_asked / max(1, total_games)
    avg_predicted_reward = total_predicted_reward / max(1, total_games)
    avg_best_possible_reward = total_best_possible_reward / max(1, total_games)
    avg_reward_regret = total_reward_regret / max(1, total_games)
    reward_efficiency = avg_predicted_reward / avg_best_possible_reward if avg_best_possible_reward > 0 else 0
    
    # Strategy-specific metrics
    valid_results = [r for r in results if "error" not in r]
    if args.strategy == "confidence" and valid_results:
        early_stops = sum(1 for r in valid_results if r["final_confidence"] >= args.confidence_threshold and r["questions_asked"] < args.max_questions)
        early_stop_rate = early_stops / len(valid_results)
    else:
        early_stop_rate = 0
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print(f"Strategy: {args.strategy}")
    print(f"Prompt method: {args.prompt_method}")
    print(f"Games evaluated: {total_games}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"Total questions asked: {total_questions_asked}")
    print(f"Average questions per game: {avg_questions_per_game:.2f}")
    print(f"Average predicted reward: {avg_predicted_reward:.4f}")
    print(f"Average best possible reward: {avg_best_possible_reward:.4f}")
    print(f"Average reward regret: {avg_reward_regret:.4f}")
    print(f"Reward efficiency: {reward_efficiency:.4f} ({reward_efficiency*100:.2f}%)")
    if args.strategy == "confidence":
        print(f"Confidence threshold: {args.confidence_threshold}")
        print(f"Early stop rate: {early_stop_rate:.3f}")
    print(f"Max questions allowed: {args.max_questions}")
    print(f"Saved: {output_file}")
    print("="*70)

    with open("result.txt", "a") as f:
        f.write(f"strategy: {args.strategy}, prompting method: {args.prompt_method}, equal_reward_weight: True model: {args.model} confidence threshold {args.confidence_threshold} utility threshold {args.question_cost}\n fixed max question {args.fixed_max_question}\n")
        f.write(f"accuracy: {final_acc:.4f}, average questions per game: {avg_questions_per_game:.2f}, average predicted reward: {avg_predicted_reward:.4f}, average reward regret: {avg_reward_regret:.4f}, reward efficiency: {reward_efficiency:.4f}\n")
        f.write("=" * 60 + "\n")
    print("Results written to result.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import pdb
        import traceback

        if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()
