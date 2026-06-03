import random
from typing import List, Dict, Any, Tuple
import textwrap
import re
import json
import copy
from constants import AIRLINE_NAMES, AIRLINES, d_price, d_arr, d_lay, d_stops, RANDOM_SEED, FEATURE_ORDER, PRICE_MAX_USD, ARRIVAL_MAX_MIN, LAYOVER_MAX_MIN, S_MAX_STOPS

def airline_from_vec(v: List[float]) -> str:
    onehots = v[1:5]
    idx = max(range(4), key=lambda i: onehots[i])
    return AIRLINE_NAMES[AIRLINES[idx]]

def humanize_option(v: List[float]) -> str:
    """Convert an 8-D normalized option vector into a *human-readable* description."""
    al = airline_from_vec(v)
    stops = d_stops(v[6])
    arr_m = d_arr(v[0])
    price = d_price(v[7])
    lay_m = d_lay(v[5])

    if stops <= 0:
        stops_str = "nonstop"
        lay_str = ""
    else:
        stops_str = f"{stops} stop" + ("" if stops == 1 else "s")
        lay_str = f", longest layover ~{lay_m} min"

    return f"{al}; {stops_str}{lay_str}; arrives ~{arr_m} min before meeting; price ${price}"

def vec_fmt(v: List[float]) -> str:
    return "[" + ", ".join(f"{x:.2f}" for x in v) + "]"

def choose_game(games: List[Dict[str, Any]], k_support: int, rng: random.Random = None) -> Dict[str, Any]:
    candidates = [g for g in games if len(g.get("rounds", [])) >= k_support + 1]
    if not candidates:
        raise RuntimeError(f"No game has at least {k_support + 1} rounds.")
    if rng is None:
        return random.choice(candidates)  # Fallback to global random for backward compatibility
    return rng.choice(candidates)  # Use seeded RNG for reproducible game selection

def round_letter(idx: int) -> str:
    return "ABC"[idx]

def shuffle_options_and_label(options: List[List[float]], gold_idx: int, rng: random.Random):
    """
    Shuffle a 3-option list and remap the correct index accordingly.
    Returns (shuffled_options, new_gold_idx, perm)
    where new_options[j] = options[perm[j]] and new_gold_idx = index j s.t. perm[j] == gold_idx.
    """
    perm = [0, 1, 2]
    rng.shuffle(perm)
    new_options = [options[i] for i in perm]
    new_gold_idx = perm.index(gold_idx)
    return new_options, new_gold_idx, perm

# --------------------------------------

def build_cot_prompt(game: Dict[str, Any], support_k: int = 5, simulate_assistant_errors: bool = True,
                     rng: random.Random = None) -> Tuple[str, Dict[str, Any]]:
    """
    COT-style: show text-only options (SHUFFLED), assistant guesses, user says correct/incorrect.
    NEW round: text-only options (SHUFFLED), request final answer with reasoning tags.
    """
    assert rng is not None, "Pass an RNG from main()"
    rounds = list(game["rounds"])
    support = rounds[:support_k]
    query = copy.deepcopy(rounds[support_k])

    sup_blocks = []
    for i, r in enumerate(support, 1):
        opts = r["options"]
        gold_idx_orig = int(r["optimal_index"])

        shuf_opts, gold_idx, _ = shuffle_options_and_label(opts, gold_idx_orig, rng)
        A, B, C = shuf_opts
        gold_letter = round_letter(gold_idx)

        first_guess_idx = rng.choice([0, 1, 2]) if simulate_assistant_errors else gold_idx
        first_guess_letter = round_letter(first_guess_idx)

        block = []
        block.append(f"Round {i}")
        block.append("Options:")
        block.append(f"A) {humanize_option(A)}")
        block.append(f"B) {humanize_option(B)}")
        block.append(f"C) {humanize_option(C)}")
        block.append(f"User: I prefer {first_guess_letter}.")
        sup_blocks.append("\n".join(block))

    # NEW Round (shuffle too)
    q_opts_orig = query["options"]
    q_shuf_opts, gold_idx, _ = shuffle_options_and_label(q_opts_orig, gold_idx=query['optimal_index'], rng=rng)  # gold idx unused for display
    query["options"] = q_shuf_opts

    qA, qB, qC = query["options"]
    query_block = []
    query_block.append("NEW Round (answer not provided):")
    query_block.append("Options:")
    query_block.append(f"A) {humanize_option(qA)}")
    query_block.append(f"B) {humanize_option(qB)}")
    query_block.append(f"C) {humanize_option(qC)}")

    header = textwrap.dedent("""\
    You are helping the same user across several rounds. In each previous round, the assistant picked one option. You must infer the user's preferences from the user history.
    You could consider user preference for airlines, price, arrival time, number of stops, and layover time.

    Based on the support history, choose the best option (A, B, or C) for the NEW round.
    
    Output format:
    <reasoning>
    [2-4 sentences explaining what preferences you inferred from the support rounds and why you chose this option]
    </reasoning>
    <answer>A</answer>
    
    (Replace A with B or C as appropriate)""").rstrip()

    task = "--- SUPPORT HISTORY ---"
    closing = textwrap.dedent("""\
    --- TASK ---
    Use ONLY the support history above to infer the user's preferences and choose the best option for the NEW round. Provide your reasoning and final answer in the specified format.
    """).rstrip()

    prompt = "\n".join([header, task, "\n\n".join(sup_blocks), "\n", "\n".join(query_block), "\n", closing])
    return prompt, query, gold_idx

def build_direct_prompt(game: Dict[str, Any], support_k: int = 5, simulate_assistant_errors: bool = True,
                        rng: random.Random = None) -> Tuple[str, Dict[str, Any]]:
    """Direct: mimic Appendix Table 1 interaction format.
    - Uses 'User'/'Model' speaker tags
    - Names options 'Flight 1/2/3'
    - Includes user feedback: 'correct' / 'incorrect. I prefer Flight k.'
    - NEW round ends with 'User: Which flight is the best option?' and 3 flights.
    """
    assert rng is not None, "Pass an RNG from main()"
    rounds = list(game["rounds"])
    support = rounds[:support_k]
    query = copy.deepcopy(rounds[support_k])

    # Opening instruction (Table 1)
    header = textwrap.dedent("""\
    User: Help me select the best flights for my trips. I have specific preferences for what I like and dislike in a flight, and these preferences remain the same. You need to figure out my preferences and select the best flights for me. Use your best judgment if you are unsure. Do not say you need more information.
    """).rstrip()

    sup_blocks = []
    for i, r in enumerate(support, 1):
        opts = r["options"]
        gold_idx_orig = int(r["optimal_index"])

        # Shuffle and remap gold
        shuf_opts, gold_idx, _ = shuffle_options_and_label(opts, gold_idx_orig, rng)
        F1, F2, F3 = shuf_opts  # Flight 1/2/3
        # Model's (possibly wrong) first guess
        first_guess_idx = rng.choice([0, 1, 2]) if simulate_assistant_errors else gold_idx

        # Build conversation-style block (Table 1)
        block = []
        block.append("User: Which flight is the best option?")
        block.append(f"Flight 1: {humanize_option(F1)}")
        block.append(f"Flight 2: {humanize_option(F2)}")
        block.append(f"Flight 3: {humanize_option(F3)}")
        block.append(f"Model: The best option is Flight {first_guess_idx+1}.")

        if first_guess_idx == gold_idx:
            block.append(f"User: Your option Flight {first_guess_idx+1} is correct.")
        else:
            block.append(f"User: Your option Flight {first_guess_idx+1} is incorrect. I prefer Flight {gold_idx+1}.")
        sup_blocks.append("\n".join(block))

    # NEW Round (shuffle and remap)
    q_shuf_opts, gold_idx, _ = shuffle_options_and_label(query["options"], gold_idx=query['optimal_index'], rng=rng)
    query["options"] = q_shuf_opts
    qF1, qF2, qF3 = query["options"]

    query_block = []
    query_block.append("User: Which flight is the best option?")
    query_block.append(f"Flight 1: {humanize_option(qF1)}")
    query_block.append(f"Flight 2: {humanize_option(qF2)}")
    query_block.append(f"Flight 3: {humanize_option(qF3)}")

    # Closing instruction to keep outputs parseable & consistent
    closing = textwrap.dedent("""\
    Respond as the model would in the conversation above for the NEW round. Answer in exactly this format:
    Model: The best option is Flight <1/2/3>.
    Do not add any extra text.
    """).rstrip()

    prompt = "\n\n".join([header, "\n\n".join(sup_blocks), "\n".join(query_block), closing])
    return prompt, query, gold_idx


def build_belief_prompt(game: Dict[str, Any], support_k: int = 5, simulate_assistant_errors: bool = True,
                        rng: random.Random = None) -> Tuple[str, Dict[str, Any]]:
    """Belief (distribution JSON). SHUFFLED options & remapped labels."""
    assert rng is not None, "Pass an RNG from main()"
    rounds = list(game["rounds"])
    support = rounds[:support_k]
    query = copy.deepcopy(rounds[support_k])

    sup_blocks = []
    for i, r in enumerate(support, 1):
        opts = r["options"]
        gold_idx_orig = int(r["optimal_index"])

        shuf_opts, gold_idx, _ = shuffle_options_and_label(opts, gold_idx_orig, rng)
        A, B, C = shuf_opts
        gold_letter = round_letter(gold_idx)

        first_guess_idx = rng.choice([0, 1, 2]) if simulate_assistant_errors else gold_idx
        first_guess_letter = round_letter(first_guess_idx)

        block = []
        block.append(f"Round {i}")
        block.append("Options:")
        block.append(f"A) {humanize_option(A)}")
        block.append(f"B) {humanize_option(B)}")
        block.append(f"C) {humanize_option(C)}")
        block.append(f"User: I prefer {first_guess_letter}.")
        sup_blocks.append("\n".join(block))

    q_shuf_opts, gold_idx, _ = shuffle_options_and_label(query["options"], gold_idx=query['optimal_index'], rng=rng)
    query["options"] = q_shuf_opts

    qA, qB, qC = query["options"]
    query_block = []
    query_block.append("NEW Round (answer not provided):")
    query_block.append("Options:")
    query_block.append(f"A) {humanize_option(qA)}")
    query_block.append(f"B) {humanize_option(qB)}")
    query_block.append(f"C) {humanize_option(qC)}")

    header = textwrap.dedent("""You are a calibrated modeling assistant.
                             
    You are helping the same user across several rounds. In each previous round, the assistant picked one option. You must infer the user's preferences from the user history.
    You could consider user preference for airlines, price, arrival time, number of stops, and layover time.
    
    Produce a calibrated probability distribution over options (A,B,C) for the NEW round.
    Output JSON ONLY with this schema:
    {
      "choice_probs": {"A": <float>, "B": <float>, "C": <float>},  // nonnegative, sum to 1
      "best_index": 0|1|2,                                         // argmax(A=0,B=1,C=2)
      "feature_explanation": "<2–4 sentences explaining the tradeoffs you inferred>"
    }""").rstrip()

    task = "--- SUPPORT HISTORY ---"
    closing = textwrap.dedent("""\
    --- TASK ---
    Use ONLY the support history above to infer the user's preferences and return JSON for the NEW round. Do not include any extra text outside JSON in your final answer.
    """).rstrip()

    prompt = "\n".join([header, task, "\n\n".join(sup_blocks), "\n", "\n".join(query_block), "\n", closing])
    return prompt, query, gold_idx

def build_belief_dist_cot_prompt(game: Dict[str, Any], support_k: int = 5, simulate_assistant_errors: bool = True,
                                 rng: random.Random = None) -> Tuple[str, Dict[str, Any]]:
    """Reasoning + distribution JSON. SHUFFLED options & remapped labels."""
    assert rng is not None, "Pass an RNG from main()"
    rounds = list(game["rounds"])
    support = rounds[:support_k]
    query = copy.deepcopy(rounds[support_k])

    sup_blocks = []
    for i, r in enumerate(support, 1):
        opts = r["options"]
        gold_idx_orig = int(r["optimal_index"])

        shuf_opts, gold_idx, _ = shuffle_options_and_label(opts, gold_idx_orig, rng)
        A, B, C = shuf_opts
        gold_letter = round_letter(gold_idx)

        first_guess_idx = rng.choice([0, 1, 2]) if simulate_assistant_errors else gold_idx
        first_guess_letter = round_letter(first_guess_idx)

        block = []
        block.append(f"Round {i}")
        block.append("Options:")
        block.append(f"A) {humanize_option(A)}")
        block.append(f"B) {humanize_option(B)}")
        block.append(f"C) {humanize_option(C)}")
        block.append(f"User: I prefer {first_guess_letter}.")
        sup_blocks.append("\n".join(block))

    q_shuf_opts, gold_idx, _ = shuffle_options_and_label(query["options"], gold_idx=query['optimal_index'], rng=rng)
    query["options"] = q_shuf_opts

    qA, qB, qC = query["options"]
    query_block = []
    query_block.append("NEW Round (answer not provided):")
    query_block.append("Options:")
    query_block.append(f"A) {humanize_option(qA)}")
    query_block.append(f"B) {humanize_option(qB)}")
    query_block.append(f"C) {humanize_option(qC)}")

    header = textwrap.dedent("""You are a calibrated modeling assistant.
                             
    You are helping the same user across several rounds. In each previous round, the assistant picked one option. You must infer the user's preferences from the user history.
    You could consider user preference for airlines, price, arrival time, number of stops, and layover time.
    
    Based on the support history, provide your reasoning and then a calibrated probability distribution over options (A,B,C) for the NEW round.
    
    Output format:
    <reasoning>
    [2-4 sentences explaining what preferences you inferred from the support rounds and your analysis of the options]
    </reasoning>
    {
      "choice_probs": {"A": <float>, "B": <float>, "C": <float>},  // nonnegative, sum to 1
      "best_index": 0|1|2,                                         // argmax(A=0,B=1,C=2)
      "feature_explanation": "<2–4 sentences explaining the tradeoffs you inferred>"
    }""").rstrip()

    task = "--- SUPPORT HISTORY ---"
    closing = textwrap.dedent("""\
    --- TASK ---
    Use ONLY the support history above to infer the user's preferences. First provide your reasoning in <reasoning> tags, then output the JSON probability distribution for the NEW round.
    """).rstrip()

    prompt = "\n".join([header, task, "\n\n".join(sup_blocks), "\n", "\n".join(query_block), "\n", closing])
    return prompt, query, gold_idx

def dot(a: List[float], b: List[float]) -> float:
    return sum(x*y for x, y in zip(a, b))

def gold_from_theta(game: Dict[str, Any], query_round: Dict[str, Any]) -> Tuple[int, List[float]]:
    theta = game.get("reward_weights") or game.get("reward_function") or game.get("reward")
    if theta is None:
        theta = query_round.get("reward_weights") or game["rounds"][0]["reward_weights"]
    scores = [dot(theta, o) for o in query_round["options"]]
    best_idx = max(range(3), key=lambda i: scores[i])
    return best_idx, scores

def extract_reasoning_answer(s: str) -> Dict[str, Any]:
    s = s.strip()
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", s, re.DOTALL | re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer_match = re.search(r"<answer>([ABC])</answer>", s, re.IGNORECASE)
    if not answer_match:
        raise ValueError("No valid answer (A/B/C) found in response.")
    answer_letter = answer_match.group(1).upper()
    best_index = ord(answer_letter) - ord('A')
    choice_probs = {"A": 0.0, "B": 0.0, "C": 0.0}
    choice_probs[answer_letter] = 1.0
    return {"reasoning": reasoning, "answer_letter": answer_letter, "best_index": best_index,
            "choice_probs": choice_probs, "feature_explanation": reasoning}

def extract_direct_answer(s: str) -> Dict[str, Any]:
    s = s.strip()

    # 1) Accept the paper-style answer: "The best option is Flight 2."
    m_num = re.search(r"Flight\s*([1-3])", s, re.IGNORECASE)
    if m_num:
        n = int(m_num.group(1))
        best_index = n - 1
        answer_letter = "ABC"[best_index]
    else:
        # 2) Backward compatibility: A/B/C anywhere in output
        if len(s) >= 1 and s[0].upper() in ['A','B','C']:
            answer_letter = s[0].upper()
        else:
            match = re.search(r'\b([ABC])\b', s, re.IGNORECASE)
            if match:
                answer_letter = match.group(1).upper()
            else:
                # 3) Fallback: bare number 1/2/3
                m_bare = re.search(r'\b([1-3])\b', s)
                if not m_bare:
                    raise ValueError("No valid answer (A/B/C or Flight 1/2/3) found in response.")
                n = int(m_bare.group(1))
                best_index = n - 1
                answer_letter = "ABC"[best_index]

        best_index = ord(answer_letter) - ord('A')

    choice_probs = {"A": 0.0, "B": 0.0, "C": 0.0}
    choice_probs["ABC"[best_index]] = 1.0
    return {
        "reasoning": "",
        "answer_letter": "ABC"[best_index],
        "best_index": best_index,
        "choice_probs": choice_probs,
        "feature_explanation": f"Direct choice: {'ABC'[best_index]}"
    }


def extract_belief_distribution(s: str) -> Dict[str, Any]:
    s = s.strip()
    candidate = s if (s.startswith("{") and s.endswith("}")) else None
    if not candidate:
        last=None
        for m in re.finditer(r"\{.*\}", s, flags=re.DOTALL): last=m
        if last: candidate = last.group(0)
    if candidate is None: raise ValueError("No JSON object found in response.")
    data = json.loads(candidate)
    if isinstance(data, dict) and "choice_probs" in data and isinstance(data["choice_probs"], dict):
        pA=float(data["choice_probs"].get("A",0.0)); pB=float(data["choice_probs"].get("B",0.0)); pC=float(data["choice_probs"].get("C",0.0))
        t=pA+pB+pC
        data["choice_probs"] = {"A": pA/t, "B": pB/t, "C": pC/t} if t>0 else {"A":1/3,"B":1/3,"C":1/3}
        if "best_index" not in data:
            argmax_idx = max(range(3), key=lambda i: [data["choice_probs"]["A"], data["choice_probs"]["B"], data["choice_probs"]["C"]][i])
            data["best_index"] = argmax_idx
    if "reasoning" not in data: data["reasoning"] = data.get("feature_explanation","")
    if "answer_letter" not in data: data["answer_letter"] = "ABC"[data.get("best_index",0)]
    return data

def extract_belief_dist_cot(s: str) -> Dict[str, Any]:
    s=s.strip()
    reasoning_match=re.search(r"<reasoning>(.*?)</reasoning>", s, re.DOTALL|re.IGNORECASE)
    reasoning=reasoning_match.group(1).strip() if reasoning_match else ""
    json_area = s[reasoning_match.end():] if reasoning_match else s
    candidate = json_area.strip() if (json_area.strip().startswith("{") and json_area.strip().endswith("}")) else None
    if not candidate:
        last=None
        for m in re.finditer(r"\{.*\}", json_area, flags=re.DOTALL): last=m
        if last: candidate=last.group(0)
    if candidate is None: raise ValueError("No JSON object found in response after reasoning.")
    data=json.loads(candidate)
    if isinstance(data, dict) and "choice_probs" in data and isinstance(data["choice_probs"], dict):
        pA=float(data["choice_probs"].get("A",0.0)); pB=float(data["choice_probs"].get("B",0.0)); pC=float(data["choice_probs"].get("C",0.0))
        t=pA+pB+pC
        data["choice_probs"]={"A":pA/t,"B":pB/t,"C":pC/t} if t>0 else {"A":1/3,"B":1/3,"C":1/3}
        if "best_index" not in data:
            argmax_idx=max(range(3), key=lambda i:[data["choice_probs"]["A"],data["choice_probs"]["B"],data["choice_probs"]["C"]][i])
            data["best_index"]=argmax_idx
    data["reasoning"]=reasoning
    if "answer_letter" not in data: data["answer_letter"]="ABC"[data.get("best_index",0)]
    return data