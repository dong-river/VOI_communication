# voi_utils.py
# Posterior-driven VOI utilities (final-action value) for the flight recommendation task.
# Pipeline implemented:
# (1) LLM estimates priors over feature preference states from support (+Q&A) history
# (2) For a candidate question, LLM returns P(answer) and posteriors P(state | answer)
# (3) Action value = max over options of expected matched-features (equal weights by default, optional unequal feature weights)
# (4) VOI = E_y[ value_after(y) ] - value_before - question_cost

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import json
import re

from api_utils import get_llm_response
from constants import d_price, d_arr, d_lay, d_stops
from utils import humanize_option

# ====== Feature schema ======

FEATURE_STATES: Dict[str, List[str]] = {
    "price":   ["lower", "higher", "none"],
    "stops":   ["direct", "with_stops", "none"],
    "layover": ["short", "long", "none"],
    "arrival": ["little_before", "lot_before", "none"],
    # Categorical feature: which carrier the user prefers (4 carriers + no preference).
    "airline": ["american", "delta", "jetblue", "southwest", "none"],
}

ANSWER_LABELS = ["A", "B", "C"]  # default multiple-choice labels (scalar features)

# Airline is a 5-way question (A=American, B=Delta, C=JetBlue, D=Southwest, E=No pref).
AIRLINE_ANSWER_LABELS = ["A", "B", "C", "D", "E"]
# Ordered carrier states matching one-hot indices 1..4 of an option vector.
_CARRIER_STATES = ["american", "delta", "jetblue", "southwest"]


def answer_labels_for_feature(feature: str) -> List[str]:
    """Answer labels for a feature's question (airline has 5; everything else 3)."""
    return AIRLINE_ANSWER_LABELS if feature == "airline" else ANSWER_LABELS


def candidate_questions(num_questions: int = 4) -> List[Dict[str, Any]]:
    """
    Generate candidate questions for VOI evaluation.
    These are the same questions used in confidence/adaptive strategies.
    
    Args:
        num_questions: Number of questions to sample from the pool. 
                      If -1, returns all available questions.
    
    Returns:
        List of question dictionaries with id, text, and outcomes
    """
    all_questions = [
        {
            "id": "price",
            "text": "Q: Do you prefer a flight with a lower price or a higher price? (A) Lower price (B) Higher price (C) No preference",
            "outcomes": ANSWER_LABELS,
        },
        {
            "id": "stops",
            "text": "Q: Do you prefer a flight with stops or a direct flight? (A) Direct flight (B) Flight with stops (C) No preference",
            "outcomes": ANSWER_LABELS,
        },
        {
            "id": "layover",
            "text": "Q: Do you prefer a flight with a short layover or a long layover? (A) Short layover (B) Long layover (C) No preference",
            "outcomes": ANSWER_LABELS,
        },
        {
            "id": "arrival",
            "text": "Q: Do you prefer to arrive a little before your meeting or a lot before your meeting? (A) A little before (B) A lot before (C) No preference",
            "outcomes": ANSWER_LABELS,
        },
        {
            "id": "airline",
            "text": "Q: Which airline do you prefer? (A) American, (B) Delta, (C) JetBlue, (D) Southwest (E) No preference",
            "outcomes": ["A", "B", "C", "D", "E"],
        },
    ]
    
    if num_questions == -1:
        return all_questions
    
    # If requesting more questions than available, return all
    if num_questions >= len(all_questions):
        return all_questions
    
    # Return the first num_questions
    return all_questions[:num_questions]


def feature_from_question_text(qtext: str) -> str:
    q = qtext.lower()
    if "airline" in q: return "airline"
    if "price" in q:   return "price"
    if "stops" in q:   return "stops"
    if "layover" in q: return "layover"
    if "arriv" in q:   return "arrival"
    raise ValueError("Unknown feature in question text.")


# ====== Robust JSON parsing (handles extra prose from LLMs) ======

def _safe_json_loads(s: str) -> Any:
    """Try to parse JSON from a string; fall back to extracting first {...} blob."""
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try to extract the largest JSON object
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
    except Exception:
        pass
    # Try a more permissive regex for nested braces
    try:
        matches = list(re.finditer(r"\{[\s\S]*\}", s))
        if matches:
            m = matches[0]
            return json.loads(s[m.start():m.end()])
    except Exception:
        pass
    raise ValueError("Failed to parse JSON from LLM output.")


# ====== Context builders ======

def _build_support_and_qa_context(
    game: Dict[str, Any],
    interaction_history: List[Dict[str, Any]] | None = None,
    max_support_rounds: int = 5
) -> str:
    lines: List[str] = []
    rounds = game.get("rounds", [])[:max_support_rounds]
    for i, r in enumerate(rounds, 1):
        if "options" in r and "optimal_index" in r:
            try:
                lines.append(f"Round {i}: user chose {humanize_option(r['options'][r['optimal_index']])}")
            except Exception:
                # Fallback if humanize_option fails for any reason
                lines.append(f"Round {i}: user chose option index {r.get('optimal_index')}")
    if interaction_history:
        lines.append("\nClarifying Q&A so far:")
        for k, qa in enumerate(interaction_history, 1):
            q = (qa.get("question") or "").strip()
            a = (qa.get("response") or "").strip()
            if q:
                lines.append(f"Q{k}: {q}")
            if a:
                lines.append(f"User: {a}")
    return "\n".join(lines) if lines else "No support or Q&A history."


# ====== LLM estimators (priors, answer distribution, posteriors) ======

def llm_estimate_feature_prior(
    game: Dict[str, Any],
    feature: str,
    model: str,
    system_prompt: str,
    provider: str = "openai",
    interaction_history: List[Dict[str, Any]] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.2
) -> Dict[str, float]:
    """Return dict state->prob for the feature from support (+Q&A) history."""
    if feature not in FEATURE_STATES:
        raise ValueError(f"Unknown feature: {feature}")
    states = FEATURE_STATES[feature]
    ctx = _build_support_and_qa_context(game, interaction_history)

    prompt = f"""You are calibrating a probabilistic user model.

Feature: {feature}
History (support + any clarifying Q&A):
{ctx}

Based ONLY on this history, estimate P(state).
Return STRICT JSON with keys exactly {states} and values as probabilities that sum to 1.
Example:
{{"{states[0]}": 0.33, "{states[1]}": 0.33, "{states[2]}": 0.34}}
JSON:"""
    try:
        resp = get_llm_response(prompt=prompt, system_prompt=system_prompt, model=model, provider=provider,
                              max_tokens=max_tokens, temperature=temperature, stop_strs=None)
        prior = _safe_json_loads(resp)
    except Exception:
        prior = {s: 1.0 / len(states) for s in states}

    # Normalize and sanitize
    ps = {s: float(max(0.0, prior.get(s, 0.0))) for s in states}
    Z = sum(ps.values())
    if Z <= 0:
        ps = {s: 1.0 / len(states) for s in states}
        Z = 1.0
    return {s: ps[s] / Z for s in states}


def llm_estimate_feature_posterior_with_options(
    game: Dict[str, Any],
    query_round: Dict[str, Any],
    feature: str,
    model: str,
    system_prompt: str,
    provider: str = "openai",
    interaction_history: List[Dict[str, Any]] | None = None,
    max_tokens: int = 700,
    temperature: float = 0.2
) -> Dict[str, float]:
    """
    Estimate P(state | history including appended Q&A, plus current options) for a given feature.
    Returns dict state->prob that sums to 1.
    """
    if feature not in FEATURE_STATES:
        raise ValueError(f"Unknown feature: {feature}")
    states = FEATURE_STATES[feature]
    hist_ctx = _build_support_and_qa_context(game, interaction_history)
    try:
        opts = query_round["options"]
        option_ctx = f"A) {humanize_option(opts[0])}\nB) {humanize_option(opts[1])}\nC) {humanize_option(opts[2])}"
    except Exception:
        opts = query_round.get("options", ["A","B","C"])  # type: ignore
        option_ctx = f"A) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}"

    prompt = f"""You are calibrating a probabilistic user model.

Feature: {feature}
History (support + any clarifying Q&A):
{hist_ctx}

Current options:
{option_ctx}

Based on the full history including the most recent Q&A, estimate the posterior distribution over the user's {feature} state.
Return STRICT JSON with keys exactly {states} and values as probabilities that sum to 1.
Example:
{{"{states[0]}": 0.33, "{states[1]}": 0.33, "{states[2]}": 0.34}}
JSON:"""
    try:
        
        resp = get_llm_response(prompt=prompt, system_prompt=system_prompt, model=model, provider=provider,
                              max_tokens=max_tokens, temperature=temperature, stop_strs=None)
        posterior = _safe_json_loads(resp)
    except Exception:
        posterior = {s: 1.0 / len(states) for s in states}

    ds = {s: float(max(0.0, posterior.get(s, 0.0))) for s in states}
    Zs = sum(ds.values())
    if Zs <= 0:
        ds = {s: 1.0 / len(states) for s in states}
        Zs = 1.0
    return {s: ds[s] / Zs for s in states}


def llm_estimate_answer_and_posteriors(
    game: Dict[str, Any],
    query_round: Dict[str, Any],
    question_text: str,
    model: str,
    system_prompt: str,
    provider: str = "openai",
    interaction_history: List[Dict[str, Any]] | None = None,
    max_tokens: int = 800,
    temperature: float = 0.2
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Ask LLM for:
      - P(answer=A/B/C | history, options)  (answer distribution)
      - For each possible answer y, posterior over states P(Z_f | history, y)
    Returns: (py, post_by_y)
      py: dict label->prob,
      post_by_y: dict label-> dict state->prob
    """
    feature = feature_from_question_text(question_text)
    states = FEATURE_STATES[feature]
    labels = answer_labels_for_feature(feature)
    n_lab = len(labels)
    uni = round(1.0 / n_lab, 2)

    opts = query_round["options"]
    try:
        option_ctx = f"A) {humanize_option(opts[0])}\nB) {humanize_option(opts[1])}\nC) {humanize_option(opts[2])}"
    except Exception:
        option_ctx = f"A) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}"
    hist_ctx = _build_support_and_qa_context(game, interaction_history)

    labels_str = ",".join(labels)
    p_answer_example = "{" + ", ".join(f'"{l}": {uni}' for l in labels) + "}"
    state_uni = round(1.0 / len(states), 2)
    posterior_example = ",\n".join(
        f'    "{l}": {{' + ", ".join(f'"{s}": {state_uni}' for s in states) + "}"
        for l in labels
    )
    prompt = f"""You are a calibrated modeling assistant.

We are about to ask the user this multiple-choice question about {feature}:
{question_text}

History (support + any clarifying Q&A):
{hist_ctx}

Current options:
{option_ctx}

1) Predict the probability of each answer choice ({labels_str}) the user would give.
2) For each possible answer y in {{{labels_str}}}, give the posterior distribution over the user's {feature} state in {states}.

Return STRICT JSON:
{{
  "P_answer": {p_answer_example},
  "posterior_given": {{
{posterior_example}
  }}
}}
JSON:"""
    try:
        resp = get_llm_response(prompt=prompt, system_prompt=system_prompt, model=model, provider=provider,
                              max_tokens=max_tokens, temperature=temperature, stop_strs=None)
        obj = _safe_json_loads(resp)
        py = obj.get("P_answer", {})
        post_by_y_seed = obj.get("posterior_given", {})
    except Exception:
        # Fallback: uniform answers and echo priors as seed posteriors
        py = {lbl: 1.0 / n_lab for lbl in labels}
        prior = llm_estimate_feature_prior(game, feature, model, system_prompt, provider, interaction_history)
        post_by_y_seed = {lbl: prior for lbl in labels}

    # Normalize and sanitize
    py2 = {lbl: float(max(0.0, py.get(lbl, 0.0))) for lbl in labels}
    Zy = sum(py2.values())
    if Zy <= 0:
        py2 = {lbl: 1.0 / n_lab for lbl in labels}
        Zy = 1.0
    py2 = {lbl: py2[lbl] / Zy for lbl in labels}

    # Recompute per-answer posteriors using appended Q&A and options.
    # Generic parser: split the question on each "(<label>)" marker, in order.
    def _extract_option_texts(qtxt: str) -> Dict[str, str]:
        opts_text: Dict[str, str] = {l: l for l in labels}
        found = [(l, qtxt.find(f"({l})")) for l in labels]
        found = sorted([(l, i) for l, i in found if i != -1], key=lambda x: x[1])
        for k, (l, i) in enumerate(found):
            start = i + len(f"({l})")
            end = found[k + 1][1] if k + 1 < len(found) else len(qtxt)
            txt = qtxt[start:end].strip().strip(",").strip()
            if txt:
                opts_text[l] = txt
        return opts_text

    option_texts = _extract_option_texts(question_text)

    post_by_y: Dict[str, Dict[str, float]] = {}
    for lbl in labels:
        appended_history = list(interaction_history or []) + [{
            "question": question_text,
            "response": f"{lbl}) {option_texts.get(lbl, lbl)}",
            "question_number": (interaction_history[-1].get("question_number", 0) + 1) if interaction_history else 1
        }]
        try:
            posterior = llm_estimate_feature_posterior_with_options(
                game=game,
                query_round=query_round,
                feature=feature,
                model=model,
                system_prompt=system_prompt,
                provider=provider,
                interaction_history=appended_history,
            )
        except Exception:
            posterior = post_by_y_seed.get(lbl, {s: 1.0 / len(states) for s in states})

        ds = {s: float(max(0.0, posterior.get(s, 0.0))) for s in states}
        Zs = sum(ds.values())
        if Zs <= 0:
            ds = {s: 1.0 / len(states) for s in states}
            Zs = 1.0
        post_by_y[lbl] = {s: ds[s] / Zs for s in states}

    return py2, post_by_y


def llm_predict_option_distribution(
    game: Dict[str, Any],
    query_round: Dict[str, Any],
    model: str,
    system_prompt: str,
    provider: str = "openai",
    interaction_history: List[Dict[str, Any]] | None = None,
    max_tokens: int = 600,
    temperature: float = 0.2
) -> Dict[str, float]:
    """
    Predict calibrated distribution over options A/B/C given history (+Q&A) and current options.
    Returns dict label->prob summing to 1.
    """
    labels = ANSWER_LABELS
    try:
        opts = query_round["options"]
        option_ctx = f"A) {humanize_option(opts[0])}\nB) {humanize_option(opts[1])}\nC) {humanize_option(opts[2])}"
    except Exception:
        opts = query_round.get("options", ["A","B","C"])  # type: ignore
        option_ctx = f"A) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}"
    hist_ctx = _build_support_and_qa_context(game, interaction_history)

    prompt = f"""You are a calibrated modeling assistant.

History (support + any clarifying Q&A):
{hist_ctx}

Current options:
{option_ctx}

Predict the calibrated probability distribution over choices A, B, C.
Return STRICT JSON:
{{
  "choice_probs": {{"A": 0.33, "B": 0.33, "C": 0.34}}
}}
JSON:"""

    try:
        resp = get_llm_response(prompt=prompt, system_prompt=system_prompt, model=model, provider=provider,
                              max_tokens=max_tokens, temperature=temperature, stop_strs=None)
        obj = _safe_json_loads(resp)
        cp = obj.get("choice_probs", {})
    except Exception:
        cp = {lbl: 1/3 for lbl in labels}

    ds = {lbl: float(max(0.0, cp.get(lbl, 0.0))) for lbl in labels}
    Zs = sum(ds.values())
    if Zs <= 0:
        ds = {lbl: 1/3 for lbl in labels}
        Zs = 1.0
    return {lbl: ds[lbl] / Zs for lbl in labels}


# ====== Attribute extraction and matching ======

def _col_from_options(options: List[List[float]], acc_or_idx) -> List[float]:
    """Extract a column from option vectors using either an accessor (callable) or an integer index."""
    if callable(acc_or_idx):
        return [float(acc_or_idx(opt)) for opt in options]
    idx = int(acc_or_idx)
    return [float(opt[idx]) for opt in options]


def _carrier_of(option: List[float]) -> str:
    """Return the carrier state of an option from its one-hot dims (indices 1..4)."""
    onehot = option[1:5]
    idx = max(range(4), key=lambda i: onehot[i])
    return _CARRIER_STATES[idx] if onehot[idx] > 0.5 else "none"


def option_attribute_views(options: List[List[float]]) -> Dict[str, Any]:
    """
    Pull attributes from option vectors. Uses integer indices to extract features.
    Feature order: [arrival, american, delta, jetblue, southwest, layover, stops, price]
    For airline (categorical) we return the carrier-state string per option.
    """
    return {
        "price":   _col_from_options(options, 7),  # index 7 for price
        "stops":   _col_from_options(options, 6),  # index 6 for stops  
        "layover": _col_from_options(options, 5),  # index 5 for layover
        "arrival": _col_from_options(options, 0),  # index 0 for arrival
        "airline": [_carrier_of(opt) for opt in options],  # carrier name per option
    }


def _median_of_three(vals: List[float]) -> float:
    s = sorted(vals)
    return s[1]  # median for 3 items


def match_indicator(feature: str, option_vals: List[float], z_state: str) -> List[int]:
    """
    Returns a length-3 list; entry j is 1 if option j matches the state for the feature, else 0.
    Ties handled conservatively; normalized floats handled with small eps.
    """
    eps = 1e-9

    if feature == "price":
        mn = min(option_vals)
        mx = max(option_vals)
        is_min = [v <= mn + eps for v in option_vals]
        is_max = [v >= mx - eps for v in option_vals]
        if z_state == "lower":
            return [1 if is_min[j] else 0 for j in range(3)]
        if z_state == "higher":
            return [1 if is_max[j] else 0 for j in range(3)]
        return [0, 0, 0]

    if feature == "stops":
        # values may be normalized floats; treat near-zero as direct
        is_direct = [abs(v) < eps for v in option_vals]
        is_with   = [v > eps      for v in option_vals]
        if z_state == "direct":
            return [1 if is_direct[j] else 0 for j in range(3)]
        if z_state == "with_stops":
            return [1 if is_with[j] else 0 for j in range(3)]
        return [0, 0, 0]

    if feature == "layover":
        med = _median_of_three(option_vals)
        is_short = [v <= med + eps for v in option_vals]
        is_long  = [v >= med - eps for v in option_vals]
        if z_state == "short":
            return [1 if (is_short[j] and not is_long[j]) else 0 for j in range(3)]
        if z_state == "long":
            return [1 if (is_long[j] and not is_short[j]) else 0 for j in range(3)]
        return [0, 0, 0]

    if feature == "arrival":
        med = _median_of_three(option_vals)
        is_little = [v <= med + eps for v in option_vals]  # arrive a little earlier
        is_lot    = [v >= med - eps for v in option_vals]  # arrive a lot earlier
        if z_state == "little_before":
            return [1 if (is_little[j] and not is_lot[j]) else 0 for j in range(3)]
        if z_state == "lot_before":
            return [1 if (is_lot[j] and not is_little[j]) else 0 for j in range(3)]
        return [0, 0, 0]

    if feature == "airline":
        # option_vals here are carrier-state strings (from option_attribute_views).
        if z_state == "none":
            return [0, 0, 0]
        return [1 if option_vals[j] == z_state else 0 for j in range(3)]

    return [0, 0, 0]


# ====== Expected scores & action value (belief-based) ======

def expected_feature_match_per_option(
    beliefs_per_feature: Dict[str, Dict[str, float]],
    options: List[List[float]],
    feature_weights: Dict[str, float] | None = None
) -> List[float]:
    """
    Expected number of matched features for each option j.
    If feature_weights is provided, it weights features accordingly.
    """
    vals = option_attribute_views(options)
    scores = [0.0, 0.0, 0.0]
    for feat, belief in beliefs_per_feature.items():
        if feat not in vals:
            continue
        w = 1.0
        if feature_weights is not None:
            w = float(feature_weights.get(feat, 0.0))
        for z_state, pz in belief.items():
            match = match_indicator(feat, vals[feat], z_state)
            scores = [s + w * pz * m for s, m in zip(scores, match)]
    return scores


def action_value_from_beliefs(
    beliefs_per_feature: Dict[str, Dict[str, float]],
    options: List[List[float]],
    feature_weights: Dict[str, float] | None = None
) -> Tuple[float, int, List[float]]:
    """
    Value = max over options of expected matched-features (weighted if feature_weights is provided).
    Returns (value, best_index, per_option_expected_scores)
    """
    scores = expected_feature_match_per_option(beliefs_per_feature, options, feature_weights)
    best_idx = max(range(3), key=lambda j: scores[j])
    return scores[best_idx], best_idx, scores


# ====== VOI using LLM posteriors (belief-based value) ======

def voi_for_question(
    game: Dict[str, Any],
    query_round: Dict[str, Any],
    question_text: str,
    priors_per_feature: Dict[str, Dict[str, float]],
    model: str,
    system_prompt: str,
    provider: str = "openai",
    interaction_history: List[Dict[str, Any]] | None = None,
    question_cost: float = 0.0,
    debug: bool = True,
) -> Tuple[float, float, float]:
    """
    VOI(q) = E_y[ V(b_{-f} ∪ posterior_f|y) ] - V(b_current) - cost
    where V is action value (best expected matched-features) computed from beliefs.

    The agent's internal value uses category-equal feature weights (arrival/layover/
    stops/price = 0.2 each); it never sees the user's true per-game reward vector.
    """
    options = query_round["options"]

    # Category-equal feature weights (the only supported setting).
    feature_weights: Dict[str, float] = {
        "arrival": 0.2,
        "layover": 0.2,
        "stops":   0.2,
        "price":   0.2,
        "airline": 0.2,
    }

    # 1) Value before asking (from priors)
    V0, _, _ = action_value_from_beliefs(
        beliefs_per_feature=priors_per_feature,
        options=options,
        feature_weights=feature_weights
    )

    if debug:
        try:
            feat_dbg = feature_from_question_text(question_text)
        except Exception:
            feat_dbg = "?"
        print(f"[VOI DEBUG] Question: {question_text}")
        print(f"[VOI DEBUG] Feature: {feat_dbg}")
        if feature_weights is None:
            print("[VOI DEBUG] Feature weights: equal")
        else:
            print(f"[VOI DEBUG] Feature weights: {json.dumps(feature_weights)}")
        print(f"[VOI DEBUG] V0 (belief-based value): {V0:.4f}")

    # 2) Predict P(answer) and posteriors given each answer
    py, post_by_y = llm_estimate_answer_and_posteriors(
        game, query_round, question_text, model, system_prompt, provider, interaction_history
    )
    if debug:
        print(f"[VOI DEBUG] P(answer): {json.dumps(py)}")

    asked_feature = feature_from_question_text(question_text)

    # 3) Expected value after asking = sum_y p(y) * V(b with asked_feature updated by posterior|y)
    Vexp = 0.0
    for y, pyv in py.items():
        beliefs_y = dict(priors_per_feature)
        beliefs_y[asked_feature] = post_by_y.get(y, beliefs_y[asked_feature])

        Vy, _, _ = action_value_from_beliefs(
            beliefs_per_feature=beliefs_y,
            options=options,
            feature_weights=feature_weights
        )
        Vexp += float(pyv) * Vy
        if debug:
            print(f"[VOI DEBUG] Vy given {y}: {Vy:.4f} (weight {float(pyv):.4f})")

    voi = Vexp - V0 - float(question_cost)
    # clamp tiny negatives due to numerical noise
    if -1e-9 < voi < 0:
        voi = 0.0
    if debug:
        print(f"[VOI DEBUG] Vexp: {Vexp:.4f}; cost: {float(question_cost):.4f}; VOI: {voi:.4f}")
    return voi, Vexp, V0


# Maps (feature, answer_label) -> deterministic state distribution
_ANSWER_TO_STATE: Dict[str, Dict[str, str]] = {
    "price":   {"A": "lower",        "B": "higher",      "C": "none"},
    "stops":   {"A": "direct",       "B": "with_stops",  "C": "none"},
    "layover": {"A": "short",        "B": "long",        "C": "none"},
    "arrival": {"A": "little_before","B": "lot_before",  "C": "none"},
    "airline": {"A": "american", "B": "delta", "C": "jetblue", "D": "southwest", "E": "none"},
}


def _deterministic_prior(feature: str, answer_label: str) -> Dict[str, float]:
    """Return a point-mass prior when the user has already answered a question."""
    states = FEATURE_STATES[feature]
    known_state = _ANSWER_TO_STATE.get(feature, {}).get(answer_label.strip().upper(), "none")
    return {s: (1.0 if s == known_state else 0.0) for s in states}


def _answered_features(
    interaction_history: List[Dict[str, Any]] | None,
) -> Dict[str, str]:
    """
    Return {feature: answer_label} for every question already answered.
    The answer label is the first uppercase letter found in the response.
    """
    answered: Dict[str, str] = {}
    for entry in (interaction_history or []):
        try:
            feat = feature_from_question_text(entry.get("question", ""))
        except Exception:
            continue
        resp = (entry.get("response") or "").strip()
        # Pull out the first capital letter A-E as the chosen label (airline uses up to E)
        m = re.search(r"\b([A-E])\b", resp)
        label = m.group(1) if m else resp[:1].upper()
        answered[feat] = label
    return answered


def choose_best_voi_question(
    game: Dict[str, Any],
    query_round: Dict[str, Any],
    model: str,
    system_prompt: str,
    provider: str = "openai",
    interaction_history: List[Dict[str, Any]] | None = None,
    question_cost: float = 0.0,
    debug: bool = True,
    num_candidate_questions: int = 4,
) -> Tuple[str | None, float]:
    """
    Compute VOI for each candidate question, return (best_question_text, best_voi).
    Returns (None, 0.0) if no question has positive VOI.
    
    Args:
        num_candidate_questions: Number of questions to consider from candidate pool.
                               -1 means use all available questions.
    """
    # Determine which features have already been answered
    answered = _answered_features(interaction_history)

    # Get candidate questions (now parameterized), excluding already-answered ones
    all_questions = candidate_questions(num_candidate_questions)
    questions = [q for q in all_questions if q["id"] not in answered]

    if debug:
        if answered:
            print(f"[VOI DEBUG] Skipping already-answered features: {list(answered.keys())}")
        print(f"[VOI DEBUG] Evaluating {len(questions)} candidate questions")

    if not questions:
        return None, 0.0

    # Build priors:
    # - For already-answered features: deterministic point mass (VOI would be 0 anyway,
    #   but we still need them in the belief dict for action_value_from_beliefs)
    # - For unanswered features: LLM estimate from history
    priors: Dict[str, Dict[str, float]] = {}

    # Seed deterministic priors for answered features (used in V0 / Vexp computations)
    for feat, label in answered.items():
        priors[feat] = _deterministic_prior(feat, label)

    # LLM-estimated priors only for features still being considered
    for q in questions:
        feat = q["id"]
        if feat not in priors:
            priors[feat] = llm_estimate_feature_prior(
                game, feat, model, system_prompt, provider, interaction_history
            )
    if debug:
        print("[VOI DEBUG] Priors per feature:")
        for feat, dist in priors.items():
            src = "known" if feat in answered else "estimated"
            print(f"  {feat} [{src}]: {json.dumps(dist)}")

    best_text: str | None = None
    best_voi = 0.0
    for q in questions:
        v, _, _ = voi_for_question(
            game=game,
            query_round=query_round,
            question_text=q["text"],
            priors_per_feature=priors,
            model=model,
            system_prompt=system_prompt,
            provider=provider,
            interaction_history=interaction_history,
            question_cost=question_cost,
            debug=debug,
        )
        if debug:
            print(f"[VOI DEBUG] Question '{q['id']}' -> VOI {v:.4f}")
        if v > best_voi:
            best_voi = v
            best_text = q["text"]

    if best_text is None or best_voi <= 0.0:
        return None, 0.0
    if debug:
        print(f"[VOI DEBUG] Best question: {best_text} (VOI {best_voi:.4f})")
    return best_text, best_voi
