"""
Value-of-Information (VoI) stopping strategy with two key design choices that
make the VoI signal well-behaved (this is the configuration used to produce the
released results):

  1. **Full-set posterior.**  The belief is always estimated by the LLM over the
     *entire* candidate set (``env.all_subjects``).  We do NOT hard-eliminate
     candidates before scoring the belief, and we do NOT short-circuit to a
     uniform distribution for large candidate sets.  This prevents the belief
     from collapsing to a single fake-certain candidate, which would pin VoI to
     0 and make the agent stop far too early.

  2. **Best-of-N question selection.**  At each turn the guesser model proposes
     ``top_question_candidates`` distinct yes/no questions; we estimate the VoI
     of each and ask the highest-VoI one.

The agent always asks until ``max_questions``; the per-turn VoI estimate is
logged so that *when to stop* can be decided offline (see ``evaluate.py``).
"""
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from prompts import INFO_GAIN_PROMPTS
from strategy_base import TwentyQuestionsStrategy
from utils import get_llm_response


class ValueOfInformationStrategy(TwentyQuestionsStrategy):
    def __init__(
        self,
        env,
        batch_size: int = 25,
        communication_cost_base: float = 0.0,
        lookahead_k: int = 1,
        top_question_candidates: int = 3,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.all_subjects: List[str] = list(env.answer_set)
        self.candidate_mask: Dict[str, bool] = {a: True for a in self.all_subjects}
        self.batch_size = batch_size
        self.communication_cost_base = communication_cost_base
        self.lookahead_k = max(1, lookahead_k)
        self.top_question_candidates = max(1, top_question_candidates)
        self.debug = debug
        self.domain = env.domain

        self.answer_cache: Dict[Tuple[str, str], str] = {}
        self.posterior_cache: Dict[str, Dict[str, float]] = {}
        self.posterior_call_count = 0
        self._voi_exclude_stack: List[str] = []

        self.name = kwargs.get(
            "name",
            f"VoI(full_set, k={self.lookahead_k}, N={self.top_question_candidates})",
        )

        self.BATCH_ANSWER_SIM_PROMPT = INFO_GAIN_PROMPTS[self.domain]["batch_answer_simulation"]

        if self.domain == "medical":
            from prompts import reward_dict
            self.action_rewards = {name.lower(): r for name, r in reward_dict.items()}
        else:
            self.action_rewards = {a.lower(): 1.0 for a in self.all_subjects}

    # ------------------------------------------------------------- utilities
    @property
    def remaining_subjects(self) -> List[str]:
        return [a for a, ok in self.candidate_mask.items() if ok]

    @staticmethod
    def _belief_key(belief: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
        return tuple(sorted(belief.items(), key=lambda t: t[0]))

    def calculate_communication_cost(self, question: str) -> float:
        return self.communication_cost_base

    # --------------------------------------------------- question generation
    def _generate_candidate_questions(self, n: int) -> List[str]:
        """Propose up to ``n`` distinct, not-yet-asked yes/no questions."""
        banned = {q.lower() for q in self.env.questions}
        banned.update(q.lower() for q in self._voi_exclude_stack)

        questions: List[str] = []
        attempts = 0
        while len(questions) < n and attempts < n * 4:
            attempts += 1
            context = self.env.get_current_prompt_context()
            raw = get_llm_response(
                prompt=context["user_prompt"]
                + "\n(Please propose a NEW yes/no question that has NOT been asked yet.)",
                system_prompt=context["system_prompt"],
                model=self.env.model,
                max_tokens=100,
                temperature=0.7,
                provider=self.env.provider,
                verbose=False,
            )
            question = raw.split(":", 1)[1].strip() if ":" in raw else raw.strip()
            q_lc = question.lower()
            if q_lc not in banned and q_lc not in (q.lower() for q in questions):
                questions.append(question)
                banned.add(q_lc)
        return questions[:n]

    # ----------------------------------------------------- value / look-ahead
    def calculate_value_function_direct(self, belief: Dict[str, float]) -> float:
        """V(belief) = max_g E[reward | guess g]."""
        if not belief:
            return 0.0
        best = float("-inf")
        for guess, p in belief.items():
            r = self.action_rewards.get(guess.lower(), 50.0)
            best = max(best, p * r + (1 - p) * (-r))
        return best

    @lru_cache(maxsize=None)
    def _best_value_k_steps(self, belief_key: Tuple[Tuple[str, float], ...], k: int) -> float:
        belief = dict(belief_key)
        if k <= 0 or not belief:
            return self.calculate_value_function_direct(belief)
        best = self.calculate_value_function_direct(belief)  # option: guess now
        for q in self._generate_candidate_questions(self.top_question_candidates):
            best = max(best, self._expected_value_after_question(q, belief, k))
        return best

    def _expected_value_after_question(self, question: str, belief: Dict[str, float], k: int) -> float:
        """E[ V_{k-1}(posterior) ] over the hypothetical oracle answers."""
        self._voi_exclude_stack.append(question)
        try:
            if k <= 0:
                return self.calculate_value_function_direct(belief)

            subj_ans = self.simulate_answers_batched(question, self.all_subjects)
            subj_ans_lc = {s.lower(): a for s, a in subj_ans.items()}

            total = 0.0
            for ans in ("Yes", "No", "Maybe"):
                cands = [s for s in belief if subj_ans_lc.get(s, "No") == ans]
                if not cands:
                    continue
                p_ans = sum(belief[s] for s in cands)
                if p_ans == 0:
                    continue
                posterior = self.get_posterior_distribution(cands, question, ans)
                v_next = self._best_value_k_steps(self._belief_key(posterior), k - 1)
                total += p_ans * v_next
            return total
        finally:
            self._voi_exclude_stack.pop()

    def calculate_value_of_information(self, question: str, k: Optional[int] = None):
        """VoI(q) = E[value after asking q] - value(guess now), with a FULL-set prior."""
        k = self.lookahead_k if k is None else max(1, k)
        full_lc = [c.lower() for c in self.all_subjects]
        prior = self.get_posterior_distribution(full_lc)
        value_before = self.calculate_value_function_direct(prior)
        value_after = self._expected_value_after_question(question, prior, k)
        voi = value_after - value_before
        cost = self.calculate_communication_cost(question)
        if self.debug:
            print(f"  [VoI] '{question[:60]}' before={value_before:.3f} "
                  f"after={value_after:.3f} VoI={voi:.3f}")
        return voi, value_after, cost, prior

    # ----------------------------------------------------- answer simulation
    def _simulate_batch(self, question: str, subjects: List[str]) -> Dict[str, str]:
        prompt = self.BATCH_ANSWER_SIM_PROMPT.format(
            question=question, candidate_list="\n".join(subjects)
        )
        resp = get_llm_response(
            prompt=prompt,
            system_prompt=f"You are an expert on {self.domain} traits.",
            model=self.env.model,
            max_tokens=1200,
            temperature=0,
            provider=self.env.provider,
            verbose=False,
        )
        ans = {}
        for line in resp.strip().splitlines():
            if ":" not in line:
                continue
            name, txt = line.split(":", 1)
            txt = txt.lower()
            if "yes" in txt:
                ans[name.strip().lower()] = "Yes"
            elif "no" in txt:
                ans[name.strip().lower()] = "No"
            else:
                ans[name.strip().lower()] = "Maybe"
        return ans

    def simulate_answers_batched(self, question: str, subjects: List[str]) -> Dict[str, str]:
        out = {}
        for i in range(0, len(subjects), self.batch_size):
            chunk = subjects[i:i + self.batch_size]
            fresh_needed = [a for a in chunk
                            if (a.lower(), question.lower()) not in self.answer_cache]
            if fresh_needed:
                fresh = self._simulate_batch(question, fresh_needed)
                for a in fresh_needed:
                    self.answer_cache[(a.lower(), question.lower())] = fresh.get(a.lower(), "No")
            for a in chunk:
                out[a] = self.answer_cache[(a.lower(), question.lower())]
        return out

    # ------------------------------------------------------------- posterior
    def get_posterior_distribution(self, candidates: List[str],
                                   question: str = None, answer: str = None) -> Dict[str, float]:
        """LLM-estimated posterior over ``candidates`` (no large-set short-circuit)."""
        n = len(candidates)
        if n == 0:
            return {}
        if n == 1:
            return {candidates[0]: 1.0}

        qa_key = "|".join(f"{q}:{a}" for q, a in zip(self.env.questions, self.env.answers))
        if question and answer:
            qa_key += f"|{question}:{answer}"
        cache_key = f"{qa_key}||{','.join(sorted(candidates))}"
        if cache_key in self.posterior_cache:
            return self.posterior_cache[cache_key]

        qa_history = "\n".join(
            f"Q{i+1}: {q}\nA{i+1}: {a}"
            for i, (q, a) in enumerate(zip(self.env.questions, self.env.answers))
        )

        if question and answer:
            prompt = f"""Based on a game of 20 Questions about {self.domain}, estimate probabilities.

Current Q&A History:
{qa_history if qa_history else "No questions asked yet."}

Now, SUPPOSE we ask: "{question}"
And SUPPOSE the answer is: "{answer}"

Given this hypothetical scenario (existing history + this new Q&A), estimate the probability that each candidate below is the correct answer.

Candidates to evaluate:
{', '.join(candidates)}

Please provide a probability estimate (0-100) for each candidate based on how well it matches BOTH the existing history AND the hypothetical Q&A.
Consider:
- How well each candidate fits ALL the given answers (both real and hypothetical)
- Some candidates may be much more likely than others after this hypothetical answer
- Candidates that are inconsistent with the answers should get very low probabilities
- Probabilities should sum to 100

Respond in the format: "CandidateName: probability" (one per line)"""
        else:
            prompt = f"""Based on the following Q&A history from a game of 20 Questions about {self.domain}, 
estimate the probability that each candidate is the correct answer.

Q&A History:
{qa_history if qa_history else "No questions asked yet."}

Candidates to evaluate:
{', '.join(candidates)}

Please provide a probability estimate (0-100) for each candidate based on how well it matches the Q&A history.
Consider:
- How well each candidate fits ALL the given answers
- Candidates that are inconsistent with the answers should get very low probabilities
- Some candidates may be much more likely than others
- Probabilities should sum to 100

Respond in the format: "CandidateName: probability" (one per line)"""

        self.posterior_call_count += 1
        resp = get_llm_response(
            prompt=prompt,
            system_prompt=f"You are an expert at {self.domain} identification and probabilistic reasoning.",
            model=self.env.model,
            max_tokens=4000,  # large enough for up to ~100 candidates
            temperature=0,
            provider=self.env.provider,
            verbose=False,
        )

        posteriors: Dict[str, float] = {}
        total_prob = 0.0
        cand_lc = [c.lower() for c in candidates]
        for line in resp.strip().splitlines():
            if ":" not in line:
                continue
            cand, prob_str = line.split(":", 1)
            cand_clean = re.sub(r"^\s*\d+[\.\)\-]?\s*", "", cand).strip()
            try:
                prob = float(prob_str.strip().rstrip("%"))
            except ValueError:
                continue
            if cand_clean.lower() in cand_lc:
                posteriors[cand_clean] = prob
                total_prob += prob

        if total_prob > 0:
            posteriors = {k: v / total_prob for k, v in posteriors.items()}

        min_prob = 0.01 / n
        listed = {k.lower() for k in posteriors}
        for cand in candidates:
            if cand not in posteriors and cand.lower() not in listed:
                posteriors[cand] = min_prob

        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v / total for k, v in posteriors.items()}
        else:
            posteriors = {c: 1.0 / n for c in candidates}

        self.posterior_cache[cache_key] = posteriors
        return posteriors

    def update_possible_answers(self, question: str, oracle_answer: str):
        """Hard-eliminate candidates whose simulated answer contradicts the oracle."""
        expect = "Yes" if oracle_answer.lower().startswith("yes") else "No"
        subj_ans = self.simulate_answers_batched(question, self.all_subjects)
        for subject, ans in subj_ans.items():
            if ans != expect and ans.lower() != "maybe":
                self.candidate_mask[subject] = False

    # ---------------------------------------------------------------- turn
    def play_turn(self) -> Tuple[bool, bool, Optional[str]]:
        # Stop only when the question budget is exhausted; the *when-to-stop*
        # decision is made offline from the logged VoI signal.
        if self.env.question_count >= self.env.max_questions:
            guess, _, _, _ = self.env.force_guess()
            return True, self.env.check_guess(guess), guess

        if self.debug:
            print(f"\nTurn {self.env.question_count + 1}/{self.env.max_questions} "
                  f"| remaining candidates: {len(self.remaining_subjects)}")

        # Best-of-N question selection.
        candidate_questions = self._generate_candidate_questions(self.top_question_candidates)
        if not candidate_questions:
            context = self.env.get_current_prompt_context()
            raw = get_llm_response(
                prompt=context["user_prompt"],
                system_prompt=context["system_prompt"],
                model=self.env.model,
                max_tokens=100,
                temperature=self.env.temperature,
                provider=self.env.provider,
                verbose=self.env.verbose,
            )
            candidate_questions = [raw.split(":", 1)[1].strip() if ":" in raw else raw.strip()]

        best = None  # (voi, question, value_after, cost, prior)
        for cq in candidate_questions:
            voi_c, eva_c, cost_c, prior_c = self.calculate_value_of_information(cq)
            if best is None or voi_c > best[0]:
                best = (voi_c, cq, eva_c, cost_c, prior_c)
        voi, question, _, cost, prior = best
        if self.debug:
            print(f"  selected (best of {len(candidate_questions)}): '{question}' VoI={voi:.3f}")

        # Ask the oracle and update belief / history.
        self.env.question_count += 1
        self.env.questions.append(question)
        answer, _, is_mc = self.env.get_answer(question)
        self.env.answers.append(answer)
        self.env.update_conversation_history(question, answer)
        self.update_possible_answers(question, answer)

        # Log a running forced guess plus the VoI estimate for this turn.
        guess, confidence, _, _ = self.env.force_guess(get_logits=False)
        is_correct = self.env.check_guess(guess)
        self.env.log_turn(self.env.format_turn_result(
            question=question,
            answer=answer,
            guess=guess,
            is_multiple_choice=is_mc,
            success=is_correct,
            info_gain=voi,
            utility=voi,          # logged signal used by the offline stopping rule
            cost=cost,
            confidence=confidence,
            candidates=self.remaining_subjects,
            all_subjects=self.all_subjects,
            prior=prior,
        ))
        return False, is_correct, guess
