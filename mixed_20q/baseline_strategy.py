"""
Baseline question-asking strategy (no Value of Information).

Each turn the guesser proposes a single yes/no question, the oracle answers, and
the model produces a forced guess together with a self-reported confidence
(0-100). The game is always played to the question budget and every turn is
logged with its ``question``, ``answer``, ``guess``, ``confidence`` and
``success`` fields.

Both baselines in the paper are derived from these same transcripts, with the
stopping rule applied offline in ``evaluate.py``:

  * **Confidence** : stop once the running confidence >= threshold.
  * **Round**      : stop after a fixed number of questions.
"""
from typing import Optional, Tuple

from strategy_base import TwentyQuestionsStrategy
from utils import extract_guess_from_response, get_llm_response


class ConfidenceBaselineStrategy(TwentyQuestionsStrategy):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.name = "ConfidenceBaseline"

    def _generate_question(self) -> str:
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
        return raw.split(":", 1)[1].strip() if ":" in raw else raw.strip()

    def play_turn(self) -> Tuple[bool, bool, Optional[str]]:
        # Hit the question budget -> make a final forced guess.
        if self.env.question_count >= self.env.max_questions:
            guess, _, _, _ = self.env.force_guess()
            return True, self.env.check_guess(guess), guess

        question = self._generate_question()
        self.env.question_count += 1
        self.env.questions.append(question)

        # The model may decide to guess directly inside its question turn.
        direct_guess = extract_guess_from_response(question)
        if direct_guess:
            correct = self.env.check_guess(direct_guess)
            self.env.answers.append("")
            self.env.log_turn(self.env.format_turn_result(
                question=question, answer="", guess=direct_guess,
                success=correct, confidence=100,
            ))
            return True, correct, direct_guess

        answer, _, is_mc = self.env.get_answer(question)
        self.env.answers.append(answer)
        self.env.update_conversation_history(question, answer)

        guess, confidence, _, _ = self.env.force_guess(get_logits=False)
        is_correct = self.env.check_guess(guess)
        self.env.log_turn(self.env.format_turn_result(
            question=question, answer=answer, guess=guess,
            is_multiple_choice=is_mc, success=is_correct, confidence=confidence,
        ))
        return False, is_correct, guess
