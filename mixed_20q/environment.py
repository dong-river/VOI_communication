"""
Twenty-Questions / Medical-diagnosis environment.

The environment owns the game mechanics (the hidden subject, the oracle that
answers questions, the forced final guess, and per-turn JSONL logging) and is
deliberately decoupled from the questioning *strategy*.
"""
import json
import os
import random
from typing import Optional

from prompts import ANIMAL_ANSWER_SET_LIST, PROMPTS
from utils import (
    check_guess_correctness,
    get_llm_response,
    is_multiple_choice_question,
    parse_guess,
    save_json_data,
)


class TwentyQuestionsEnvironment:
    def __init__(
        self,
        max_questions: int = 20,
        model: str = "openai/gpt-5.4",
        temperature: float = 0.0,
        verbose: bool = False,
        question_type: str = "yes_no",
        domain: str = "20q",
        exp_id: Optional[int] = None,
        data=None,
        provider: str = "openrouter",
        log_dir: str = "logs",
        run_id: int = 0,
        args=None,
    ):
        self.max_questions = max_questions
        self.model = model
        self.args = args
        self.animal_set = getattr(args, "animal_set", "default")

        self.guess_model = model
        self.answer_model = model

        self.temperature = temperature
        self.verbose = verbose
        self.question_type = question_type
        self.domain = domain
        self.exp_id = exp_id
        self.data = data
        self.provider = provider
        self.log_dir = log_dir
        self.run_id = run_id
        self.no_self_repo = True

        self._load_prompts_and_answers()

        # Per-game JSONL log file.
        self.log_subdir = None
        self.jsonl_log_path = None
        if log_dir:
            model_name = model.replace("/", "_")
            self.log_subdir = f"{log_dir}/{question_type}_{domain}_{model_name}_set_{self.animal_set}"
            os.makedirs(self.log_subdir, exist_ok=True)
            if self.exp_id is not None:
                self.jsonl_log_path = os.path.join(
                    self.log_subdir, f"{self.exp_id}_{self.run_id}_{self.animal_set}.jsonl"
                )
                if os.path.exists(self.jsonl_log_path):
                    open(self.jsonl_log_path, "w").close()  # truncate stale run

    # ------------------------------------------------------------------ setup
    def _load_prompts_and_answers(self):
        domain_prompts = PROMPTS.get(self.domain, {})

        if self.domain == "20q":
            self.answer_set = ANIMAL_ANSWER_SET_LIST.get(
                self.animal_set, ANIMAL_ANSWER_SET_LIST["large"]
            )
        else:
            self.answer_set = domain_prompts.get("answer_set", [])

        # Optionally cap the candidate set size.
        cap = getattr(self.args, "candidate_set_size", None)
        if cap is not None and cap > 0:
            if len(self.answer_set) < cap:
                raise ValueError(
                    f"candidate_set_size {cap} exceeds answer-set size {len(self.answer_set)}"
                )
            if self.domain == "20q" and self.exp_id is not None:
                # Keep a fixed pool of `cap` candidates, but slide the window by
                # blocks of `cap` as exp_id grows, so games past the first `cap`
                # draw fresh subjects from deeper in the large set (game i ->
                # animal i) instead of repeating the first `cap` animals. The
                # hidden subject (picked via exp_id % cap in reset) therefore
                # always lives inside its own pool. Wraps if the set is exhausted.
                start = (self.exp_id // cap * cap) % len(self.answer_set)
                self.answer_set = (self.answer_set + self.answer_set)[start:start + cap]
            else:
                self.answer_set = self.answer_set[:cap]

        answer_set_str = ", ".join(self.answer_set)
        self.all_subjects = list(self.answer_set)
        if self.verbose:
            print(f"[env] domain={self.domain} | candidates={len(self.answer_set)}")

        self.force_guess_prompt = domain_prompts.get("force_guess", "").format(answer_set=answer_set_str)
        self.final_guess_request = domain_prompts.get("final_guess_request", "")
        self.answerer_prompt = domain_prompts.get("answerer", "")
        self.answerer_prompt_free = domain_prompts.get("answerer_free", "")

        guesser_prompts = domain_prompts.get("guesser", {})
        self.guesser_prompt = guesser_prompts[self.question_type].format(
            question_count="{question_count}",
            remaining_questions="{remaining_questions}",
            answer_set=answer_set_str,
        )

    def reset(self):
        if self.domain == "20q":
            self.subject = self.answer_set[self.exp_id % len(self.answer_set)]
            self.self_repo = None
        elif self.domain == "medical":
            assert self.data is not None and self.exp_id is not None, "medical domain needs data"
            self.subject = self.data[self.exp_id]["target"]
            self.self_repo = "" if self.no_self_repo else self.data[self.exp_id]["self_repo"]
            self.conv_hist = self.data[self.exp_id]["conv_hist"]
        else:
            raise ValueError(f"Unknown domain: {self.domain}")

        self.conversation_history = ""
        self.question_count = 0
        self.questions = []
        self.answers = []
        self.turn_results = []

        if self.domain == "medical" and self.self_repo and not self.no_self_repo:
            self.conversation_history = f"Patient description: {self.self_repo}\n"

        if self.verbose:
            print(f"[env] new game -> hidden subject: {self.subject}")

    # ----------------------------------------------------------- interactions
    def get_current_prompt_context(self):
        remaining_questions = self.max_questions - self.question_count
        system_prompt = self.guesser_prompt.format(
            question_count=self.question_count, remaining_questions=remaining_questions
        )
        if self.domain == "20q":
            if self.conversation_history:
                user_prompt = (
                    f"Here's the conversation so far:\n{self.conversation_history}\n\n"
                    "What is your next question about the animal? Directly write the question "
                    "below without any preamble."
                )
            else:
                user_prompt = (
                    "You are starting a game of 20 Questions about an animal. What is your first "
                    "question? Directly write the question below without any preamble."
                )
        else:  # medical
            if self.conversation_history:
                user_prompt = (
                    f"Here's the conversation so far:\n{self.conversation_history}\n\n"
                    "What is your next question for the patient?\nDoctor: "
                )
            else:
                user_prompt = (
                    "You are the doctor trying to diagnose the patient's disease. What is your "
                    "first question for the patient?\nDoctor:"
                )
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "remaining_questions": remaining_questions,
        }

    def get_answer(self, question):
        """Ask the oracle (which knows the hidden subject) to answer a question."""
        if self.domain == "medical":
            system_prompt = self.answerer_prompt.format(
                subject=self.subject, self_repo=getattr(self, "self_repo", "")
            )
        else:
            system_prompt = self.answerer_prompt.format(subject=self.subject)

        raw_answer = get_llm_response(
            prompt=f"Question: {question}\nAnswer:",
            system_prompt=system_prompt,
            model=self.answer_model,
            max_tokens=50,
            temperature=self.temperature,
            provider=self.provider,
            verbose=self.verbose,
        )

        answer = self._parse_yes_no_answer(raw_answer) if self.domain == "20q" else raw_answer

        if is_multiple_choice_question(question):
            answer_code = 0.5
        elif answer.lower().startswith("yes"):
            answer_code = 1
        elif answer.lower().startswith("maybe"):
            answer_code = 0.5
        else:
            answer_code = 0
        return answer, answer_code, is_multiple_choice_question(question)

    @staticmethod
    def _parse_yes_no_answer(raw_answer):
        raw = raw_answer.lower().strip()
        if raw.startswith("yes"):
            return "Yes"
        if raw.startswith("no"):
            return "No"
        if raw.startswith("maybe"):
            return "Maybe"
        if "maybe" in raw or "uncertain" in raw or "not sure" in raw:
            return "Maybe"
        if "yes" in raw and "no" not in raw:
            return "Yes"
        if "no" in raw and "yes" not in raw:
            return "No"
        return "Maybe"

    def check_guess(self, guess):
        return bool(guess) and check_guess_correctness(guess, self.subject)

    def update_conversation_history(self, question, answer):
        question = question.strip().replace("Q:", "").strip()
        answer = answer.strip().replace("A:", "").replace("Answer:", "").strip()
        self.conversation_history += f"Question {self.question_count}: {question}\n"
        self.conversation_history += f"Answer: {answer}\n"

    def force_guess(self, conv_history=None, get_logits=False):
        """Force a final guess (+confidence) from the model given the conversation."""
        if get_logits:
            raise NotImplementedError("Logit-based guessing is disabled in this release.")
        if conv_history is None:
            conv_history = self.conversation_history
        prompt = self.final_guess_request.format(conversation_history=conv_history)
        raw_guess = get_llm_response(
            prompt=prompt,
            system_prompt=self.force_guess_prompt,
            model=self.guess_model,
            max_tokens=2000,
            temperature=0,
            provider=self.provider,
            verbose=self.verbose,
        )
        guess, confidence, cleaned_response = parse_guess(raw_guess)
        return guess, confidence, cleaned_response, None

    # ----------------------------------------------------------------- logging
    def format_turn_result(self, question, answer, guess=None, confidence=0,
                           raw_guess="NA", is_multiple_choice=False, success=False, **extra):
        result = {
            "subject": self.subject,
            "question": question,
            "answer": answer,
            "question_count": self.question_count,
            "success": success,
            "guess": guess if guess else "unknown",
            "confidence": confidence,
            "raw_guess": raw_guess,
            "is_multiple_choice": is_multiple_choice,
        }
        result.update(extra)
        return result

    def log_turn(self, turn_result):
        self.turn_results.append(turn_result)
        if self.jsonl_log_path:
            with open(self.jsonl_log_path, "a") as f:
                f.write(json.dumps(turn_result) + "\n")

    def save_game_results(self, strategy_name):
        if not self.log_subdir:
            return None
        game_result = {
            "subject": self.subject,
            "question_count": self.question_count,
            "success": any(r["success"] for r in self.turn_results),
            "strategy": strategy_name,
            "domain": self.domain,
            "question_type": self.question_type,
            "answer_set": self.animal_set,
            "answer_set_size": len(self.answer_set),
        }
        save_json_data(game_result, os.path.join(self.log_subdir, f"{self.exp_id}_summary.json"))
        return game_result
