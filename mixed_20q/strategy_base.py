"""Shared base class for question-asking strategies."""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class TwentyQuestionsStrategy(ABC):
    """Owns the play loop; subclasses implement ``play_turn``."""

    def __init__(self, env, **kwargs):
        self.env = env
        self.name = self.__class__.__name__

    @abstractmethod
    def play_turn(self) -> Tuple[bool, bool, Optional[str]]:
        """Play one turn; return (done, success, guess)."""

    def play_game(self) -> Dict:
        self.env.reset()
        done = False
        while not done:
            done, success, guess = self.play_turn()
        return self.env.save_game_results(self.name)
