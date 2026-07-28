# rf_lab/core/optimizers/direct_search.py

from __future__ import annotations

import logging
import random
from typing import Callable, Optional

logger = logging.getLogger("MeasurementSystem")


class DirectSearchOptimizer:
    def __init__(self, objective_fn: Callable[[list[float], Optional[str]], float], x0: list[float], values_set: list[float]):
        
        self.objective_fn = objective_fn
        self.x = list(x0)
        self.values_set = list(values_set)
        self.rng = random.Random()
        self.n_params = len(self.x)
        self.best_score = float("-inf")

        self.iteration = 0
        self.tested_params = 0
        self.order = self._new_order()

    def name(self):
        return "Direct Search"

    def _new_order(self) -> list[int]:
        order = list(range(self.n_params))
        self.rng.shuffle(order)
        return order

    def step(self) -> tuple[list[float], float, dict]:


        if self.iteration == 0:
            self.best_score = self.objective_fn(self.x)
            self.iteration += 1
            return self.x, self.best_score, {}
        

        # Which channel are we testing this step?
        current_channel = self.order[self.tested_params]
        current_value = self.x[current_channel]

        # Pick a random alternative value for that channel
        candidates = [v for v in self.values_set if v != current_value]
        proposed_value = self.rng.choice(candidates)

        # Build and evaluate the candidate configuration
        test_configuration = self.x.copy()
        test_configuration[current_channel] = proposed_value
        new_score = self.objective_fn(test_configuration)

        if new_score > self.best_score:
            self.x = test_configuration
            self.best_score = new_score
            logger.info(f"[DirectSearch] Improved — channel {current_channel}: {current_value} -> {proposed_value} | score: {new_score:.6f}")
        else:
            logger.info(f"[DirectSearch] No improvement on channel {current_channel} | score: {new_score:.6f}")

        # Advance to the next channel; start a new round when all have been tested
        self.tested_params += 1
        if self.tested_params >= self.n_params:
            self.tested_params = 0
            self.iteration += 1
            self.order = self._new_order()
            logger.info("[DirectSearch] Round complete. Starting new round with shuffled order.")

        return self.x, self.best_score, {"tested_config": list(test_configuration), "current_channel": current_channel, "round_progress": f"{self.tested_params}/{self.n_params}"}



