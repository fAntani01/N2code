# rf_lab/core/optimizers/spsa.py

from __future__ import annotations

import logging
import random
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("MeasurementSystem")


class SPSAOptimizer:
    def __init__(self, objective_fn: Callable[[list[float], Optional[str]], float], x0: list[float], bounds: tuple[float, float], a = 1.0, c = 0.1, A = 10, alpha = 0.602, gamma = 0.101):
        """
        Args:
            objective_fn:   f(x) -> float, the function to maximize.
            x0:             Initial parameter vector.
            bounds:         [(min, max)] per parameter, used to clip updates.
            algo_params:    Algorithm hyperparameters:
                                a      : controls initial step size
                                c      : controls initial perturbation size
                                A      : stability constant (typically 10% of max iterations)
                                alpha  : step size decay exponent (default 0.602)
                                gamma  : perturbation decay exponent (default 0.101)
        """
        self.objective_fn = objective_fn
        self.x = np.array(x0, dtype=float)
        self.bounds = bounds
        self.n_params = len(self.x)
        self.best_x = self.x.copy()
        self.best_score = float("-inf")
        self.k = -1  # iteration counter
        self.rng = random.Random()

        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma

    def name(self):
        return "SPSA"


    def _gain_a(self) -> float:
        """Step size at iteration k."""
        return self.a / (self.k + 1 + self.A) ** self.alpha

    def _gain_c(self) -> float:
        """Perturbation size at iteration k."""
        return self.c / (self.k + 1) ** self.gamma

    def _clip(self, x: np.ndarray) -> np.ndarray:
        """Clip x to stay within bounds."""
        low = self.bounds[0]
        high = self.bounds[1]
        return np.clip(x, low, high)

    def step(self) -> tuple[list[float], float, dict]:

        if self.k == -1:

            self.best_score = self.objective_fn(self.x.tolist())
            self.k += 1
            return self.x, self.best_score, {}


        ak = self._gain_a()
        ck = self._gain_c()

        # 1. Random ±1 perturbation vector (Rademacher distribution)
        delta = np.array([self.rng.choice([-1.0, 1.0]) for _ in range(self.n_params)])

        # 2. Two evaluations at symmetrically perturbed points
        x_plus = self._clip(self.x + ck * delta)
        x_minus = self._clip(self.x - ck * delta)

        f_plus = self.objective_fn(x_plus.tolist(), subfolder="plus")
        f_minus = self.objective_fn(x_minus.tolist(), subfolder="minus")

        # 3. Simultaneous gradient estimate
        gradient = (f_plus - f_minus) / (2 * ck * delta)

        # 4. Update — gradient ascent (maximizing)
        self.x = self._clip(self.x + ak * gradient)

        # Track best seen so far (based on f_plus as a proxy)
        current_score = max(f_plus, f_minus)
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_x = self.x.copy()

        self.k += 1

        optimizer_state = {
            "ak": round(ak, 6),
            "ck": round(ck, 6),
            "f_plus": round(f_plus, 6),
            "f_minus": round(f_minus, 6),
            "gradient_norm": round(float(np.linalg.norm(gradient).tolist()), 6),
            "best_score": round(self.best_score, 6),
        }

        logger.info(f"[SPSA] iter {self.k} | f+: {f_plus:.4f} f-: {f_minus:.4f} | ak: {ak:.4f} ck: {ck:.4f} | grad norm: {optimizer_state['gradient_norm']:.4f}")

        return self.x, current_score, optimizer_state
