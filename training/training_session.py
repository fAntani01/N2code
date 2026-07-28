from __future__ import annotations

import json
import logging
import os
import threading
from typing import Callable, Optional

import pandas as pd

from core.controller_fake import MeasurementController
from core.measurement_data import MeasurementData

logger = logging.getLogger("MeasurementSystem")


class TrainingSession:
    def __init__(
        self,
        controller: MeasurementController,
        score_fn: Callable[[MeasurementData], float],
        magnetic_field: Optional[float] = None,
        session_folder: str = "trainings",
        on_iteration_end: Optional[Callable[[int, MeasurementData, list[dict], dict], None]] = None,  # (Iteration, MeasurementData, history, optimizer_state) -> None
    ):
        self.controller = controller
        self.score_fn = score_fn
        self.magnetic_field = magnetic_field
        self.on_iteration_end = on_iteration_end

        self.history: list[dict] = []
        self.iteration = 0
        self._last_data: MeasurementData | None = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.is_running = False

        self.session_folder = session_folder

        os.makedirs(session_folder, exist_ok=True)


    # ------------------------------------------------------------------ #
    # objective — the only function the optimizer calls                    #
    # ------------------------------------------------------------------ #

    def _objective(self, x: list[float], subfolder: str = "") -> float:
        """Measure and score. Stashes data for _run_loop to pick up —
        history logging happens there, once the optimizer state is known."""

        iter_name = f"iter{self.iteration}"
        
        output_folder = os.path.join(self.session_folder, iter_name)

        if subfolder:
            output_folder = os.path.join(output_folder, subfolder)

        self._last_data = self.controller.measure(voltages=x, magnetic_field=self.magnetic_field, output_folder=output_folder)

        return self.score_fn(self._last_data)

    # ------------------------------------------------------------------ #
    # loop control                                                         #
    # ------------------------------------------------------------------ #

    def step(self):
        """Run one optimizer iteration (blocking). Returns (x, score, optimizer_state)."""
        new_values, score, optimizer_state = self.optimizer.step()

        entry = {"iter": self.iteration, "score": score, "voltages": {f"ch{self.controller.setup_params.daq_channels[i]}": v for i, v in enumerate(new_values)}, "optimizer_state": optimizer_state}
        self.history.append(entry)

        self._save_iteration_json(entry)
        self._update_summary_csv()

        logger.info(f"[Training] iter {self.iteration:04d} | score: {score:.6f} | optimizer: {optimizer_state}")

        return self.iteration, score, self._last_data, self.history, optimizer_state

    def run(self, n_iter: int) -> None:
        """Run n_iter steps to completion, blocking."""
        self._run_loop(n_iter)

    def start(self, n_iter: int) -> None:
        """Start the loop in a background thread, returning immediately."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, args=(n_iter,), daemon=True)
        self._thread.start()
        self.is_running = True

    def stop(self) -> None:
        """Ask the loop to stop after the current iteration finishes."""
        self._stop_event.set()

    def join(self) -> None:
        """Block until the background thread finishes."""
        if self._thread:
            self._thread.join()

    def _run_loop(self, n_iter: int) -> None:
        try:
            for _ in range(n_iter):
                if self._stop_event.is_set():
                    logger.info(f"[{"Training"}] Training stopped early at iteration {self.iteration}.")
                    break

                new_values, score, optimizer_state = self.optimizer.step()

                entry = {"iter": self.iteration, "score": score, "voltages": {f"ch{self.controller.setup_params.daq_channels[i]}": v for i, v in enumerate(new_values)}, "optimizer_state": optimizer_state}
                self.history.append(entry)

                self._save_iteration_json(entry)
                self._update_summary_csv()

                logger.info(f"[Training] iter {self.iteration:04d} | score: {score:.6f} | optimizer: {optimizer_state}")

                if self.on_iteration_end is not None:
                    self.on_iteration_end(self.iteration, score, self._last_data, self.history, optimizer_state)

                self.iteration += 1

        except Exception as e:
            logger.error(f"[{"Training"}] Training loop raised an exception: {e}", exc_info=True)
        finally:
            logger.info(f"[{"Training"}] Training finished — {self.iteration} iterations completed.")
            self.is_running = False


    # ------------------------------------------------------------------ #
    # saving                                                               #
    # ------------------------------------------------------------------ #

    def _save_iteration_json(self, entry: dict) -> None:
        """Save a JSON with iter, score, voltages and optimizer state
        into the iteration's own subfolder (already created by measure_point)."""
        iter_name = f"iter{self.iteration}"
        iter_folder = os.path.join(self.session_folder, iter_name)
        path = os.path.join(iter_folder, "iteration.json")
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)

    def _update_summary_csv(self) -> None:
        """Rewrite the summary CSV in the session folder after every iteration,
        so it's always up to date even if the run is stopped early."""
        rows = []
        for entry in self.history:
            row = {
                "iter": entry["iter"],
                "score": entry["score"],
                **entry["voltages"],  # ch0, ch1, ... as flat columns
                **(entry["optimizer_state"] or {}),  # step_size, temperature, ... as flat columns
                "optimizer_name" : self.optimizer.name(),
            }
            rows.append(row)

        path = os.path.join(self.session_folder, "summary.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
