from __future__ import annotations

import itertools
import logging
import numpy as np
from typing import Callable, Optional

logger = logging.getLogger("MeasurementSystem")


class ExhaustiveSearch:
    def __init__(self, objective_fn: Callable[[list[float], Optional[str]], float], x0: list[float], values_set: list[float], fixed_indices: Optional[list[int]] = None):
        """
        x0: valore di partenza per ogni canale. Per i canali spazzati
            (non in fixed_indices) x0 non ha alcun effetto sulla ricerca
            (viene sovrascritto dalle combinazioni di values_set); per i
            canali fissi, x0 e' invece il valore che resta invariato per
            tutta la ricerca.
        values_set: valori da combinare esaustivamente per ogni canale
            spazzato (prodotto cartesiano).
        fixed_indices: indici (in x0) dei canali da NON includere nella
            combinatoria: restano fissi al loro valore in x0. None o []
            -> tutti i canali vengono spazzati (comportamento precedente).
        """
        self.objective_fn = objective_fn
        self.x0 = list(x0)
        self.values_set = list(values_set)
        self.n_params = len(self.x0)

        self.fixed_indices = set(fixed_indices or [])
        if not self.fixed_indices.issubset(range(self.n_params)):
            raise ValueError(f"fixed_indices {sorted(self.fixed_indices)} out of range for x0 of length {self.n_params}")

        # Indici spazzati: tutti gli altri, in ordine crescente. Solo
        # questi entrano nel prodotto cartesiano; i fissi restano al
        # valore letto da x0 (vedi step()).
        self.swept_indices = [i for i in range(self.n_params) if i not in self.fixed_indices]

        self.tested_config = 0
        self.test_configs, self.n_configs = self._create_test_configs(self.swept_indices, self.values_set)

    def name(self) -> str:
        return "Exhaustive Search"

    def _create_test_configs(self, swept_indices: list[int], values_set: list[float]) -> tuple[np.ndarray, int]:
        """Genera il prodotto cartesiano di values_set per i soli canali
        spazzati. Se swept_indices e' vuoto (tutti i canali fissi),
        produce una singola configurazione (nessuna combinazione da
        testare oltre a x0 stesso).

        Esempio:
            values_set = [0.0, 1.0], swept_indices = [0, 2] (canale 1 fisso)
            -> test_configs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
               (2 colonne, una per ciascun indice spazzato, non per canale)
        """
        n_swept = len(swept_indices)
        raw_configs = list(itertools.product(values_set, repeat=n_swept))

        test_configs = np.array(raw_configs, dtype=float).reshape(len(raw_configs), n_swept)
        n_configs = len(test_configs)

        return test_configs, n_configs

    def step(self) -> tuple[list[float], float, dict]:
        # Gestione fine griglia (evita IndexError se n_iter supera n_configs)
        if self.tested_config >= self.n_configs:
            logger.warning("[Exhaustive Search] All combinations have been tested.")
            return [], 0, {}

        # Configurazione completa: parte da x0 (i canali fissi restano
        # cosi'), poi sovrascrive solo le posizioni spazzate con la
        # combinazione corrente.
        full_config = list(self.x0)
        swept_values = self.test_configs[self.tested_config].tolist()
        for idx, value in zip(self.swept_indices, swept_values):
            full_config[idx] = value

        # Esegui la misura e ottieni il punteggio
        new_score = self.objective_fn(full_config)

        # Incrementa il contatore delle configurazioni testate
        self.tested_config += 1

        optimizer_state = {"progress": f"{self.tested_config}/{self.n_configs}", "config_idx": self.tested_config}

        return full_config, new_score, optimizer_state
