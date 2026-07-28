"""
Caricamento di misure salvate su disco (Single Measurement, Training,
Exhaustive Search) e ricerca dell'obiettivo migliore in un dataset.

Riusa la classe MeasurementData del programma di acquisizione, cosi' il
codice di analisi (accesso a .vna_results, .freq_axis, ecc.) e' identico
a quello usato nella GUI.

FORMATO SU DISCO (vedi MeasurementStorage.save_to_folder in storage.py):

    <folder>/
        params.json   # timestamp, applied_voltages, applied_field,
                       # measured_field_vector_mt, measurement_parameters,
                       # setup_parameters
        data.csv       # colonne: Frequency_Hz, e per ogni parametro S
                       # selezionato: {param}_gated_real, {param}_gated_imag,
                       # {param}_raw_real, {param}_raw_imag
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
from collections.abc import Callable

import numpy as np
import pandas as pd

from core.measurement_data import MeasurementData


# ----------------------------------------------------------------------
# CARICAMENTO
# ----------------------------------------------------------------------
def load_measurement(folder) -> MeasurementData:
    """Ricostruisce un oggetto MeasurementData da una cartella salvata da
    MeasurementStorage.save_to_folder() (vedi storage.py).

    setup_parameters/measurement_parameters vengono ricreati come
    SimpleNamespace invece che istanze "vere" delle classi originali: in
    analisi servono solo in lettura (es. data.measurement_parameters.
    vna_parameters), quindi non serve importare/costruire le classi
    complete del programma di acquisizione. to_dict() le rende comunque
    gia' dizionari semplici in params.json, quindi il mapping e' diretto.
    """
    folder = Path(folder)

    with open(folder / "params.json") as f:
        params = json.load(f)

    measurement_parameters = SimpleNamespace(**params["measurement_parameters"])
    setup_parameters = SimpleNamespace(**params["setup_parameters"])

    df = pd.read_csv(folder / "data.csv")
    freq_axis = df["Frequency_Hz"].to_numpy()

    vna_results = {}
    vna_results_raw = {}
    for trace in measurement_parameters.vna_parameters:
        vna_results[trace] = df[f"{trace}_gated_real"].to_numpy() + 1j * df[f"{trace}_gated_imag"].to_numpy()
        vna_results_raw[trace] = df[f"{trace}_raw_real"].to_numpy() + 1j * df[f"{trace}_raw_imag"].to_numpy()

    data = MeasurementData(
        applied_voltages=params.get("applied_voltages"),
        measured_field_vector=params.get("measured_field_vector_mt", []),
        vna_results=vna_results,
        vna_results_raw=vna_results_raw,
        freq_axis=freq_axis,
        setup_parameters=setup_parameters,
        measurement_parameters=measurement_parameters,
        applied_magnetic_field=params.get("applied_field"),
    )
    if "timestamp" in params:
        data.timestamp = params["timestamp"]

    # Comodo in analisi per risalire alla cartella/etichettare i plot,
    # anche se non fa parte dei campi originali di MeasurementData.
    data.folder = folder

    return data


def load_measurements(root_folder, pattern: str = "iter*") -> List[MeasurementData]:
    """Carica tutte le misure sotto root_folder le cui sottocartelle
    matchano pattern (es. le iterazioni "iter0", "iter1", ... di una
    sessione di Training o Exhaustive Search). Ordina per nome cartella,
    quindi per numero di iterazione se il pattern e' "iter*".

    Cartelle che non contengono i file attesi vengono ignorate con un
    avviso, invece di far fallire l'intero caricamento.
    """
    root_folder = Path(root_folder)
    folders = sorted(root_folder.glob(pattern), key=lambda p: p.name)

    measurements = []
    for folder in folders:
        try:
            measurements.append(load_measurement(folder))
        except (FileNotFoundError, KeyError) as e:
            print(f"[load_measurements] Skipping {folder}: {e}")

    return measurements


# ----------------------------------------------------------------------
# RICERCA DELL'OBIETTIVO MIGLIORE IN UN DATASET
# ----------------------------------------------------------------------
def search_best_objective(
    measurements: List[MeasurementData],
    objective_fn: Callable[..., float],
    param_grid: Optional[Dict[str, List]] = None,
) -> List[dict]:
    """Per ogni misura, valuta objective_fn su tutte le combinazioni di
    param_grid e tiene il punteggio migliore ottenuto (con quali
    parametri). Ritorna la lista dei risultati ordinata per score
    decrescente (il primo elemento e' la misura/combinazione migliore).

    objective_fn: objective_fn(data: MeasurementData, **params) -> float
    param_grid: dict {nome_parametro: [valori...]} di parametri liberi su
        cui scandire ogni misura (es. {"notch_freq": np.arange(2.0, 2.5, 0.01)}).
        None o {} -> objective_fn viene chiamata una sola volta per misura,
        senza parametri liberi.

    Esempio di risultato:
        [{"measurement": <MeasurementData>, "score": 42.1, "params": {"notch_freq": 2.14}}, ...]
    """
    param_grid = param_grid or {}
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values())) if keys else [()]

    results = []
    for data in measurements:
        best = None
        for combo in combos:
            params = dict(zip(keys, combo))
            score = objective_fn(data, **params)
            if best is None or score > best["score"]:
                best = {"measurement": data, "score": score, "params": params}
        results.append(best)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def magnitude_db(data: MeasurementData, trace: str, gated: bool = True) -> np.ndarray:
    """Helper comune per gli objective: |trace| in dB, dalla variante
    gated o raw. Solleva KeyError se trace non e' presente (misura in cui
    quel parametro S non era selezionato)."""
    source = data.vna_results if gated else data.vna_results_raw
    return 20 * np.log10(np.abs(source[trace]) + 1e-30)


# Esempio di objective_fn con un parametro libero (frequenza del notch),
# nello spirito delle celle "NOTCH FILTER" gia' presenti nel notebook.
# Punteggio piu' alto = notch piu' profondo attorno a notch_freq.
def notch_depth_objective(data: MeasurementData, notch_freq: float, bandwidth: float = 0.01, trace: str = "S21", gated: bool = True) -> float:
    freq_ghz = data.freq_axis / 1e9
    mag_db = magnitude_db(data, trace, gated=gated)

    mask = np.abs(freq_ghz - notch_freq) <= bandwidth / 2
    if not np.any(mask):
        return -np.inf

    return -np.min(mag_db[mask])
