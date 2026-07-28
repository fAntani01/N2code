from core.measurement_data import MeasurementData
import numpy as np

def _objective_maximise_s31(measurement_data: MeasurementData) -> float:
    """Massimizza |S31| medio in banda 4.3-4.4 GHz (banda hardcoded qui)."""
    freq = measurement_data.freq_axis
    mask = (freq >= 4.3e9) & (freq <= 4.4e9)
    return float(np.mean(np.abs(measurement_data.vna_results["S31"][mask])))


def _objective_maximise_s21(measurement_data: MeasurementData) -> float:
    """Massimizza |S21| medio in banda 4.3-4.4 GHz (banda hardcoded qui)."""
    freq = measurement_data.freq_axis
    mask = (freq >= 1.1e9) & (freq <= 1.5e9)
    return float(np.mean(np.abs(measurement_data.vna_results["S21"][mask])))


def _objective_minimise_s21(measurement_data: MeasurementData) -> float:
    """Minimizza |S21| medio in banda 4.3-4.4 GHz (banda hardcoded qui)."""
    return -_objective_maximise_s21(measurement_data)


# Ogni obiettivo: nome (= testo mostrato in train_combo_objective_2),
# descrizione (per un tooltip o simile) e la funzione da chiamare come
# objective(measurement_data) -> float. Nessun parametro configurabile
# dalla GUI: se serve cambiare la banda, si modifica qui nel codice.
OBJECTIVE_FUNCTIONS = {
    "Maximise |S31| in band": {
        "description": "Averages |S31| magnitude over a fixed 4.3-4.4 GHz band.",
        "function": _objective_maximise_s31,
    },
    "Maximise |S21| in band": {
        "description": "Averages |S21| magnitude over a fixed 4.3-4.4 GHz band.",
        "function": _objective_maximise_s21,
    },
    "Minimise |S21| in band": {
        "description": "Negative of averaged |S21| magnitude over a fixed 4.3-4.4 GHz band.",
        "function": _objective_minimise_s21,
    },
}