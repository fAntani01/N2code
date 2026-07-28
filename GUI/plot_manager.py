"""
Gestione di tutti i plot dell'applicazione: spettro Single Measurement (con
trace control a checkbox, raw + time-gated), spettro Training, spettro
Exhaustive Search, e andamento del training (objective + voltaggi) in
funzione dell'iterazione.

Una sola classe PlotManager, inizializzata dal MainWindow passando i
container su cui costruire ciascun plot. Nessuna logica applicativa qui
dentro (niente controller, niente lettura widget della GUI): PlotManager
riceve solo MeasurementData e numeri gia' pronti.
"""

import logging
from typing import List, Optional, Dict, Tuple, Callable

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QScrollArea, QCheckBox, QLabel

from core.measurement_data import MeasurementData

logger = logging.getLogger("MeasurementSystem")

# ----------------------------------------------------------------------
# COLORI E STILI
# ----------------------------------------------------------------------
# Un hue fisso per parametro S: tutte le tracce di S21, di qualunque
# misura/iterazione, condividono la stessa tonalita' di base. La "shade"
# (luminosita') distingue invece misure/iterazioni diverse dello stesso
# parametro: la prima e' la piu' scura, le successive via via piu'
# chiare, riciclando se ce ne sono piu' di quante shade definite.
_S_PARAM_HUES = {"S11": 210, "S21": 30, "S31": 130, "S41": 0}  # blu, arancio, verde, rosso
# 6 valori: indice 0 riservato alla prima iterazione (fissa), indici 1..5
# usati per le 5 posizioni della finestra scorrevole (piu' vecchia ->
# piu' recente). Con esattamente una shade per posizione, le tracce
# mostrate contemporaneamente sono sempre ben distinguibili tra loro.
_SHADE_LIGHTNESS = [90, 115, 140, 165, 190, 215]  # scuro -> chiaro, scala 0-255 di QColor.fromHsl

# Palette qualitativa "classica", per curve che non sono parametri S
# (objective, voltaggi per canale in Training).
_QUALITATIVE_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
]

# Stile linea: raw sempre continua, gated sempre tratteggiata, indipendente
# dal colore (che codifica invece parametro S + shade).
_LINESTYLE_RAW = Qt.PenStyle.SolidLine
_LINESTYLE_GATED = Qt.PenStyle.DashLine


def _trace_color(trace: str, series_index: int) -> QColor:
    hue = _S_PARAM_HUES.get(trace, 0)
    lightness = _SHADE_LIGHTNESS[series_index % len(_SHADE_LIGHTNESS)]
    return QColor.fromHsl(hue, 200, lightness)


_ALL_S_PARAMETERS = ["S11", "S21", "S31", "S41"]

# Quante iterazioni recenti restano sempre visibili negli spettri di
# Training/Exhaustive Search, oltre alla prima (sempre fissa).
_SPECTRUM_ROLLING_WINDOW = 5

# Controllo grafico globale log/lineare, condiviso da Single Measurement,
# Training e Exhaustive Search.
SCALE_LOG = "log"
SCALE_LINEAR = "linear"


# ----------------------------------------------------------------------
# FILTRI DI SELEZIONE ITERAZIONE, PER ALGORITMO (solo Training)
# ----------------------------------------------------------------------
# Stesso pattern di ALGORITHM_PARAMS/OBJECTIVE_FUNCTIONS: un dizionario
# algo_name -> filter_fn. Un'iterazione la cui firma ritorna False viene
# SALTATA COMPLETAMENTE: niente spettro, niente punto in objective/
# voltaggi. La prima iterazione (iteration == 0) e' SEMPRE plottata,
# indipendentemente dal filtro, e non passa da qui.
#
# Firma: filter_fn(iteration: int, score: float, history: list[dict]) -> bool
def _direct_search_spectrum_filter(iteration: int, score: float, history: list) -> bool:

    return True

    if len(history) < 2:
        return False
    return score > history[-2]["score"]


TRAINING_SPECTRUM_FILTERS: Dict[str, Callable[[int, float, list], bool]] = {
    "Direct Search": _direct_search_spectrum_filter,
    "SPSA": lambda iteration, score, history: True,
    "Optuna (TPE)": lambda iteration, score, history: True,
}


def _make_frequency_plot(container: QWidget) -> Tuple[pg.PlotWidget, pg.LegendItem]:
    """Crea un pyqtgraph.PlotWidget con asse frequenza dentro container, e
    ne ritorna il widget e la legenda. L'etichetta dell'asse Y viene
    impostata separatamente da PlotManager (dipende dalla scala log/
    lineare corrente, che e' un controllo globale).
    """
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    plot_widget = pg.PlotWidget()
    plot_widget.setLabel("bottom", "Frequency", units="Hz")
    plot_widget.showGrid(x=True, y=True, alpha=0.3)
    legend = plot_widget.addLegend()
    layout.addWidget(plot_widget)

    return plot_widget, legend


class _SpectrumPanel:
    """Un plot spettro con politica "prima iterazione fissa + finestra
    scorrevole delle ultime N". Usato dallo spettro di Training. Chi usa
    questa classe ha gia' deciso a monte se l'iterazione va disegnata
    (vedi il filtro per algoritmo in TRAINING_SPECTRUM_FILTERS).
    """

    def __init__(self, plot_widget: pg.PlotWidget, legend: pg.LegendItem, magnitude_fn: Callable[[np.ndarray], np.ndarray]):
        self.plot_widget = plot_widget
        self.legend = legend
        self._magnitude_fn = magnitude_fn  # bound method di PlotManager: legge sempre la scala corrente
        self.reset()

    def reset(self):
        self.plot_widget.clear()
        self.legend.clear()
        self._entries: Dict[int, dict] = {}  # iteration -> {"data", "curves"}
        self._rolling: List[int] = []  # iterazioni non-prima attualmente visibili, FIFO

    def add(self, iteration: int, data: MeasurementData, is_first: bool):
        self._entries[iteration] = {"data": data, "curves": []}

        if is_first:
            self.redraw()
            return

        self._rolling.append(iteration)
        while len(self._rolling) > _SPECTRUM_ROLLING_WINDOW:
            oldest_iteration = self._rolling.pop(0)
            self._entries.pop(oldest_iteration, None)  # le curve vengono spazzate via dal clear() in redraw()

        self.redraw()

    def redraw(self):
        """Ridisegna tutto cio' che e' attualmente visibile: la prima
        iterazione (shade fissa, indice 0) e le tracce della finestra
        scorrevole, con la shade calcolata dalla loro POSIZIONE CORRENTE
        nella finestra (indice 1 = piu' vecchia visibile, ... 5 = piu'
        recente) — non da un contatore che continua a crescere. Cosi' le
        tracce mostrate hanno sempre shade ben distinte tra loro, anche a
        finestra piena, invece di convergere tutte sulla stessa shade
        "stabile". Va richiamato ogni volta che cambia cosa e' visibile
        (nuova iterazione, o cambio scala log/lineare).
        """
        self.plot_widget.clear()
        self.legend.clear()

        if 0 in self._entries:
            entry = self._entries[0]
            entry["curves"] = self._draw(0, entry["data"], shade_index=0)

        for position, iteration in enumerate(self._rolling):
            entry = self._entries[iteration]
            entry["curves"] = self._draw(iteration, entry["data"], shade_index=1 + position)

    def _draw(self, iteration: int, data: MeasurementData, shade_index: int) -> list:
        curves = []
        for trace in data.measurement_parameters.vna_parameters:
            if trace not in data.vna_results:
                logger.warning(f"[Plot] Trace '{trace}' not found, skipping.")
                continue
            magnitude = self._magnitude_fn(data.vna_results[trace])
            curve = self.plot_widget.plot(
                data.freq_axis, magnitude,
                pen=pg.mkPen(color=_trace_color(trace, shade_index), width=2),
                name=f"iter {iteration} - {trace}",
            )
            curves.append(curve)
        return curves


class PlotManager:

    def __init__(
        self,
        single_plot_container: QWidget,
        single_trace_control_container: QWidget,
        train_spectrum_container: QWidget,
        train_objective_container: QWidget,
        train_voltages_container: QWidget,
    ):
        # Controllo grafico globale: log (dB) o lineare, condiviso da
        # TUTTI i plot di ampiezza (Single Measurement, Training).
        self._scale_mode = SCALE_LOG

        self._init_single_measurement(single_plot_container, single_trace_control_container)
        self._init_train_spectrum(train_spectrum_container)
        self._init_train_objective(train_objective_container)
        self._init_train_voltages(train_voltages_container)

    # ================================================================
    # SCALA LOG/LINEARE (globale)
    # ================================================================
    def _magnitude(self, complex_trace: np.ndarray) -> np.ndarray:
        if self._scale_mode == SCALE_LOG:
            return 20 * np.log10(np.abs(complex_trace) + 1e-30)
        return np.abs(complex_trace)

    def set_scale_mode(self, mode: str):
        """Cambia la scala di TUTTI i plot di ampiezza (Single
        Measurement, Training, Exhaustive Search) e ridisegna cio' che e'
        gia' a schermo con i nuovi valori Y. mode: SCALE_LOG o SCALE_LINEAR.
        """
        if mode not in (SCALE_LOG, SCALE_LINEAR):
            raise ValueError(f"Unknown scale mode: {mode!r}")
        self._scale_mode = mode

        if mode == SCALE_LOG:
            self.single_plot_widget.setLabel("left", "Magnitude", units="dB")
            self.train_spectrum_widget.setLabel("left", "Magnitude", units="dB")
        else:
            self.single_plot_widget.setLabel("left", "Magnitude")
            self.train_spectrum_widget.setLabel("left", "Magnitude")

        self._refresh_single_plot()
        self._train_spectrum_panel.redraw()

    # ================================================================
    # SINGLE MEASUREMENT: spettro + trace control a checkbox (raw + gated)
    # ================================================================
    def _init_single_measurement(self, plot_container: QWidget, trace_control_container: QWidget):
        self.single_plot_widget, self.single_legend = _make_frequency_plot(plot_container)
        self.single_plot_widget.setLabel("left", "Magnitude", units="dB")

        # Tabella a griglia: colonna 0 = nome misura, poi una coppia
        # (raw, gated) per ciascun parametro S, sempre nello stesso
        # ordine (vedi _ALL_S_PARAMETERS) cosi' le checkbox restano
        # allineate riga per riga. Riga 0 riservata all'header.
        layout = QVBoxLayout(trace_control_container)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_contents = QWidget()
        self._trace_grid = QGridLayout(scroll_contents)
        self._trace_grid.setContentsMargins(4, 4, 4, 4)

        # Colonne fisse: [(trace, variant), ...]. Ogni misura ha sempre
        # entrambe le varianti disponibili (raw e gated), quindi non
        # serve un controllo di disponibilita' separato da quello sul
        # parametro S stesso (misurato o no).
        self._trace_columns: List[Tuple[str, str]] = []
        for trace in _ALL_S_PARAMETERS:
            self._trace_columns.append((trace, "raw"))
            self._trace_columns.append((trace, "gated"))

        header_labels = ["Measurement"] + [f"{trace} gated" if variant == "gated" else trace for trace, variant in self._trace_columns]
        for col, text in enumerate(header_labels):
            header = QLabel(f"<b>{text}</b>")
            self._trace_grid.addWidget(header, 0, col)

        scroll.setWidget(scroll_contents)
        layout.addWidget(scroll)

        # Prossima riga libera nella griglia (riga 0 = header, quindi si
        # parte da 1). Le righe non vengono mai rimosse singolarmente,
        # solo tutte insieme da clear_session().
        self._next_trace_row = 1

        # Cartella della sessione correntemente mostrata nel trace control.
        # Quando cambia, la sessione precedente viene azzerata.
        self.current_session_folder: Optional[str] = None

        # label -> {"data", "row_widgets", "row_checkbox", "trace_checkboxes"}
        self._measurements: Dict[str, dict] = {}

    def register_measurement(self, measurement_data, label: str, output_folder: Optional[str] = None):
        """Registra una misura appena eseguita: aggiunge una riga alla
        tabella del trace control (nome + checkbox raw/gated per ognuno
        dei 4 parametri S, abilitate solo per i parametri effettivamente
        misurati) e aggiorna lo spettro secondo lo stato corrente delle
        checkbox.

        I parametri S disponibili si leggono da
        measurement_data.measurement_parameters.vna_parameters. Raw e
        gated sono invece sempre entrambi disponibili per ogni parametro
        misurato (vedi MeasurementData.vna_results_raw / .vna_results).
        """
        if output_folder is not None and output_folder != self.current_session_folder:
            self.clear_session()
            self.current_session_folder = output_folder

        measured_traces = set(measurement_data.measurement_parameters.vna_parameters)

        if label in self._measurements:
            logger.warning(f"[Plot] Measurement '{label}' already registered in this session, overwriting data.")
            self._measurements[label]["data"] = measurement_data
            self._refresh_single_plot()
            return

        row_widgets, row_checkbox, trace_checkboxes = self._build_trace_row(label, measured_traces)

        self._measurements[label] = {
            "data": measurement_data,
            "row_widgets": row_widgets,
            "row_checkbox": row_checkbox,
            "trace_checkboxes": trace_checkboxes,
        }

        self._refresh_single_plot()

    def _build_trace_row(self, label: str, measured_traces: set):
        """Crea una riga della tabella: checkbox nome misura in colonna 0,
        una checkbox per ognuna delle varianti (trace, raw/gated) nelle
        colonne successive (sempre presenti, ma abilitate solo se il
        parametro S e' stato davvero misurato). Ogni toggle ricalcola
        l'intero plot.
        """
        row = self._next_trace_row
        self._next_trace_row += 1
        row_widgets = []

        row_checkbox = QCheckBox(label)
        row_checkbox.setChecked(True)
        row_checkbox.toggled.connect(self._refresh_single_plot)
        self._trace_grid.addWidget(row_checkbox, row, 0)
        row_widgets.append(row_checkbox)

        trace_checkboxes: Dict[Tuple[str, str], QCheckBox] = {}
        for col, (trace, variant) in enumerate(self._trace_columns, start=1):
            cb = QCheckBox()
            is_measured = trace in measured_traces
            cb.setEnabled(is_measured)
            # Di default solo la variante raw e' spuntata: la gated resta
            # disponibile ma va selezionata esplicitamente, per non
            # raddoppiare subito il numero di curve visibili.
            cb.setChecked(is_measured and variant == "raw")
            cb.toggled.connect(self._refresh_single_plot)
            self._trace_grid.addWidget(cb, row, col)
            row_widgets.append(cb)
            trace_checkboxes[(trace, variant)] = cb

        return row_widgets, row_checkbox, trace_checkboxes

    def _refresh_single_plot(self, *_args):
        """Ridisegna lo spettro da zero secondo lo stato corrente delle
        checkbox. *_args assorbe l'argomento bool che QCheckBox.toggled
        passa al callback, che qui non serve.
        """
        self.single_plot_widget.clear()
        self.single_legend.clear()

        for series_index, (label, entry) in enumerate(self._measurements.items()):
            if not entry["row_checkbox"].isChecked():
                continue

            data = entry["data"]
            freq_axis = data.freq_axis

            for (trace, variant), checkbox in entry["trace_checkboxes"].items():
                if not checkbox.isEnabled() or not checkbox.isChecked():
                    continue

                # vna_results = dato gated, vna_results_raw = dato grezzo
                # (vedi controller.measure(): gated_results, raw_results).
                source = data.vna_results if variant == "gated" else data.vna_results_raw
                if trace not in source:
                    continue

                magnitude = self._magnitude(source[trace])
                style = _LINESTYLE_GATED if variant == "gated" else _LINESTYLE_RAW
                suffix = " (gated)" if variant == "gated" else ""
                self.single_plot_widget.plot(
                    freq_axis, magnitude,
                    pen=pg.mkPen(color=_trace_color(trace, series_index), width=2, style=style),
                    name=f"{label} - {trace}{suffix}",
                )

    def clear_session(self):
        """Svuota il trace control e lo spettro (nuova sessione, o
        richiesto esplicitamente da single_btn_clear_traces)."""
        for entry in self._measurements.values():
            for widget in entry["row_widgets"]:
                widget.deleteLater()
        self._measurements.clear()
        self._next_trace_row = 1

        self.single_plot_widget.clear()
        self.single_legend.clear()
        self.current_session_folder = None

    def clear_traces(self):
        for entry in self._measurements.values():
            entry["row_checkbox"].setChecked(False)
        self._refresh_single_plot()

    # ================================================================
    # TRAINING: spettro (via _SpectrumPanel) e objective/voltaggi vs
    # iterazione. Un solo punto d'ingresso per iterazione:
    # on_training_iteration(), chiamato con esattamente la tupla che
    # TrainingSession produce ad ogni passo
    # (iteration, score, last_data, history, optimizer_state).
    # ================================================================
    def _init_train_spectrum(self, container: QWidget):
        self.train_spectrum_widget, self.train_spectrum_legend = _make_frequency_plot(container)
        self.train_spectrum_widget.setLabel("left", "Magnitude", units="dB")
        self._train_spectrum_panel = _SpectrumPanel(self.train_spectrum_widget, self.train_spectrum_legend, magnitude_fn=self._magnitude)

    def _init_train_objective(self, container: QWidget):
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        plot_widget = pg.PlotWidget()
        plot_widget.setLabel("bottom", "Iteration")
        plot_widget.setLabel("left", "Objective")
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(plot_widget)
        self.train_objective_widget = plot_widget

    def _init_train_voltages(self, container: QWidget):
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        plot_widget = pg.PlotWidget()
        plot_widget.setLabel("bottom", "Iteration")
        plot_widget.setLabel("left", "Voltages", units="V")
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        legend = plot_widget.addLegend()
        layout.addWidget(plot_widget)
        self.train_voltages_widget = plot_widget
        self.train_voltages_legend = legend

    def reset_training(self, algo_name: Optional[str]):
        """Unico reset da chiamare all'avvio di un nuovo training: azzera
        sia lo spettro sia il plot objective/voltaggi, e imposta il
        criterio di selezione delle iterazioni per l'algoritmo scelto
        (vedi TRAINING_SPECTRUM_FILTERS). algo_name=None -> nessuna
        iterazione plottata oltre alla prima.
        """
        self._iteration_filter_fn = TRAINING_SPECTRUM_FILTERS.get(algo_name, lambda *_a: False)
        self._train_spectrum_panel.reset()

        self._history_iterations: List[int] = []
        self._history_objective: List[float] = []
        self._history_voltages: Dict[str, List[float]] = {}

        self.train_objective_widget.clear()
        self._objective_curve = self.train_objective_widget.plot(pen=pg.mkPen(color=_QUALITATIVE_COLORS[0], width=2))

        self.train_voltages_widget.clear()
        self.train_voltages_legend.clear()
        self._voltage_curves: Dict[str, pg.PlotDataItem] = {}

    def on_training_iteration(self, iteration: int, score: float, last_data: MeasurementData, history: list, optimizer_state: dict):
        """Chiamato una volta per ogni iterazione completata (stessa
        firma di TrainingSession.on_iteration_end).

        Il filtro dell'algoritmo (TRAINING_SPECTRUM_FILTERS) decide se
        l'iterazione viene plottata DEL TUTTO: se rifiutata, non si
        disegna ne' lo spettro ne' un punto in objective/voltaggi. La
        prima iterazione e' sempre accettata, a prescindere dal filtro.
        """
        is_first = iteration == 0
        if not is_first and not self._iteration_filter_fn(iteration, score, history):
            return

        self._update_history_plot(iteration, score, history)
        self._train_spectrum_panel.add(iteration, last_data, is_first)

    def _update_history_plot(self, iteration: int, score: float, history: list):
        self._history_iterations.append(iteration)
        self._history_objective.append(score)
        self._objective_curve.setData(self._history_iterations, self._history_objective)

        voltages = history[-1]["voltages"]  # {"ch0": v0, "ch1": v1, ...}
        for i, (key, value) in enumerate(voltages.items()):
            if key not in self._history_voltages:
                self._history_voltages[key] = []
                color = _QUALITATIVE_COLORS[i % len(_QUALITATIVE_COLORS)]
                self._voltage_curves[key] = self.train_voltages_widget.plot(pen=pg.mkPen(color=color, width=2), name=key)
            self._history_voltages[key].append(value)
            self._voltage_curves[key].setData(self._history_iterations, self._history_voltages[key])