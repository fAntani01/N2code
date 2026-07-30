"""
GUI PyQt6 per il controllo del setup di misura RF.

Struttura:
- MeasurementController resta sincrono/bloccante (nessuna dipendenza Qt).
- Worker: classe generica che esegue una qualsiasi funzione bloccante su un
  QThread separato (init, shutdown, measure...), cosi' non serve una classe
  Worker diversa per ogni operazione.
- AppState + _update_ui_state(): unico punto che decide quali bottoni sono
  abilitati, in base allo stato corrente. Ogni transizione di stato passa
  da _set_state(), niente setEnabled() sparsi nei vari handler.
- Convenzione errori: i metodi del controller sollevano eccezioni in caso
  di fallimento. La GUI non controlla codici di ritorno: ogni eccezione
  arriva sempre e solo tramite il segnale error del Worker, gestita da
  _on_operation_error (init/shutdown) o _on_measurement_error (misura).
"""

from enum import Enum, auto
from typing import List, Optional, Callable
import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtWidgets import QApplication, QMainWindow, QRadioButton, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox
from PyQt6 import QtCore, uic, QtWidgets
from PyQt6.QtCore import QProcess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.storage import setup_logger
from core.controller import MeasurementController
from core.measurement_parameters import MeasurementParameters
from core.setup_parameters import SetupParameters  # adatta il path all'import reale
from plot_manager import PlotManager, SCALE_LOG, SCALE_LINEAR
from device_overview import DeviceOverviewWidget
from training.algorithms.params import ALGORITHM_PARAMS
from training.objectives import OBJECTIVE_FUNCTIONS
from training.training_session import TrainingSession
from training.algorithms.direct_search import DirectSearchOptimizer
from training.algorithms.spsa import SPSAOptimizer
from training.algorithms.optuna import OptunaOptimizer
from training.algorithms.exhaustive_search import ExhaustiveSearch  # adatta il path reale

UI_PATH = os.path.join(os.path.dirname(__file__), "GUI.ui")

os.makedirs("logs", exist_ok=True)
setup_logger("logs")
logger = logging.getLogger("MeasurementSystem")

# Registro algoritmo -> classe optimizer. Le chiavi devono combaciare con
# quelle di ALGORITHM_PARAMS e con le voci di train_combo_algo, cosi' non
# serve un if/elif per instanziare l'optimizer giusto.
OPTIMIZER_CLASSES = {"Direct Search": DirectSearchOptimizer, "SPSA": SPSAOptimizer, "Optuna (TPE)": OptunaOptimizer}


class QtLogHandler(logging.Handler, QtCore.QObject):
    """Instrada i log dello stdlib logging verso la QTextEdit della console."""

    message_logged = QtCore.pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QtCore.QObject.__init__(self)
        self.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        self.message_logged.emit(self.format(record))


# ----------------------------------------------------------------------
# STATO GENERALE DELLA GUI
# ----------------------------------------------------------------------
class AppState(Enum):
    DISCONNECTED = auto()  # Hardware non inizializzato
    BUSY = auto()  # Operazione in corso (init, shutdown, misura...)
    READY = auto()  # Hardware inizializzato, pronto per operare
    TRAINING = auto()
    SEARCHING = auto()  # Exhaustive Search in corso


# ----------------------------------------------------------------------
# WORKER GENERICO: esegue una funzione bloccante su un QThread separato
# ----------------------------------------------------------------------
class Worker(QObject):
    """Esegue fn(*args, **kwargs) in background, senza bloccare la GUI.

    Riutilizzabile per qualsiasi operazione bloccante del controller
    (initialize_setup, shutdown, measure...), evitando di dover scrivere
    una classe Worker dedicata per ognuna.
    """

    finished = pyqtSignal(object)  # emesso con il valore di ritorno di fn
    error = pyqtSignal(str)  # emesso in caso di eccezione NON gestita da fn

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            # Le operazioni del controller loggano gia' i propri errori
            # internamente; questo except cattura solo eccezioni impreviste
            # (es. bug, tipi di dato sbagliati) che sfuggono al try/except interno.
            logger.exception("[GUI] Eccezione non gestita nel worker")
            self.error.emit(str(e))


# ----------------------------------------------------------------------
# MAIN WINDOW
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    # TrainingSession.run() viene eseguito su un QThread separato tramite
    # _run_async; on_iteration_end e' quindi chiamato DA QUEL THREAD, non da
    # quello della GUI. Questo segnale fa da ponte thread-safe: chiamare
    # .emit() da un altro thread e' consentito, Qt mette in coda la
    # consegna sul thread di appartenenza di MainWindow.
    training_iteration = pyqtSignal(int, float, object, list, dict)  # (iteration, score, last_data, history, optimizer_state)
    # Stesso motivo del segnale sopra: ExhaustiveSearch gira anch'esso su
    # TrainingSession.run() in un QThread separato.
    search_iteration = pyqtSignal(int, dict)  # (iteration, optimizer_state)

    voltages_updated_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        uic.loadUi(UI_PATH, self)

        self._setup_logging()

        self.setup_params: Optional[SetupParameters] = None
        self.controller: Optional[MeasurementController] = None
        self.measurement_params: Optional[MeasurementParameters] = None


        # Controllo grafico globale log/lineare
        self._scale_log_radio = QRadioButton("Log (dB)")
        self._scale_linear_radio = QRadioButton("Linear")
        self._scale_log_radio.setChecked(True)
        self._scale_button_group = QButtonGroup(self)
        self._scale_button_group.addButton(self._scale_log_radio)
        self._scale_button_group.addButton(self._scale_linear_radio)
        self.statusBar().addPermanentWidget(self._scale_log_radio)
        self.statusBar().addPermanentWidget(self._scale_linear_radio)

        # Polling semplice del sensore di campo magnetico: un QTimer sul
        # thread della GUI, nessun thread/lock dedicato. Parte quando
        # l'hardware e' inizializzato, si ferma allo shutdown.
        self.field_timer = QTimer(self)
        self.field_timer.setInterval(200)  # ms
        self.field_timer.timeout.connect(self._poll_field_sensor)

        # Riferimenti a thread/worker attivi: tenuti come attributi
        # dell'istanza per evitare che vengano garbage-collected mentre girano.
        self._thread: Optional[QThread] = None
        self._worker: Optional[Worker] = None

        # TrainingSession.run() e' bloccante: la eseguiamo tramite lo stesso
        # _run_async usato per init/shutdown/misura. Teniamo solo il
        # riferimento alla sessione corrente per poterle chiamare .stop().
        self.train_session: Optional[TrainingSession] = None
        # Anch'essa una TrainingSession (ExhaustiveSearch e' un optimizer
        # come gli altri, vedi search_start), tenuta separata da
        # train_session per poter distinguere quale sessione fermare.
        self.search_session: Optional[TrainingSession] = None

        # self.voltage_inputs = None #[self.set_voltage_ch0, self.set_voltage_ch1, self.set_voltage_ch2, self.set_voltage_ch3, self.set_voltage_ch4, self.set_voltage_ch5, self.set_voltage_ch6, self.set_voltage_ch7]
        #self.load_settings()
        self.settings = QSettings("MyLab", "NeuromorphicApp")
        self.load_widget_states()

        self._populate_voltage_inputs()
        self._populate_vna_params()
        self._populate_algo_params(self.train_combo_algo.currentText())
        self._populate_objective_combo()
        self._on_objective_changed(self.train_combo_objective.currentText())


        self.plot_manager = PlotManager(
            single_plot_container=self.single_plot_container,
            single_trace_control_container=self.single_trace_control_container,
            train_spectrum_container=self.train_spectrum_container,
            train_objective_container=self.train_objective_container,
            train_voltages_container=self.train_voltages_container,
            s_params=self.vna_s_params
        )


        # Riquadro di stato del device (MEMS + antenne): cmn_group_overview
        overview_layout = QtWidgets.QVBoxLayout(self.cmn_group_overview)
        overview_layout.setContentsMargins(4, 4, 4, 4)
        self.device_overview = DeviceOverviewWidget()
        overview_layout.addWidget(self.device_overview)

        self.voltages_updated_signal.connect(self.update_overview)

        voltages = [0]*len(self.active_channels)
        self.device_overview.update_state(dict(zip(self.active_channels, voltages)))


        self.connect_widgets()

        self.state = AppState.DISCONNECTED
        self._update_ui_state()

    def _setup_logging(self):
        self._log_handler = QtLogHandler()
        self._log_handler.message_logged.connect(self.txt_console.append)
        logger.addHandler(self._log_handler)
        logger.setLevel(logging.INFO)

    def connect_widgets(self):
        self.btn_connect_shutdown.clicked.connect(self.controller_initialize_shutdown)
        self.cmn_btn_browse.clicked.connect(self._on_browse_folder)
        self.btn_single_measure.clicked.connect(self.start_measurement)
        self.btn_set_field.clicked.connect(self.set_field)
        self.cmn_input_folder.textChanged.connect(self._on_save_info_change)
        self.cmn_input_name.textChanged.connect(self._on_save_info_change)
        self.train_combo_algo.currentTextChanged.connect(self._populate_algo_params)
        self.train_combo_objective.currentTextChanged.connect(self._on_objective_changed)
        self.train_btn_start_stop.clicked.connect(self.training_start_stop)
        self.single_btn_clear_traces.clicked.connect(self.plot_manager.clear_traces)
        self.training_iteration.connect(self._on_training_iteration_finished)

        self._scale_log_radio.toggled.connect(lambda checked: checked and self.plot_manager.set_scale_mode(SCALE_LOG))
        self._scale_linear_radio.toggled.connect(lambda checked: checked and self.plot_manager.set_scale_mode(SCALE_LINEAR))

        self.search_btn_start_stop.clicked.connect(self.search_start_stop)
        self.search_iteration.connect(self._on_search_iteration_finished)

        self.btn_settings_apply.clicked.connect(self.save_settings)

    # ================================================================
    # GESTIONE STATO GUI
    # ================================================================
    def _set_state(self, state: AppState):
        """Unico punto da cui cambia lo stato applicativo della GUI."""
        self.state = state
        self._update_ui_state()

    def _update_ui_state(self):
        """Applica lo stato corrente a tutti i widget interessati.

        Nessun altro punto del codice deve chiamare setEnabled() su questi
        widget direttamente: passare sempre da _set_state() (o, per campi
        che non cambiano lo stato applicativo, richiamando direttamente
        questo metodo, come fa _on_save_info_change).
        """
        is_busy = self.state == AppState.BUSY
        is_ready = self.state == AppState.READY
        is_training = self.state == AppState.TRAINING
        is_searching = self.state == AppState.SEARCHING
        # "Occupato" in senso lato: qualunque operazione esclusiva sia in
        # corso, non solo BUSY. Serve a bloccare le altre tab mentre gira
        # un training O una ricerca esaustiva.
        is_exclusive_busy = is_busy or is_training or is_searching

        # Bottone Initialize/Shutdown: disponibile sempre tranne durante
        # un'operazione in corso.
        self.btn_connect_shutdown.setEnabled(not is_exclusive_busy)

        # La misura richiede anche che cartella e nome siano stati compilati,
        # altrimenti non c'e' un output_folder valido su cui salvare.
        save_info_ready = bool(self.cmn_input_folder.text().strip()) and bool(self.cmn_input_name.text().strip())
        self.btn_single_measure.setEnabled(is_ready and not is_exclusive_busy and save_info_ready)

        daq_connected = self.controller is not None and self.controller.daq is not None
        self.btn_set_voltages.setEnabled(is_ready and not is_exclusive_busy and daq_connected)
        self.btn_reset_voltages.setEnabled(is_ready and not is_exclusive_busy and daq_connected)

        # Il campo puo' essere impostato solo se il Power Supply e' stato
        # effettivamente connesso in fase di inizializzazione.
        ps_connected = self.controller is not None and self.controller.ps is not None
        self.btn_set_field.setEnabled(is_ready and not is_exclusive_busy and ps_connected)

        self.train_btn_start_stop.setEnabled(is_ready or is_training)
        self.train_btn_start_stop.setText("Stop Training" if is_training else "Start Training")

        # Exhaustive Search: due bottoni separati (non un toggle), quindi
        # niente cambio di testo, solo enable/disable incrociato.
        self.search_btn_start_stop.setEnabled(is_ready or is_searching)
        self.search_btn_start_stop.setText("Stop Search" if is_searching else "Start Search")

        # Indicatore testuale di stato nella barra in basso
        status_text = {
            AppState.DISCONNECTED: "Hardware not initialized.",
            AppState.BUSY: "Operation in progress...",
            AppState.READY: "Ready.",
            AppState.TRAINING: "Training in progress...",
            AppState.SEARCHING: "Exhaustive search in progress...",
        }[self.state]
        self.statusBar().showMessage(status_text)

    # ================================================================
    # LETTURA PARAMETRI DAI CAMPI DELLA GUI
    # ================================================================
    def _on_browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select save folder")
        if folder:
            self.cmn_input_folder.setText(folder)

    def _on_save_info_change(self):
        """Richiamato ad ogni modifica di cartella o nome misura.

        Non cambia lo stato applicativo (self.state resta invariato), ma
        influenza l'abilitazione di btn_single_measure: si passa quindi
        direttamente da _update_ui_state() invece che da _set_state().
        """
        self._update_ui_state()

    # Automatic channels setup
    def _populate_voltage_inputs(self):
        """
        Clears the current inputs in single_grp_voltages and dynamically
        rebuilds them based on the channels configured in the settings.
        """
        # 1. Parse active channels from settings_input_daq_channels line edit
        # Example: if input is "[0, 1, 2, 3]" or "0, 1, 2, 3"
        raw_text = self.settings_input_daq_channels.text().strip()

        # Clean formatting brackets if present
        for char in ["[", "]", " "]:
            raw_text = raw_text.replace(char, "")

        try:
            self.active_channels = [int(ch) for ch in raw_text.split(",") if ch.strip() != ""]
        except ValueError:
            # Fallback to default if user entered non-numeric string
            self.active_channels = [0, 1, 2, 3, 4, 5, 6, 7]

        # 2. Safely clear the existing layouts (fl_single_voltages)
        layout_single = self.fl_single_voltages
        layout_training = self.fl_train_voltages
        layout_search = self.vl_search_channels

        while layout_single.count() > 0:
            item = layout_single.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        while layout_training.count() > 0:
            item = layout_training.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        while layout_search.count() > 0:
            item = layout_search.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        # 3. Dynamically insert a QLineEdit and QLabel pair for each active channel
        self.voltage_inputs = {}  # Store references to access them easily during measurements
        self.initial_voltages = {}
        self.search_channels = {}

        for ch in self.active_channels:
            # Single Measurement
            label = QtWidgets.QLabel(f"Ch {ch}:")
            spin_box = QtWidgets.QDoubleSpinBox()
            spin_box.setObjectName(f"single_input_ch{ch}")
            self.voltage_inputs[ch] = spin_box
            layout_single.addRow(label, spin_box)

            # Training
            label = QtWidgets.QLabel(f"Ch {ch}:")
            spin_box = QtWidgets.QDoubleSpinBox()
            spin_box.setObjectName(f"train_input_ch{ch}")
            self.initial_voltages[ch] = spin_box
            layout_training.addRow(label, spin_box)

            # Exhaustive Search
            check_box = QtWidgets.QCheckBox(f"Channel {ch}")
            check_box.setObjectName(f"search_chk_ch{ch}")
            self.search_channels[ch] = check_box
            layout_search.addWidget(check_box)

        self.btn_set_voltages = QtWidgets.QPushButton("Set")
        self.btn_set_voltages.setObjectName("btn_set_voltages")
        self.btn_set_voltages.clicked.connect(self.set_voltages)

        
        self.btn_reset_voltages = QtWidgets.QPushButton("Reset")
        self.btn_reset_voltages.setObjectName("btn_reset_voltages")
        self.btn_reset_voltages.clicked.connect(self.reset_voltages)

        layout_single.addRow(self.btn_reset_voltages, self.btn_set_voltages)

    def _populate_vna_params(self):
        raw_text = self.settings_input_vna_params.text().strip()

        # Clean formatting brackets and spaces
        for char in ["[", "]", " "]:
            raw_text = raw_text.replace(char, "")

        # Parse parameters as strings (e.g., "S11, S21" or "11, 21")
        if raw_text:
            self.vna_s_params = [ch for ch in raw_text.split(",") if ch]
        else:
            # Fallback to default if empty
            self.vna_s_params = ["S11", "S21", "S31", "S41"]

        layout_params = self.vna_params_layout

        # Clear previous widgets from the horizontal layout
        while layout_params.count() > 0:
            item = layout_params.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.vna_param_checkbox = {}

        # Dynamically create and append checkboxes
        for p in self.vna_s_params:
            checkbox = QtWidgets.QCheckBox(f"{p}")
            checkbox.setObjectName(f"vna_param_{p}")
            self.vna_param_checkbox[p] = checkbox

            # Add to QHBoxLayout
            layout_params.addWidget(checkbox)

    # Automatic training algorithms params setup
    def _populate_algo_params(self, algo_name: str):
        """Ricostruisce le righe di fl_train_algo_params in base
        all'algoritmo selezionato in train_combo_algo.

        La riga 0 (Iterations) e' fissa e non viene toccata; tutte le righe
        successive vengono rimosse e ricreate secondo ALGORITHM_PARAMS.
        """
        layout = self.fl_train_algo_params

        # Rimuovi tutte le righe tranne la 0 (Iterations), partendo dal fondo
        # per non invalidare gli indici mentre rimuovi.
        while layout.rowCount() > 1:
            layout.removeRow(layout.rowCount() - 1)

        # Riferimenti alle QLineEdit create, per rileggerle in read_algorithm_params()
        self.algo_param_inputs = {}

        for param_def in ALGORITHM_PARAMS.get(algo_name, []):
            label = QtWidgets.QLabel(param_def["label"])
            line_edit = QtWidgets.QLineEdit(param_def["default"])
            layout.addRow(label, line_edit)
            self.algo_param_inputs[param_def["key"]] = line_edit

    def read_algorithm_params(self) -> dict:
        """Legge i parametri dell'algoritmo attualmente selezionato,
        gia' convertiti secondo il parser definito in ALGORITHM_PARAMS.
        """
        algo_name = self.train_combo_algo.currentText()
        params_def = ALGORITHM_PARAMS.get(algo_name, [])

        return {param_def["key"]: param_def["parser"](self.algo_param_inputs[param_def["key"]].text()) for param_def in params_def}

    # Automatic objectives setup
    def _populate_objective_combo(self):
        """Riempie train_combo_objective con i nomi definiti in
        OBJECTIVE_FUNCTIONS, cosi' il dizionario resta l'unica fonte di verita'
        (niente rischio di combo e dizionario che vanno fuori sincrono)."""
        self.train_combo_objective.clear()
        self.train_combo_objective.addItems(OBJECTIVE_FUNCTIONS.keys())

    def _on_objective_changed(self, objective_name: str):
        description = OBJECTIVE_FUNCTIONS.get(objective_name, {}).get("description", "")
        self.train_combo_objective.setToolTip(description)
        # self.label_7.setText(description)

    def get_objective_callable(self) -> Callable:
        """Ritorna la funzione obiettivo selezionata in train_combo_objective,
        pronta per essere chiamata come objective(measurement_data) -> float."""
        objective_name = self.train_combo_objective.currentText()
        return OBJECTIVE_FUNCTIONS[objective_name]["function"]

    def read_measurement_voltages(self) -> List[float]:

        daq_connected = self.controller is not None and self.controller.daq is not None

        if daq_connected:
            return [self.voltage_inputs[ch].value() for ch in self.active_channels]
        else:
            return [0]*len(self.active_channels)

    def read_inital_voltages(self) -> List[float]:

        return [self.initial_voltages[ch].value() for ch in self.active_channels]

    def read_search_channels(self) -> List[bool]:

        return [self.search_channels[ch].isChecked() for ch in self.active_channels]

    def _poll_field_sensor(self):
        """Legge il sensore 3-assiale e aggiorna i campi Bx/By/Bz.

        Chiamato ogni 200ms dal field_timer, sul thread della GUI.
        Eventuali errori di lettura vengono loggati ma non fermano il
        polling (puo' trattarsi di un glitch transitorio della seriale).
        """
        if not self.controller or not self.controller.sensor:
            return

        try:
            field_vector = self.controller.sensor.read() * 1e-3  # mT conversion, vedi controller.measure()
            self.field_Bx.setText(f"{field_vector[0]:.3f}")
            self.field_By.setText(f"{field_vector[1]:.3f}")
            self.field_Bz.setText(f"{field_vector[2]:.3f}")
        except Exception as e:
            logger.warning(f"[GUI] Field sensor read failed: {e}")

    def get_field_setpoint(self):

        ps_connected = self.controller is not None and self.controller.ps is not None

        if ps_connected:
            return float(self.field_setpoint.value())
        else:
            return None

    def set_field(self):
        """Handler per il bottone 'Set Field'."""

        target_field = self.get_field_setpoint()

        if target_field:
            self.controller.set_field(target_field)

    def set_voltages(self):
        """Handler per il bottone 'Set Voltages'."""
        voltages = self.read_measurement_voltages()
        self.controller.set_voltages(voltages)
        self.device_overview.update_state(dict(zip(self.active_channels, voltages)))

    def reset_voltages(self):
        """Handler per il bottone 'Set Voltages'."""
        for ch in self.active_channels:
            self.voltage_inputs[ch].setValue(0)
        voltages = self.read_measurement_voltages()
        self.controller.set_voltages(voltages)
        self.device_overview.update_state(dict(zip(self.active_channels, voltages)))

    def read_setup_params(self) -> SetupParameters:
        text = self.settings_input_daq_channels.text()

        return SetupParameters(
            # NI-6738 (analog voltage output)
            daq_device=self.settings_input_daq_device.text().strip() or None,
            daq_channels=[int(ch.strip()) for ch in text.split(",") if ch.strip() != ""],
            daq_vmin=float(self.settings_input_vmin.value()),
            daq_vmax=float(self.settings_input_vmax.value()),
            daq_amp_factor=float(self.settings_input_amp_factor.value()),
            # Power supply (bias field)
            ps_port=self.settings_input_ps_port.text().strip() or None,
            ps_baud=int(self.settings_input_ps_baud.text()),
            ps_offset=float(self.settings_input_ps_offset.text()),
            ps_conversion=float(self.settings_input_ps_conversion.text()),
            # VNA (R&S ZNA)
            vna_address=self.settings_input_vna_address.text().strip(),
            vna_calibration=self.settings_input_vna_cal.text().strip() or None,
        )

    def _read_vna_parameters(self) -> List[str]:

        checked_params = []

        for p in self.vna_s_params:
            if self.vna_param_checkbox[p].isChecked():
                checked_params.append(p)

        if not checked_params:
            raise ValueError("Select at least one S-parameter to measure.")
        return checked_params

    def read_measurement_params(self):

        gate_start = float(self.vna_gate_start.value()) if self.vna_grp_gate.isChecked() else None
        gate_stop = float(self.vna_gate_stop.value()) if self.vna_grp_gate.isChecked() else None

        measurement_params = MeasurementParameters(
            vna_parameters=self._read_vna_parameters(),
            vna_start_freq=self.vna_start_freq.value() * 1e9,
            vna_stop_freq=self.vna_stop_freq.value() * 1e9,
            vna_bandwidth=self.vna_bandwidth.value(),
            vna_samples=self.vna_samples.value(),
            vna_power=self.vna_power.value(),
            gate_start=gate_start,  # Provide floats (e.g. 1e-9) to enable time gating
            gate_stop=gate_stop,
            metadata={"experiment_type": "GUI measurement"},
        )

        output_folder = os.path.join(self.cmn_input_folder.text().strip(), self.cmn_input_name.text().strip())

        return measurement_params, output_folder

    # ================================================================
    # INIZIALIZZAZIONE / SHUTDOWN HARDWARE (ASYNC)
    # ================================================================
    def controller_initialize_shutdown(self):
        if not self.controller:
            self.controller = MeasurementController()

        if not self.controller.is_initialized:
            setup_params = self.read_setup_params()
            if not setup_params.self_check():
                logger.error("[GUI] Invalid setup parameters, cannot initialize hardware.")
                return

            self.setup_params = setup_params
            self._set_state(AppState.BUSY)
            self._run_async(self.controller.initialize_setup, on_finished=self._on_init_done, on_error=self._on_operation_error, setup_params=self.setup_params)
        else:
            self._set_state(AppState.BUSY)
            self._run_async(self.controller.shutdown, on_finished=self._on_shutdown_done, on_error=self._on_operation_error)

    
        self.controller.on_voltages_changed = self.voltages_updated_signal.emit

    
    def update_overview(self, voltages: List[float]):
        self.device_overview.update_state(dict(zip(self.active_channels, voltages)))


    def _on_init_done(self, _result):
        self.btn_connect_shutdown.setText("Shutdown")
        self.field_timer.start()
        self._set_state(AppState.READY)

    def _on_shutdown_done(self, _result):
        self.field_timer.stop()
        self.btn_connect_shutdown.setText("Initialize Hardware")
        self.measurement_params = None
        self._set_state(AppState.DISCONNECTED)

    def _on_operation_error(self, error_message: str):
        """Gestisce eccezioni impreviste (non i return code 0/1 del controller)."""
        logger.error(f"[GUI] Unexpected error: {error_message}")
        self._set_state(AppState.DISCONNECTED)

    # ================================================================
    # MISURA SINGOLA (ASYNC)
    # ================================================================
    def start_measurement(self):

        measurement_params, output_folder = self.read_measurement_params()

        voltages = self.read_measurement_voltages()
        magnetic_field = self.get_field_setpoint()

        if os.path.isdir(output_folder):
            reply = QtWidgets.QMessageBox.warning(
                self,
                "Measurement already exists",
                f"'{output_folder}' already exists. Overwrite?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        if not measurement_params.self_check():
            logger.error("[GUI] Invalid measurement parameters, cannot initialize hardware.")
            return

        # Riconfigura il VNA solo se i parametri di misura sono cambiati
        if measurement_params != self.controller.measurement_params:
            try:
                self.controller.setup_measurement(measurement_params)
            except Exception as e:
                self._on_measurement_error(str(e))
                return
            self.measurement_params = measurement_params

        self._pending_label = self.cmn_input_name.text().strip()
        self._pending_output_folder = output_folder

        self._set_state(AppState.BUSY)
        self._run_async(
            self.controller.measure,
            on_finished=self._on_measurement_done,
            on_error=self._on_measurement_error,
            voltages=voltages,
            magnetic_field=magnetic_field,
            output_folder=self._pending_output_folder,
        )

    def _on_measurement_done(self, data):
        logger.info("[GUI] Measurement completed.")
        self.plot_manager.register_measurement(data, label=self._pending_label, output_folder=Path(self._pending_output_folder).parent)
        self.reset_voltages()
        self._set_state(AppState.READY)

    def _on_measurement_error(self, error_message: str):
        logger.error(f"[GUI] Measurement failed: {error_message}")
        # L'hardware resta inizializzato, torna semplicemente pronto
        self.reset_voltages()
        self.controller_initialize_shutdown()
        self._set_state(AppState.DISCONNECTED)

    # ================================================================
    # TRAINING
    # ================================================================
    def training_start_stop(self):
        """Handler del bottone Start/Stop Training.

        Se un training e' gia' in corso, il click ferma quello in corso.
        Altrimenti ne avvia uno nuovo con i parametri correnti della GUI.
        """
        if self.state == AppState.TRAINING:
            self._stop_training()
            return

        try:
            n_iterations = int(self.train_input_n_iter.text())
        except ValueError as e:
            logger.error(f"[GUI] Invalid number of iterations: {e}")
            return

        measurement_params, session_folder = self.read_measurement_params()

        if os.path.isdir(session_folder):
            reply = QtWidgets.QMessageBox.warning(
                self,
                "Measurement already exists",
                f"'{session_folder}' already exists. Overwrite?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        if not measurement_params.self_check():
            logger.error("[GUI] Invalid measurement parameters, cannot start training.")
            return

        # Riconfigura il VNA solo se i parametri di misura sono cambiati
        if measurement_params != self.controller.measurement_params:
            try:
                self.controller.setup_measurement(measurement_params)
            except Exception as e:
                self._on_measurement_error(str(e))
                return
            self.measurement_params = measurement_params

        algo_name = self.train_combo_algo.currentText()
        self.plot_manager.reset_training(algo_name)

        # on_iteration_end viene chiamato dal thread di background di
        # TrainingSession (vedi _run_async piu' sotto): non tocchiamo i
        # widget qui dentro, ci limitiamo a rilanciare il segnale Qt, che
        # verra' consegnato in modo thread-safe sul thread della GUI.
        session = TrainingSession(
            controller=self.controller,
            score_fn=self.get_objective_callable(),
            magnetic_field=self.get_field_setpoint(),
            session_folder=session_folder,
            on_iteration_end=lambda iteration, score, last_data, history, optimizer_state: self.training_iteration.emit(iteration, score, last_data, history, optimizer_state),
        )

        try:
            optimizer_cls = OPTIMIZER_CLASSES[algo_name]
            # ASSUNZIONE: ogni optimizer accetta objective e x0 come kwarg,
            # piu' i parametri specifici dell'algoritmo gia' letti da
            # ALGORITHM_PARAMS (es. "values" per Direct Search, "a"/"c" per
            # SPSA...). Se le firme reali usano nomi diversi (es.
            # objective_fn invece di objective), va adattato solo qui.
            optimizer = optimizer_cls(objective_fn=session._objective, x0=self.read_inital_voltages(), **self.read_algorithm_params())
        except Exception as e:
            logger.error(f"[GUI] Could not build optimizer: {e}")
            return

        session.optimizer = optimizer
        self.train_session = session

        self._set_state(AppState.TRAINING)
        self._run_async(self.train_session.run, on_finished=self._on_training_finished, on_error=self._on_training_error, n_iter=n_iterations)

    def _stop_training(self):
        """Richiede l'interruzione del training in corso.

        Non ferma il thread immediatamente: TrainingSession completa
        l'iterazione corrente (misura in corso inclusa) e poi esce dal loop;
        _on_training_finished riporta lo stato a READY quando il thread
        termina davvero.
        """
        if self.train_session:
            self.train_session.stop()
        logger.info("[GUI] Stop requested, waiting for current iteration to complete...")

    def _on_training_finished(self, _result):
        logger.info("[GUI] Training finished.")
        self.reset_voltages()
        self._set_state(AppState.READY)

    def _on_training_error(self, error_message: str):
        # TrainingSession._run_loop cattura gia' internamente ogni
        # eccezione (e la logga), quindi questo ramo scatta solo per
        # errori davvero imprevisti al di fuori del loop stesso.
        logger.error(f"[GUI] Unexpected error while running training: {error_message}")
        self.reset_voltages()
        self.controller_initialize_shutdown()
        self._set_state(AppState.DISCONNECTED)

    def _on_training_iteration_finished(self, iteration: int, score: float, last_data, history: list, optimizer_state: dict):
        """Slot collegato al segnale training_iteration: gira sul thread
        della GUI (Qt lo garantisce), puo' toccare i widget in sicurezza.
        """
        self.plot_manager.on_training_iteration(iteration, score, last_data, history, optimizer_state)

        voltages = {int(key[2:]): value for key, value in history[-1]["voltages"].items()}
        self.device_overview.update_state(voltages)

        n_iterations = int(self.train_input_n_iter.text())
        self.train_progress.setValue(int(100 * (iteration + 1) / n_iterations))
        self.train_status_lbl.setText(f"Iteration {iteration + 1}/{n_iterations} — score: {score:.4f}")

    # ================================================================
    # EXHAUSTIVE SEARCH
    # ================================================================
    # Riusa TrainingSession/optimizer.step() come Training: lo score non
    # conta (score_fn e' un placeholder) e il numero di iterazioni deriva
    # dal numero di combinazioni generate da ExhaustiveSearch stesso
    # (optimizer.n_configs), cosi' il loop si ferma esattamente quando
    # tutte sono state testate. Nessun plot: solo la progress bar viene
    # aggiornata ad ogni iterazione.
    #
    # x0 e' semplicemente un vettore di zeri lungo quanto i canali attivi:
    # ExhaustiveSearch usa x0 solo per determinare il numero di canali da
    # combinare (len(x0)), non i valori effettivi — i voltaggi testati
    # vengono generati dal prodotto cartesiano di values_set, non da x0
    # (vedi ExhaustiveSearch.step()). Nessun objective custom necessario:
    # session._objective viene passato cosi' com'e'.

    def search_start_stop(self):

        if self.state == AppState.SEARCHING:
            self.search_stop()
            return

        start = self.search_input_start.value()
        stop = self.search_input_stop.value()
        step = self.search_input_step.value()
        if step <= 0:
            logger.error("[GUI] Invalid step for Exhaustive Search: must be > 0.")
            return
        values_set = np.arange(start, stop + step / 2, step).tolist()  # +step/2: include stop nonostante arrotondamenti float

        measurement_params, session_folder = self.read_measurement_params()
        if not measurement_params.self_check():
            logger.error("[GUI] Invalid measurement parameters, cannot start search.")
            return

        if os.path.isdir(session_folder):
            reply = QtWidgets.QMessageBox.warning(
                self,
                "Measurement already exists",
                f"'{session_folder}' already exists. Overwrite?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        if measurement_params != self.controller.measurement_params:
            try:
                self.controller.setup_measurement(measurement_params)
            except Exception as e:
                self._on_measurement_error(str(e))
                return
            self.measurement_params = measurement_params

        session = TrainingSession(
            controller=self.controller,
            score_fn=lambda data: 0.0,  # placeholder: lo score non ci interessa
            magnetic_field=self.get_field_setpoint(),
            session_folder=session_folder,
            on_iteration_end=lambda iteration, score, last_data, history, optimizer_state: self.search_iteration.emit(iteration, optimizer_state),
        )

        x0 = [0.0] * len(self.active_channels)

        # Indici (posizioni in x0/active_channels) dei canali NON
        # spuntati: restano fissi al loro valore in x0 invece di essere
        # inclusi nella combinatoria.
        fixed_indices = [i for i, ch in enumerate(self.active_channels) if not self.search_channels[ch].isChecked()]

        try:
            optimizer = ExhaustiveSearch(objective_fn=session._objective, x0=x0, values_set=values_set, fixed_indices=fixed_indices)
        except Exception as e:
            logger.error(f"[GUI] Could not build Exhaustive Search optimizer: {e}")
            return

        session.optimizer = optimizer
        self.search_session = session
        self._search_n_iterations = optimizer.n_configs

        # self.search_lbl_status.setText(f"Status: running (0/{self._search_n_iterations})")
        self.search_progress.setValue(0)

        self._set_state(AppState.SEARCHING)
        self._run_async(self.search_session.run, on_finished=self._on_search_finished, on_error=self._on_search_error, n_iter=optimizer.n_configs)

    def search_stop(self):
        """Richiede l'interruzione della ricerca in corso.

        Come per il training, non ferma il thread immediatamente:
        TrainingSession completa l'iterazione corrente e poi esce dal loop.
        """
        if self.search_session:
            self.search_session.stop()
        logger.info("[GUI] Search stop requested, waiting for current iteration to complete...")

    def _on_search_finished(self, _result):
        logger.info("[GUI] Exhaustive search finished.")
        self.search_status_lbl.setText("Status: Idle")
        self.reset_voltages()
        self._set_state(AppState.READY)

    def _on_search_error(self, error_message: str):
        logger.error(f"[GUI] Unexpected error while running search: {error_message}")
        self.search_status_lbl.setText("Status: Idle")
        self.reset_voltages()
        self.controller_initialize_shutdown()
        self._set_state(AppState.DISCONNECTED)

    def _on_search_iteration_finished(self, iteration: int, optimizer_state: dict):
        """Slot collegato al segnale search_iteration: gira sul thread
        della GUI. Nessun plot per Exhaustive Search: solo la progress bar
        e lo status label vengono aggiornati.
        """

        if self.search_session and self.search_session.history:
            voltages = {int(key[2:]): value for key, value in self.search_session.history[-1]["voltages"].items()}
            self.device_overview.update_state(voltages)

        self.search_progress.setValue(int(100 * (iteration + 1) / self._search_n_iterations))
        self.search_status_lbl.setText(f"Status: running ({iteration + 1}/{self._search_n_iterations})")


    def save_settings(self):

        QProcess.startDetached(sys.executable, sys.argv)
        self.close()


    def get_target_widgets(self):
        """Define which widgets to persist.

        You can return an explicit list, or fetch all children of a specific
        container (e.g., self.settings_grp_vna.findChildren(QWidget))
        """
        return [
            self.cmn_input_folder,
            self.cmn_input_name,
            self.vna_start_freq,
            self.vna_stop_freq,
            self.vna_bandwidth,
            self.vna_samples,
            self.vna_power,
            self.vna_gate_start,
            self.vna_gate_stop,
            self.field_setpoint,
            self.train_combo_algo,
            self.train_combo_objective,
            self.search_input_start,
            self.search_input_stop,
            self.search_input_step,
            self.settings_input_daq_device,
            self.settings_input_daq_channels,
            self.settings_input_amp_factor,
            self.settings_input_vmin,
            self.settings_input_vmax,
            self.settings_input_ps_port,
            self.settings_input_ps_baud,
            self.settings_input_ps_offset,
            self.settings_input_ps_conversion,
            self.settings_input_vna_address,
            self.settings_input_vna_cal,
            self.settings_input_vna_params         

            # Add any other specific widgets here...
        ]

    def save_widget_states(self):
        """Saves current state of target widgets to QSettings."""
        for widget in self.get_target_widgets():
            name = widget.objectName()
            if not name:
                continue

            if isinstance(widget, QLineEdit):
                self.settings.setValue(f"UI/{name}", widget.text())
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                self.settings.setValue(f"UI/{name}", widget.value())
            elif isinstance(widget, QCheckBox):
                self.settings.setValue(f"UI/{name}", widget.isChecked())
            elif isinstance(widget, QComboBox):
                self.settings.setValue(f"UI/{name}", widget.currentIndex())

    def load_widget_states(self):
        """Restores target widgets from QSettings."""
        for widget in self.get_target_widgets():
            name = widget.objectName()
            key = f"UI/{name}"

            if not name or not self.settings.contains(key):
                continue

            val = self.settings.value(key)

            if isinstance(widget, QLineEdit):
                widget.setText(str(val))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(val))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(val))
            elif isinstance(widget, QCheckBox):
                # QSettings converts bools to strings in some backends
                is_checked = val == True or str(val).lower() == "true"
                widget.setChecked(is_checked)
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(int(val))

    # ================================================================
    # HELPER: esecuzione asincrona generica
    # ================================================================
    def _run_async(self, fn: Callable, on_finished: Callable = None, on_error: Callable = None, *args, **kwargs):
        """Esegue fn(*args, **kwargs) su un QThread separato senza bloccare la GUI."""
        thread = QThread()
        worker = Worker(fn, *args, **kwargs)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        if on_finished:
            worker.finished.connect(on_finished)
        if on_error:
            worker.error.connect(on_error)

        # Chiudi il thread quando il worker ha finito (successo o errore)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)

        # Mantieni i riferimenti: se non li tieni, Python li garbage-collecta
        # mentre girano e il programma crasha silenziosamente.
        self._thread = thread
        self._worker = worker

        thread.start()

    def closeEvent(self, event):
        """Chiamato automaticamente da Qt alla chiusura della finestra
        (X, Alt+F4, ecc.). Ferma eventuali operazioni in corso e spegne
        l'hardware prima di lasciare che la finestra si chiuda davvero.
        """
        if self.state == AppState.TRAINING and self.train_session:
            self.train_session.stop()
            self.train_session.join()
        elif self.state == AppState.SEARCHING and self.search_session:
            self.search_session.stop()
            self.search_session.join()

        if self.controller and self.controller.is_initialized:
            self.controller.shutdown()

        self.save_widget_states()
        event.accept()


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
