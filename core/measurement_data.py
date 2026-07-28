from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from core.measurement_parameters import MeasurementParameters
from core.setup_parameters import SetupParameters

class MeasurementData:
    """Holds captured single-point dataset matrices containing 3-axial sensor and dual VNA readings."""
    
    def __init__(
        self,
        applied_voltages: List[float],
        measured_field_vector: List[float],
        vna_results: Dict[str, np.ndarray],
        vna_results_raw: Dict[str, np.ndarray],
        freq_axis: np.ndarray,
        setup_parameters: SetupParameters,
        measurement_parameters: MeasurementParameters,
        applied_magnetic_field: Optional[float] = None
    ):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.applied_voltages = applied_voltages
        self.measured_field_vector = list(measured_field_vector)
        self.vna_results = vna_results
        self.vna_results_raw = vna_results_raw
        self.freq_axis = freq_axis
        self.setup_parameters = setup_parameters
        self.measurement_parameters = measurement_parameters
        self.applied_magnetic_field = applied_magnetic_field