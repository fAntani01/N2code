import os
import json
import logging
import numpy as np
import pandas as pd
from .measurement_data import MeasurementData
import time

logger = logging.getLogger("MeasurementSystem")

def setup_logger(output_folder: str) -> None:
    """Configures the logger to stream to both the console and a session log file."""
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    session_name = time.strftime("Session_%Y%m%d_%H%M%S")

    log_file_path = os.path.join(output_folder, f"{session_name}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class MeasurementStorage:
    """Utility class to save data into a specified folder using Pandas and JSON."""
    
    @staticmethod
    def save_to_folder(output_folder: str, data: MeasurementData) -> None:
        """Saves data configuration to params.json and matrix traces to data.csv."""
        os.makedirs(output_folder, exist_ok=True)
        
        # 1. Export params.json
        param_path = os.path.join(output_folder, "params.json")
        param_payload = {
            "timestamp": data.timestamp,
            "applied_voltages": data.applied_voltages,
            "applied_field": data.applied_magnetic_field,
            "measured_field_vector_mt": data.measured_field_vector,
            "measurement_parameters": data.measurement_parameters.to_dict(),
            "setup_parameters": data.setup_parameters.to_dict()
        }
        with open(param_path, 'w') as f:
            json.dump(param_payload, f, indent=4)
            
        # 2. Structure VNA traces inside Pandas DataFrame
        csv_path = os.path.join(output_folder, "data.csv")
        df_dict = {"Frequency_Hz": data.freq_axis}
        
        for param in data.measurement_parameters.vna_parameters:
            # Gated profiles
            df_dict[f"{param}_gated_real"] = np.real(data.vna_results[param])
            df_dict[f"{param}_gated_imag"] = np.imag(data.vna_results[param])
            # Raw profiles
            df_dict[f"{param}_raw_real"] = np.real(data.vna_results_raw[param])
            df_dict[f"{param}_raw_imag"] = np.imag(data.vna_results_raw[param])
            
        df = pd.DataFrame(df_dict)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"[Saver] Measurement files successfully saved to: {output_folder}")