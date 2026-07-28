import os
import time
import logging
from core.measurement_parameters import MeasurementParameters
from core.setup_parameters import SetupParameters
from core.controller_fake import MeasurementController
from core.storage import setup_logger


def run_single_test_measurement():
    # 1. Create a unique session directory for this test run
    session_base_dir = "test_runs"
    session_name = time.strftime("Test_%Y%m%d_%H%M%S")
    session_folder = os.path.join(session_base_dir, session_name)
    os.makedirs(session_folder, exist_ok=True)

    # 2. Initialize the global session logger (streams to terminal and session_log.log)
    setup_logger(session_folder)
    logger = logging.getLogger("MeasurementSystem")
    logger.info("Starting a simple standalone test measurement sequence.")

    # 3. Define your initialization parameters configuration profile
    # Update these values to match your actual hardware address/ports!
    setup_params = SetupParameters(
        daq_device="Dev1",
        daq_channels=[0, 1, 2, 3],  # Assuming you are controlling 4 voltage lines
        ps_port="COM4",
        ps_offset=2.5797,
        ps_conversion=49.917,
        # VNA Configuration (Frequency Sweep only)
        vna_calibration=None,  # Provide a calibration string name if needed
       
    )


    measurement_params = MeasurementParameters(
        vna_parameters=["S21", "S11"],
        vna_start_freq=4.0e9,
        vna_stop_freq=4.5e9,
        vna_bandwidth=100.0,
        vna_samples=401,
        vna_power=0.0,
        gate_start=None,  # Provide floats (e.g. 1e-9) to enable time gating
        gate_stop=None,
        # Optional metadata tracker
        metadata={"experiment_type": "Test fake measurement", "operator": "User"}
    )

    # 4. Instantiate the Controller
    controller = MeasurementController()

    try:
        # Step 1: Initialize Setup (Connect and configure instruments)
        controller.initialize_setup(setup_params)

        # Define target values for this specific measurement point
        target_voltages = [1.5, -2.0, 0.5, 3.1]  # Must match the length of daq_channels!
        target_magnetic_field = 12.5  # Target field in mT

        # Define the subfolder where this specific measurement point should save its data
        point_output_folder = os.path.join(session_folder, "measurement_point_1")

        logger.info("Executing measure_point routine...")
        # Step 2 & 3: Run the measurement routine
        # This will set the field, apply voltages, read the sensor, sweep the VNA, and save data

        controller.setup_measurement(measurement_params)

        measurement_results = controller.measure(
            voltages=target_voltages,
            output_folder=point_output_folder,
            magnetic_field=target_magnetic_field,  # Set to None if you want to control field externally
        )

        logger.info("Measurement routine executed successfully!")
        logger.info(f"Timestamp of execution: {measurement_results.timestamp}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the test script: {e}", exc_info=True)

    finally:
        # Step 4: System Shutdown (safely zeros voltages/currents and drops handles)
        controller.shutdown()


if __name__ == "__main__":
    run_single_test_measurement()
