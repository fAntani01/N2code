import time
import logging
from typing import List, Optional


# Import hardware drivers
from instruments.fake_instruments import NI6738
from instruments.fake_instruments import VNA_ZNA
from instruments.fake_instruments import FakeFieldSensor

# Import local data layout entities
from .setup_parameters import SetupParameters
from .measurement_parameters import MeasurementParameters
from .measurement_data import MeasurementData
from .storage import MeasurementStorage


logger = logging.getLogger("MeasurementSystem")


class MeasurementController:
    """Orchestrates instrument initialization, routines, and safe teardowns."""

    def __init__(self):
        self.vna: Optional[VNA_ZNA] = None
        self.ps = None
        self.sensor: Optional[FakeFieldSensor] = None
        self.daq: Optional[NI6738] = None
        self.is_initialized = False  # Track whether the controller has been initialized
        self.measurement_params = None

    def initialize_setup(self, setup_params: SetupParameters):
        """Step 1: Sets up permanent instrument parameters."""
        logger.info("[Controller] Beginning System Setup Initialization")

        self.setup_params = setup_params

        # 1. VNA initialization
        logger.info(f"[Controller] Connecting to VNA at {self.setup_params.vna_address}...")
        self.vna = VNA_ZNA(self.setup_params.vna_address)
        self.vna.connect()

        if self.setup_params.vna_calibration:
            self.vna.load_calibration(self.setup_params.vna_calibration)

        logger.info("[Controller] Power Supply not connected: port not specified")

        # 3. 3-Axial Sensor hookup
        logger.info("[Controller] Connecting to 3-Axial Field Sensor...")
        self.sensor = FakeFieldSensor()
        self.sensor.connect()
        if not self.sensor.is_connected:
            logger.warning("[Controller] Connection to magnetic field sensor failed")

        # 4. Analog Output DAQ configuration
        if self.setup_params.daq_device:
            logger.info(f"[Controller] Opening DAQ Task profile: {self.setup_params.daq_device}...")
            self.daq = NI6738(device_name=self.setup_params.daq_device, channels=self.setup_params.daq_channels, vmin=self.setup_params.daq_vmin, vmax=self.setup_params.daq_vmax, amp_factor=self.setup_params.daq_amp_factor)
        else:
            logger.info("[Controller] DAQ not connected: device not specified")


        self.is_initialized = True  # Mark the controller as initialized

        logger.info("[Controller] System Configuration Complete. Hardware Ready.")

    def setup_measurement(self, measurement_params: MeasurementParameters):

        if not self.is_initialized:
            raise RuntimeError("Cannot setup measurement parameters: Setup not initialized.")

        logger.info("[Controller] Setup Measurement Parameters")

        self.measurement_params = measurement_params

        self.vna.setup_measurement_freq_sweep(
            parameters=self.measurement_params.vna_parameters,
            start_frequency=self.measurement_params.vna_start_freq,
            stop_frequency=self.measurement_params.vna_stop_freq,
            bandwidth=self.measurement_params.vna_bandwidth,
            samples=self.measurement_params.vna_samples,
            power=self.measurement_params.vna_power,
            start_time=self.measurement_params.gate_start,
            stop_time=self.measurement_params.gate_stop,
        )

    def measure(self, voltages: Optional[List[float]] = None, magnetic_field: Optional[float] = None, output_folder: str = None) -> MeasurementData:
        """Step 2 & 3: Optionally drives power supply field, sets target voltages, reads fields,

        gathers VNA sweeps, and saves the data directly to disk.
        """
        logger.info(f"[Controller] Running measure sequence. Saving to: {output_folder}")

        # 1. Update dynamic magnetic field if optionally specified
        if magnetic_field is not None:
            if self.ps:
                logger.info(f"[Controller] Setting dynamic magnetic field target to: {magnetic_field} mT")
                self.ps.setField(magnetic_field)
            else:
                logger.warning("[Controller] Magnetic field update requested, but Power Supply is not initialized!")
                magnetic_field = None
        else:
            logger.info("[Controller] No magnetic field explicitly passed; maintaining current state (or controlled externally).")

        # 2. Update dynamic voltages down the DAQ lines
        if voltages is not None:
            if len(voltages) != len(self.setup_params.daq_channels):
                raise ValueError(f"Length of voltages ({len(voltages)}) does not match number of DAQ channels ({len(self.setup_params.daq_channels)}).")

            if self.daq:
                logger.info("[Controller] Updating voltages")
                self.daq.set_voltages_array(voltages)
                time.sleep(0.01)  # Settling buffer delay
            else: 
                logger.warning("[Controller] Voltage field update requested, but DAQ is not initialized!")
                #voltages = [None] * len(self.setup_params.daq_channels)

        # 3. Read 3-Axial Sensor feedback loop TO BE IMPLEMENTED
        field_vector = self.sensor.read() * 1e-3  # mT conversion

        logger.info(f"[Controller] Measured 3-Axial Magnetic Vector [mT]: {field_vector}")

        # 4. Trigger frequency sweep trace captures
        gated_results, raw_results, freq_axis = self.vna.immediate_measure_freq_sweep()

        # Pack data object
        data_point = MeasurementData(
            applied_voltages=voltages,
            measured_field_vector=field_vector,
            vna_results=gated_results,
            vna_results_raw=raw_results,
            freq_axis=freq_axis,
            setup_parameters=self.setup_params,
            measurement_parameters=self.measurement_params,
            applied_magnetic_field=magnetic_field,
        )

        # 5. Direct internal execution saving routine command
        if output_folder:
            MeasurementStorage.save_to_folder(output_folder, data_point)

        return data_point

    def reset(self):
        """Reset all instruments to a safe state."""
        if self.daq:
            self.daq.set_zero()
        if self.ps:
            self.ps.setCurrent(0.0)
        if self.vna:
            self.vna.preset()
        logger.info("[Controller] Instruments reset complete.")

    def set_field(self, field_value: float):
        """Sets the magnetic field using the power supply."""
        if self.ps:
            logger.info(f"[Controller] Setting magnetic field to {field_value} mT")
            self.ps.setField(field_value)
        else:
            logger.warning("[Controller] Power Supply not initialized; cannot set field.")

    def set_voltages(self, voltages: List[float]):
        """Sets the voltages on the DAQ channels."""
        if self.daq:
            logger.info(f"[Controller] Setting voltages to {voltages}")
            self.daq.set_voltages_array(voltages)
        else:
            logger.warning("[Controller] DAQ not initialized; cannot set voltages.")

    def shutdown(self):
        """Step 4: Securely drops power tracks, zeros outputs, and clears physical references."""
        logger.info("[Controller] Executing Secure Instrument Disconnection Loop")

        if self.daq:
            try:
                self.daq.close()
            except Exception as e:
                logger.error(f"[Controller] Error while stopping DAQ tasks: {e}")

        if self.ps:
            try:
                self.ps.setCurrent(0.0)
                self.ps.closeConnection()
            except Exception as e:
                logger.error(f"[Controller] Error resetting power supply: {e}")

        if self.sensor:
            try:
                self.sensor.close()
            except Exception as e:
                logger.error(f"[Controller] Error releasing field sensor: {e}")

        if self.vna:
            try:
                self.vna.close()
            except Exception as e:
                logger.error(f"[Controller] Error terminating VNA connection: {e}")

        self.is_initialized = False

        logger.info("[Controller] Shutdown routine safely completed.")
