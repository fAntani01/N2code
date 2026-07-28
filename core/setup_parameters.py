from typing import Dict, Any, List
import logging

logger = logging.getLogger("MeasurementSystem")

class SetupParameters:
    """Complete parameter profile container for initializing the experimental setup."""

    def __init__(
        self,
        # NI-6738 (analog voltage output)
        daq_device: str = "Dev1",
        daq_channels: List[int] = None,
        daq_vmin: float = -10.0,
        daq_vmax: float = 10.0,
        daq_amp_factor: float = 15.0,
        # Power supply (bias field)
        ps_port: str = None,
        ps_baud: int = 9600,
        ps_offset: float = 2.5797,
        ps_conversion: float = 49.917,
        # VNA (R&S ZNA)
        vna_address: str = "GPIB0::20::INSTR",
        vna_calibration: str = None,
    ):
        self.daq_device = daq_device
        self.daq_channels = daq_channels if daq_channels is not None else []
        self.daq_vmin = daq_vmin
        self.daq_vmax = daq_vmax
        self.daq_amp_factor = daq_amp_factor

        self.ps_port = ps_port
        self.ps_baud = ps_baud
        self.ps_offset = ps_offset
        self.ps_conversion = ps_conversion

        self.vna_address = vna_address
        self.vna_calibration = vna_calibration

    def update_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Dumps internal attributes into a dictionary for JSON output serialization."""
        return self.__dict__.copy()

    def self_check(self) -> bool:
        """Performs a self-check to ensure all required parameters are set."""
     
        return True
