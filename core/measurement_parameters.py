from typing import Dict, Any, List, Optional

class MeasurementParameters:
    """Complete parameter profile container for initializing the experimental setup."""
    
    def __init__(
        self,
        vna_parameters: List[str] = None,
        vna_start_freq: float = 4.0e9,
        vna_stop_freq: float = 4.5e9,
        vna_bandwidth: float = 100.0,
        vna_samples: int = 401,
        vna_power: float = 0.0,
        gate_start: Optional[float] = None,
        gate_stop: Optional[float] = None,
        
        # Extra metadata tracker
        metadata: Optional[Dict[str, Any]] = None
    ):

        self.vna_parameters = vna_parameters
        self.vna_start_freq = vna_start_freq
        self.vna_stop_freq = vna_stop_freq
        self.vna_bandwidth = vna_bandwidth
        self.vna_samples = vna_samples
        self.vna_power = vna_power
        self.gate_start = gate_start
        self.gate_stop = gate_stop
        self.metadata = metadata if metadata is not None else {}


        
    def update_metadata(self, key: str, value: Any): 
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Dumps internal attributes into a dictionary for JSON output serialization."""
        return self.__dict__.copy()
    

    def self_check(self) -> bool:

        # implement checks

        ok = True
        if self.vna_parameters is None: ok = False
        if (self.vna_start_freq is None or self.vna_start_freq <= 0): ok = False
        if (self.vna_stop_freq is None or self.vna_stop_freq <= 0): ok = False


        return True