from typing import List, Optional
from .senis3MTS_driver import Senis3MTSDriver
from .sensor_worker import SensorWorker
import time
import numpy as np

class FieldSensor:
    """Wrapper that verifies physical sensor connections instantly at startup."""
    
    def __init__(self):
        self.worker: Optional[SensorWorker] = None
        self._is_connected: bool = False
        self.last_error: Optional[str] = None

        try:
            # 1. Instantiating the driver maps the DLL
            sonda = Senis3MTSDriver(device_number=0)
            
            # 2. Directly attempt a low-level test to open the device context handle 
            # using the C-DLL connection signature
            res = sonda.lib.open_device(sonda.device_number)
            sonda._check_status(res)
            
            # 3. If no status exceptions are thrown, close the temporary test handle...
            sonda.lib.close_device(sonda.device_number)
            
            # 4. ...and hand it over safely to the background worker thread
            self._is_connected = True
            self.worker = SensorWorker(sonda)
            self.worker.start()
            time.sleep(1)
            print("Field Sensor successfully verified on main thread. Stream started.")

            
        except Exception as e:
            # If the driver fails to open the hardware, fall back cleanly
            self._is_connected = False
            self.last_error = str(e)
            print(f"Sensor not detected ({e}). Running in simulation/fallback mode.")

    def is_connected(self) -> bool:
        """Instant check matching your test logic script directly."""
        return self._is_connected

    def read(self) -> List[float]:
        """Returns [Bx, By, Bz] or zero-fallbacks without cross-thread delays."""
        if not self._is_connected or self.worker is None:
            return np.array*([0.0, 0.0, 0.0])
        return np.array(self.worker.get_latest_vector())

    def get_error(self) -> Optional[str]:
        return self.last_error

    def close(self) -> None:
        if self.worker:
            self.worker.stop()
        print("FieldSensor interface closed.")