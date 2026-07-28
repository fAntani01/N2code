import threading
import time
from typing import List, Optional

class SensorWorker:
    """Continuously polls an already opened and verified hardware driver instance."""
    
    def __init__(self, driver_instance):
        self.sonda = driver_instance
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        
        self._latest_vector: List[float] = [0.0, 0.0, 0.0]
        self._lock = threading.Lock()

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def get_latest_vector(self) -> List[float]:
        with self._lock:
            return list(self._latest_vector)

    def _acquisition_loop(self) -> None:
        """Dedicated thread loop for streaming data."""
        try:
            # The context manager (__enter__) handles baseline configurations
            with self.sonda:
                self.sonda.set_range(self.sonda.RANGE_0_5T)
                self.sonda.set_speed(10)
                self.sonda.clear_buffer()
                
                while self.running:
                    dati = self.sonda.read_values()
                    with self._lock:
                        self._latest_vector = [dati["x"], dati["y"], dati["z"]]
                    time.sleep(0.002)
        except Exception as e:
            print(f"Runtime hardware pooling error: {e}")