import logging
import numpy as np
from typing import List, Union
from .senis3MTS_driver import Senis3MTSDriver

logger = logging.getLogger("MeasurementSystem")

class FieldSensor:
    """
    Synchronous on-demand Field Sensor wrapper.
    Interfaces directly with Senis3MTSDriver without any background threads.
    """
    def __init__(self, device_number: int = 0):
        self.device_number = device_number
        self.driver = None
        self._connected = False
        
        # Automatically attempt initialization on instantiation
        self.connect()

    def connect(self) -> bool:
        """Initializes the low-level driver and opens the hardware device link."""
        try:
            logger.info(f"[FieldSensor] Connecting to Senis 3MTS device index {self.device_number}...")
            self.driver = Senis3MTSDriver(device_number=self.device_number)
            self.driver.open()
            
            # Optional default configuration: set range to 0.5T or similar if desired
            self.driver.set_range(Senis3MTSDriver.RANGE_0_5T)
            
            self._connected = True
            logger.info("[FieldSensor] Successfully connected to magnetic field sensor hardware.")
            return True
        except Exception as e:
            logger.error(f"[FieldSensor] Connection failed: {e}", exc_info=True)
            self.driver = None
            self._connected = False
            return False

    def is_connected(self) -> bool:
        """Returns the active connectivity state expected by MeasurementController."""
        return self._connected

    def read(self) -> np.ndarray:
        """
        Performs a fast, synchronous reading of the field axes.
        
        Returns:
            np.ndarray: Array containing [x, y, z] values. 
                         If disconnected, returns [0.0, 0.0, 0.0].
        """
        if not self._connected or self.driver is None:
            logger.warning("[FieldSensor] Attempted to read field, but sensor is not connected. Returning zero vector.")
            return np.array([0.0, 0.0, 0.0])

        try:
            # Synchronous read from the DLL via your driver file wrapper
            data = self.driver.read_values()
            return np.array([data["x"], data["y"], data["z"]])
        except Exception as e:
            logger.error(f"[FieldSensor] Error reading values from sensor hardware: {e}")
            return np.array([0.0, 0.0, 0.0])

    def close(self) -> None:
        """Closes the hardware communication layer."""
        if self.driver is not None:
            try:
                logger.info("[FieldSensor] Releasing device connection index channel...")
                self.driver.close()
            except Exception as e:
                logger.error(f"[FieldSensor] Error closing low-level driver interface: {e}")
            finally:
                self.driver = None
                self._connected = False
                logger.info("[FieldSensor] Disconnected cleanly.")