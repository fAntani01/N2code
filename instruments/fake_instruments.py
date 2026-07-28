# instruments/fake/fake_instruments.py

from __future__ import annotations

import logging
import random
from typing import List
import numpy as np
import time

logger = logging.getLogger("MeasurementSystem")


def _lorentzian(freq: np.ndarray, f0: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude / (1 + 2j * (freq - f0) / width)


class VNA_ZNA:
    """Drop-in replacement for VNA_ZNA. Returns Lorentzian S-parameter data
    with noise so the full pipeline can be tested without a physical VNA.

    Gated data: narrow resonance, low noise.
    Raw data:   broader resonance, more noise — mimicking pre-gate response.
    """

    _RESONANCES = {
        "S11": {"f0_offset":  0.05e9, "width": 0.08e9, "amplitude": -0.8},
        "S21": {"f0_offset":  0.10e9, "width": 0.06e9, "amplitude":  0.6},
        "S31": {"f0_offset":  0.15e9, "width": 0.05e9, "amplitude":  0.5},
        "S41": {"f0_offset":  0.20e9, "width": 0.07e9, "amplitude":  0.4},
    }

    def __init__(self, gpib_address: str = "FAKE"):
        self.gpib_address = gpib_address
        self.parameters: list[str] = []
        self.freq_axis: np.ndarray | None = None
        self.rng = np.random.default_rng(seed=42)

    def connect(self) -> None:
        logger.info(f"[FakeVNA] Connected (fake) at {self.gpib_address}")

    def load_calibration(self, cal_name: str) -> None:
        logger.info(f"[FakeVNA] Calibration loaded (fake): {cal_name}")

    def setup_measurement_freq_sweep(self, parameters, start_frequency, stop_frequency,
                                     bandwidth, samples, power,
                                     start_time=None, stop_time=None) -> None:
        self.parameters = list(parameters)
        self.freq_axis = np.linspace(start_frequency, stop_frequency, samples)
        logger.info(f"[FakeVNA] Sweep configured: {start_frequency/1e9:.3f}-"
                    f"{stop_frequency/1e9:.3f} GHz, {samples} pts, params={parameters}")

    def immediate_measure_freq_sweep(self) -> tuple[dict, dict, np.ndarray]:
        freq = self.freq_axis
        f_center = (freq[0] + freq[-1]) / 2
        gated, raw = {}, {}

        for param in self.parameters:
            res = self._RESONANCES.get(param, {"f0_offset": 0.0, "width": 0.05e9, "amplitude": 0.3})
            f0 = f_center + res["f0_offset"]

            signal = _lorentzian(freq, f0, res["width"], res["amplitude"])
            noise  = self.rng.normal(0, 0.02, len(freq)) + 1j * self.rng.normal(0, 0.02, len(freq))
            gated[param] = signal + noise

            signal_raw = _lorentzian(freq, f0, res["width"] * 2.5, res["amplitude"] * 1.3)
            noise_raw  = self.rng.normal(0, 0.08, len(freq)) + 1j * self.rng.normal(0, 0.08, len(freq))
            raw[param] = signal_raw + noise_raw

            time.sleep(1)

        return gated, raw, freq

    def close(self) -> None:
        logger.info("[FakeVNA] Connection closed (fake).")


class NI6738:
    """Drop-in replacement for NI6738. Logs voltage writes, does nothing to hardware."""

    def __init__(self, device_name="Dev1", channels=None, vmin=-10.0, vmax=10.0, amp_factor=15.0):
        self.channels = channels or []
        self.current_voltages = [0.0] * len(self.channels)
        logger.info(f"[FakeDAQ] Initialised — device={device_name}, channels={channels}")

    def set_voltages_array(self, voltages: list[float]) -> None:
        self.current_voltages = list(voltages)
        logger.debug(f"[FakeDAQ] Voltages set: {voltages}")

    def set_voltages(self, voltages: dict[int, float]) -> None:
        ordered = [voltages[ch] for ch in self.channels]
        self.set_voltages_array(ordered)

    def close(self) -> None:
        logger.info("[FakeDAQ] Task closed (fake).")


class FakeFieldSensor:
    """Mock implementation of the 3MTS Field Sensor for offline testing."""
    
    def __init__(self, port: str = "MOCK_PORT"):
        self.port = port
        self.is_connected = False
        logger.info(f"[FakeSensor] Initialized mock sensor on configuration port {port}")
        self.connect()

    def connect(self) -> bool:
        """Simulates establishing communication with the sensor."""
        time.sleep(0.1) # Simulate brief connection delay
        self.is_connected = True
        logger.info("[FakeSensor] Successfully connected to mock field sensor.")
        return True
    
    def is_connected(self) -> bool:
        """Returns the current connection state of the mock sensor."""
        return self.is_connected

    def read(self, target_field_reference: float = 0.0) -> List[float]:
        """
        Returns a mock 3-axis magnetic field vector [Bx, By, Bz] in mT.
        Adds random noise/fluctuations around the values.
        """
        if not self.is_connected:
            raise RuntimeError("Attempted to read field vector from a disconnected sensor.")
        
        # Simulate slight physical measurement noise
        noise_x = random.uniform(-0.05, 0.05)
        noise_y = random.uniform(-0.05, 0.05)
        noise_z = random.uniform(-0.02, 0.02)
        
        # Guide Bx toward the applied target setpoint, keeping By and Bz as residual environment fields
        bx = target_field_reference + noise_x
        by = 1.24 + noise_y  # Mock constant ambient field on Y axis
        bz = -0.45 + noise_z # Mock constant ambient field on Z axis
        
        return np.array([bx, by, bz])*1e3

    def close(self):
        """Simulates safely releasing the sensor resource hook."""
        self.is_connected = False
        logger.info("[FakeSensor] Released connection handle safely.")
