import numpy as np
import nidaqmx
import nidaqmx.system
from nidaqmx.stream_writers import AnalogMultiChannelWriter


class NI6738:
    """
    Manage analog voltage output on multiple AO channels of an NI device (e.g. NI-6738),
    using one common frequency and sampling clock.

    Supports continuous generation of sine or rectangular waveforms.
    """

    def __init__(self, device_name: str = "Dev1", channels: list[int] | None = None, vmin: float = 0, vmax: float = 60.0, amp_factor: float = 15.0):
        if not channels:
            raise ValueError("channels must be a non-empty list of AO channel indices.")

        self.device_name = device_name
        self.channels = list(channels)
        self.amp_factor = amp_factor
        self.vmin = vmin
        self.vmax = vmax

        self.task = nidaqmx.Task()

        # Add all AO channels
        for ch in self.channels:
            physical_name = f"{self.device_name}/ao{ch}"
            self.task.ao_channels.add_ao_voltage_chan(physical_name)

        self.analogWriter = AnalogMultiChannelWriter(self.task.out_stream)

    def set_voltages(self, voltages: dict[int, float]):
        """Set output voltages by channel number. Convenient for manual /
        test measurements where you only care about a couple of channels.

        Args:
            voltages: {channel: voltage} for every channel this DAQ has.
        """
        missing = [ch for ch in self.channels if ch not in voltages]
        if missing:
            raise KeyError(f"voltages is missing entries for channel(s) {missing}")

        ordered = [voltages[ch] for ch in self.channels]
        self.set_voltages_array(ordered)

    def set_voltages_array(self, voltages: list[float]):
        """Set output voltages from an ordered array, matching self.channels
        order. No dict lookups — this is what optimizer / training loops
        should call, since they already work with a flat vector of values.

        Args:
            voltages: list of voltages, one per channel, in self.channels order.
        """
        if len(voltages) != len(self.channels):
            raise ValueError("Length of voltages list must match number of channels.")

        # Scale voltages by amp_factor and clip to min/max
        scaled_voltages = np.array(voltages, dtype=float)
        scaled_voltages = np.clip(scaled_voltages, self.vmin, self.vmax)
        scaled_voltages = scaled_voltages / self.amp_factor

        # Write voltages to the channels
        self.analogWriter.write_one_sample(scaled_voltages)


    def set_zero(self):
        """Set all channels to zero voltage."""
        voltages_0 = np.zeros(len(self.channels), dtype=float)
        self.analogWriter.write_one_sample(voltages_0)

    def close(self):
        """Close and clean up the DAQ task."""
        self.set_zero()  # Set all channels to zero
        self.task.close()
        nidaqmx.system.Device(self.device_name).reset_device()
        print("DAQ task closed.")
