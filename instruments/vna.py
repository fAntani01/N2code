from RsInstrument.RsInstrument import RsInstrument
import numpy as np
import time


class VNA_ZNA:
    """
    Control an R&S ZNA in CW mode.
    Allows setting individual measurement parameters and measuring
    specific S-parameters (e.g., S21, S31, S41), with a check to ensure
    all parameters are set before measurement.
    """

    def __init__(self, gpib_address):
        self.gpib_address = gpib_address
        self.inst = None
        self.timeout = 300  # 1 minute timeout
        self.calibration = None

    def connect(self):
        self.inst = RsInstrument(self.gpib_address, True, False)
        self.inst.write("*RST")
        self.inst.write("CALC1:PAR:DEL 'Trc1'")

        print("Connected to:", self.inst.query("*IDN?").strip())

    def close(self):
        self.inst.go_to_local()
        self.inst.close()

    def preset(self):
        self.inst.write("*RST")
        

    def setup_measurement_CW(self, parameters, frequency, bandwidth, samples, sweep_time=None, power=0, detector_time=None):

        self.parameters = list(parameters)
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.sweep_points = samples
        self.sweep_time = sweep_time
        self.power = power
        self.detector_time = detector_time

        self.inst.write("SENS1:SWE:TYPE CW")
        self.inst.write(f"SENS1:FREQ:FIX {frequency}")
        self.inst.write(f"SENS1:BAND {bandwidth}")
        self.inst.write(f"SENS1:SWE:POIN {samples}")

        if self.sweep_time is not None:
            self.inst.write(f"SENS1:SWE:TIME {sweep_time} S")

        if detector_time is not None:
            self.inst.write(f"SENS1:SWE:DET:TIME {detector_time} S")

        self.inst.write(f"SOUR1:POW {power}")
        self.inst.write("SENS1:COUP ALL")  # Chopped mode
        self.inst.write("SENS1:IFP WID")

        for i, param in enumerate(parameters, start=1):
            trace = f"Tr{i}"
            self.inst.write(f"CALC1:PAR:SDEF '{trace}', '{param}'")
            self.inst.write(f"DISP:WIND{i}:STAT ON")
            self.inst.write(f"DISP:WIND{i}:TRAC{i}:FEED '{trace}'")

        self.inst.write(":INIT1:CONT:ALL OFF")

        print(f"CW measurement setup: {frequency/1e9:.3f} GHz, "
              f"{bandwidth} Hz BW, {samples} points, "
              f"sweep {sweep_time}s, power {power} dBm, "
              f"detector {detector_time}s, "
              f"parameters: {', '.join(parameters)}")

    def setup_measurement_freq_sweep(self, parameters, start_frequency, stop_frequency, bandwidth, samples, power, start_time=None, stop_time=None):

        self.parameters = list(parameters)
        self.bandwidth = bandwidth
        self.sweep_points = samples
        self.power = power

        self.inst.write("SENS1:SWE:TYPE LIN")
        self.inst.write(f"SENS1:FREQ:STAR {start_frequency}")
        self.inst.write(f"SENS1:FREQ:STOP {stop_frequency}")
        self.inst.write(f"SENS1:BAND {bandwidth}")
        self.inst.write(f"SENS1:SWE:POIN {samples}")
        self.inst.write(f"SOUR1:POW {power}")
        self.inst.write("SENS1:COUP ALL")  # Chopped mode
        self.inst.write("SENS1:IFP WID")

        for i, param in enumerate(parameters, start=1):
            trace = f"Tr{i}"
            self.inst.write(f"CALC1:PAR:SDEF '{trace}', '{param}'")
            self.inst.write(f"DISP:WIND{i}:STAT ON")
            self.inst.write(f"DISP:WIND{i}:TRAC{i}:FEED '{trace}'")

            if start_time and stop_time:
                self.inst.write("CALC1:FILT:TIME:STAT ON")  # Enable time gating
                self.inst.write(f"CALC1:FILT:TIME:STAR {start_time}")
                self.inst.write(f"CALC1:FILT:TIME:STOP {stop_time}")

        self.inst.write(":INIT1:CONT:ALL OFF")

        print(f"Frequency sweep measurement setup: Start frequency: {start_frequency/1e9:.3f} GHz, "
              f"Stop frequency: {stop_frequency/1e9:.3f} GHz, "
              f"{bandwidth} Hz BW, {samples} points, "
              f"power {power} dBm, "
              f"parameters: {', '.join(parameters)} "
              f"start time: {start_time*1e9 if start_time else None}ns, "
              f"stop time: {stop_time*1e9 if stop_time else None}ns")

    def ext_trigger_setup(self, slope: str = "POS", delay=0.0):
        # Trigger setup and measurement arm

        self.inst.write("TRIG:SOUR EXT")
        self.inst.write(f"TRIG:SLOP {slope}")
        self.inst.write(":INIT1:IMM:ALL; *OPC")

    def load_calibration(self, cal_name):
        self.inst.write(f":MMEM:LOAD:CORR 1, '{cal_name}.cal'")
        self.inst.write(f"SENS1:FREQ:CONV:GAIN:LMC OFF")  # Disable load match correction
        self.calibration = cal_name
        print(f"Calibration loaded: {cal_name}.cal")

    def query_results_CW(self):

        self.inst.query_opc(self.timeout * 1000)

        results = {}
        for i, param in enumerate(self.parameters, start=1):
            trace = f"Tr{i}"
            data_str = self.inst.query(f"CALC:DATA:TRAC? '{trace}', SDATA")
            data = np.array(data_str.split(','), dtype='float32')
            results[param] = data[::2] + 1j * data[1::2]

        t_str = self.inst.query("CALC1:DATA:STIM?")
        time_axis = np.array(t_str.split(','), dtype='float32')

        return results, time_axis  # in reality the measure lasts double time...

    def immediate_measure_CW(self):

        self.inst.query_with_opc(":INIT1:IMM:ALL; *OPC?", self.timeout * 1000)

        results = {}
        for i, param in enumerate(self.parameters, start=1):
            trace = f"Tr{i}"
            data_str = self.inst.query(f"CALC:DATA:TRAC? '{trace}', SDATA")
            data = np.array(data_str.split(','), dtype='float32')
            results[param] = data[::2] + 1j * data[1::2]

        t_str = self.inst.query("CALC1:DATA:STIM?")
        time_axis = np.array(t_str.split(','), dtype='float32')

        return results, time_axis  # in reality the measure lasts double time...

    def immediate_measure_freq_sweep(self):

        self.inst.query_with_opc(":INIT1:IMM:ALL; *OPC?", self.timeout * 1000)

        results = {}
        for i, param in enumerate(self.parameters, start=1):
            trace = f"Tr{i}"
            data_str = self.inst.query(f"CALC:DATA:TRAC? '{trace}', MDATA")
            data = np.array(data_str.split(','), dtype='float32')
            results[param] = data[::2] + 1j * data[1::2]

        results_raw = {}
        for i, param in enumerate(self.parameters, start=1):
            trace = f"Tr{i}"
            data_str = self.inst.query(f"CALC:DATA:TRAC? '{trace}', SDATA")
            data = np.array(data_str.split(','), dtype='float32')
            results_raw[param] = data[::2] + 1j * data[1::2]

        t_str = self.inst.query("CALC1:DATA:STIM?")
        freq_axis = np.array(t_str.split(','), dtype='float32')

        return results, results_raw, freq_axis

    def get_frequency(self): return self.frequency
    def get_bandwidth(self): return self.bandwidth
    def get_sweep_points(self): return self.sweep_points
    def get_sweep_time(self): return self.sweep_time
    def get_detector_time(self): return self.detector_time
    def get_power(self): return self.power
    def get_calibration(self): return self.calibration
    def get_parameters(self): return self.parameters
