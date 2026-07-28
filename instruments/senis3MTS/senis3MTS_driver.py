import ctypes as C
import os
from pathlib import Path

class Senis3MTSDriver:
    # Costanti per i range di misura documentati
    RANGE_0_1T = 0
    RANGE_0_5T = 1
    RANGE_3T    = 2

    def __init__(self, device_number = 0):
        """
        Inizializza la libreria SENIS 3MTS.
        :param dll_path: Percorso assoluto o relativo della DLL (es. '3mtslib64.dll')
        :param device_number: Indice del dispositivo da gestire (default: 0)

        """

        if '__file__' in globals():
            base_dir = Path(__file__).resolve().parent
        else:
            base_dir = Path.cwd()

        dll_path = base_dir / "libraries" / "a3mtslib64.dll"

        #dll_path = r"C:\Users\VNA Desktop\OneDrive - Politecnico di Milano\Tesi\Neuromorphic\N2\N2 code\instruments\senis3MTS\libraries\a3mtslib64.dll"

        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DLL non trovata al percorso: {dll_path}")
        
        self.lib = C.CDLL(dll_path)
        self.device_number = C.c_int(device_number)
        self._setup_ctypes_signatures()

    def _setup_ctypes_signatures(self):
        """Configura i tipi di argomento e di ritorno per le funzioni della DLL."""
        self.lib.count_devices.argtypes = [C.POINTER(C.c_ushort)]
        self.lib.count_devices.restype = C.c_int

        self.lib.open_device.argtypes = [C.POINTER(C.c_int)]
        self.lib.open_device.restype = C.c_int

        self.lib.close_device.argtypes = [C.POINTER(C.c_int)]
        self.lib.close_device.restype = C.c_int

        self.lib.get_sensor_count.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_int)]
        self.lib.get_sensor_count.restype = C.c_int

        self.lib.set_range.argtypes = [C.POINTER(C.c_int), C.c_ushort]
        self.lib.set_range.restype = C.c_int

        self.lib.get_range.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_ushort)]
        self.lib.get_range.restype = C.c_int

        self.lib.set_speed.argtypes = [C.POINTER(C.c_int), C.c_ushort]
        self.lib.set_speed.restype = C.c_int

        self.lib.get_speed.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_ushort)]
        self.lib.get_speed.restype = C.c_int

        self.lib.clear_buffer.argtypes = [C.POINTER(C.c_int)]
        self.lib.clear_buffer.restype = C.c_int

        # Funzioni scoperte dallo script di esempio (interfaccia flat)
        self.lib.get_device_name_ch.argtypes = [C.POINTER(C.c_int), C.c_char_p]
        self.lib.get_device_name_ch.restype = C.c_int

        self.lib.get_sensor_values_fl.argtypes = [
            C.POINTER(C.c_int),       # device_number
            C.POINTER(C.c_ulong),     # timestamp
            C.POINTER(C.c_float),     # sensorx
            C.POINTER(C.c_float),     # sensory
            C.POINTER(C.c_float)      # sensorz
        ]
        self.lib.get_sensor_values_fl.restype = C.c_int

    def _check_status(self, result_code):
        """Verifica i codici di ritorno della DLL e solleva eccezioni appropriate."""
        if result_code == 0x0:
            return
        elif result_code == 0x8000:
            raise RuntimeError("Errore 3MTS: Dispositivo non inizializzato.")
        elif result_code == 0x8001:
            raise ValueError("Errore 3MTS: Range al di fuori dell'intervallo consentito.")
        else:
            raise RuntimeError(f"Errore 3MTS imprevisto. Codice: {hex(result_code)}")

    def count_devices(self):
        """Restituisce il numero di dispositivi disponibili sul sistema."""
        count = C.c_ushort(0)
        res = self.lib.count_devices(C.byref(count))
        self._check_status(res)
        return count.value

    def open(self):
        """Apre la connessione con il dispositivo di misura."""
        res = self.lib.open_device(C.byref(self.device_number))
        self._check_status(res)

    def close(self):
        """Chiude la connessione con il dispositivo di misura."""
        res = self.lib.close_device(C.byref(self.device_number))
        #self._check_status(res)

    def get_sensor_count(self):
        """Restituisce il numero di canali del sensore."""
        count = C.c_int(0)
        res = self.lib.get_sensor_count(C.byref(self.device_number), C.byref(count))
        self._check_status(res)
        return count.value

    def set_range(self, range_enum):
        """
        Imposta il range di misura del sensore.
        :param range_enum: 0=0.1mT, 1=0.5mT, 2=3T, 3=20T
        """
        res = self.lib.set_range(C.byref(self.device_number), C.c_ushort(range_enum))
        self._check_status(res)

    def get_range(self):
        """Restituisce il range attuale (0=0.1mT, 1=0.5mT, 2=3T, 3=20T)."""
        current_range = C.c_ushort(0)
        res = self.lib.get_range(C.byref(self.device_number), C.byref(current_range))
        self._check_status(res)
        return current_range.value

    def set_speed(self, period_ms):
        """
        Imposta il periodo di tempo della misura (frequenza di campionamento).
        :param period_ms: Periodo in millisecondi (1=1ms, 2=2ms, 3=3ms, ecc.)
        """
        res = self.lib.set_speed(C.byref(self.device_number), C.c_ushort(period_ms))
        self._check_status(res)

    def get_speed(self):
        """Restituisce il periodo di campionamento attuale in millisecondi."""
        current_speed = C.c_ushort(0)
        res = self.lib.get_speed(C.byref(self.device_number), C.byref(current_speed))
        self._check_status(res)
        return current_speed.value

    # def get_device_name(self):
    #     """Restituisce il nome identificativo del dispositivo."""
    #     p= C.create_string_buffer(40)

    #     device_number = C.c_int()

    #     res = self.lib.get_device_name_ch(C.byref(device_number),C.byref(p))

    #     print (p.value)

    def clear_buffer(self):
        """Svuota il buffer interno dei valori memorizzati dal dispositivo."""
        res = self.lib.clear_buffer(C.byref(self.device_number))
        self._check_status(res)

    def read_values(self):
        """
        Esegue la lettura istantanea dei canali X, Y, Z e del relativo timestamp.
        :return: Dizionario contenente timestamp, x, y, z
        """
        timestamp = C.c_ulong(0)
        sensor_x = C.c_float(0.0)
        sensor_y = C.c_float(0.0)
        sensor_z = C.c_float(0.0)
        
        res = self.lib.get_sensor_values_fl(
            C.byref(self.device_number),
            C.byref(timestamp),
            C.byref(sensor_x),
            C.byref(sensor_y),
            C.byref(sensor_z)
        )
        #self._check_status(res)
        
        return {
            "timestamp": timestamp.value,
            "x": sensor_x.value,
            "y": sensor_y.value,
            "z": sensor_z.value
        }

    # Supporto Context Manager per l'uso nativo con l'istruzione 'with'
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()