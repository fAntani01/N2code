# run_training.py

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import logging
from core.setup_parameters import SetupParameters
from core.controller import MeasurementController
from core.storage import setup_logger
from training.training_session import TrainingSession
from training.algorithms.direct_search import DirectSearchOptimizer


# --------------------------------------------------------------------------- #
# Setup                                                                        #
# --------------------------------------------------------------------------- #


# 1. Create a unique session directory for this test run
session_base_dir = "test_runs"
session_name = time.strftime("Training_%Y%m%d_%H%M%S")
session_folder = os.path.join(session_base_dir, session_name)
os.makedirs(session_base_dir, exist_ok=True)
os.makedirs(session_folder, exist_ok=True)

# 2. Initialize the global session logger (streams to terminal and session_log.log)
setup_logger(session_folder)
logger = logging.getLogger("MeasurementSystem")
logger.info("Starting a simple training test sequence.")

# 3. Define your initialization parameters configuration profile
# Update these values to match your actual hardware address/ports!
params = SetupParameters(
    daq_channels=[0, 1, 2, 3],  # Assuming you are controlling 4 voltage lines
    ps_port="COM4",
    ps_offset=2.5797,
    ps_conversion=49.917,
    # VNA Configuration (Frequency Sweep only)
    vna_calibration=None,  # Provide a calibration string name if needed
    vna_parameters=["S21", "S11"],
    vna_start_freq=4.0e9,
    vna_stop_freq=4.5e9,
    vna_bandwidth=100.0,
    vna_samples=401,
    vna_power=0.0,
    gate_start=None,  # Provide floats (e.g. 1e-9) to enable time gating
    gate_stop=None,
)

# 4. Instantiate the Controller
controller = MeasurementController(params)

controller.initialize_setup()

# --------------------------------------------------------------------------- #
# Score function                                                                #
# --------------------------------------------------------------------------- #

def score_fn(data):
    s31  = data.vna_results["S21"]
    band = (data.freq_axis > 4.3e9) & (data.freq_axis < 4.4e9)
    return float(np.mean(np.abs(s31[band])))    # maximise mean |S31| in band

# --------------------------------------------------------------------------- #
# Live plots                                                                   #
# --------------------------------------------------------------------------- #

plt.ion()
fig, (ax_spectrum, ax_history) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle(session_name)
fig.tight_layout(pad=3.0)


def on_iteration_end(data, history):
    iteration = history[-1]["iter"]
    score     = history[-1]["score"]
    freq_GHz  = data.freq_axis / 1e9

    # Left: latest spectra — gated solid, raw dashed
    ax_spectrum.clear()
    for param in params.vna_parameters:
        gated_dB = 20 * np.log10(np.abs(data.vna_results[param])     + 1e-12)
        raw_dB   = 20 * np.log10(np.abs(data.vna_results_raw[param]) + 1e-12)
        ax_spectrum.plot(freq_GHz, gated_dB,              label=param)
        ax_spectrum.plot(freq_GHz, raw_dB,   "--", alpha=0.4, label=f"{param} raw")
    ax_spectrum.set_title(f"iter {iteration} | score {score:.4f}")
    ax_spectrum.set_xlabel("Frequency [GHz]")
    ax_spectrum.set_ylabel("|S| [dB]")
    ax_spectrum.legend(fontsize=7)
    ax_spectrum.grid(True, alpha=0.3)

    # Right: voltage trajectories + score
    ax_history.clear()
    iters = [h["iter"] for h in history]
    for ch in params.daq_channels:
        ax_history.plot(iters, [h["voltages"][f"ch{ch}"] for h in history], label=f"ch{ch}")
    ax_history.set_xlabel("iteration")
    ax_history.set_ylabel("voltage [V]")
    ax_history.legend(fontsize=7, loc="upper left")
    ax_history.grid(True, alpha=0.3)

    ax2 = ax_history.twinx()
    ax2.plot(iters, [h["score"] for h in history], "k--", alpha=0.5)
    ax2.set_ylabel("score")

    # Print summary line
    voltages_str = "  ".join(f"ch{ch}={history[-1]['voltages'][f'ch{ch}']:.1f}V"
                              for ch in params.daq_channels)
    #print(f"  iter {iteration:03d} | score {score:.4f} | {voltages_str}")

    fig.canvas.draw()
    fig.canvas.flush_events()

# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #

session = TrainingSession(
    controller=controller,
    score_fn=score_fn,
    magnetic_field=50,
    session_folder=session_folder,
    on_iteration_end=on_iteration_end,
)

optimizer = DirectSearchOptimizer(
    objective_fn=session._objective,
    x0=[0.0] * len(params.daq_channels),
    values_set=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
)
session.optimizer = optimizer

N_ITER = 10

print(f"\nStarting training '{session_name}' — {N_ITER} iterations\n")


try:
    session.start(n_iter=N_ITER)
    input("Press Enter to stop...\n")
    session.stop()
    session.join()
finally:
    controller.shutdown()

print(f"\nBest score : {optimizer.best_score:.4f}")
print(f"Best config: {optimizer.x}")

plt.ioff()
plt.show()
