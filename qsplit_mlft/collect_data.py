#!/usr/bin/env python3
"""
Script to collect the data plotted in the maximum likelihood fragment tomography paper.

Author: Michael A. Perlin (github.com/perlinm)
"""
import os

from qsplit_mlft import circuit_ansatz
from qsplit_mlft import compute_fidelities
import numpy as np


data_dir = "../data/"
num_trials = 100  # number of random circuit variants to average over

# fragment numbers to use
frag_nums = [2, 3, 4]

# number of qubits and repetitions for fixed-shot trials
qubit_nums = range(6, 21)
log10_repetitions_default = 6

# number of qubits and repetitions for fixed-qubit trials
log10_shot_nums = np.arange(4.5, 7.3, 0.25)
qubit_num_default = 18

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)


##########################################################################################
# data collection methods


def collect_data(
    num_qubits: int,
    num_frags: int,
    log10_repetitions: float,
    num_trials: int = num_trials,
) -> None:
    """Collect simulation data for the given parameters."""

    if num_qubits < 2 * num_frags:
        print("skipping because num_qubits < 2 * num_frags")
        return

    repetitions = int(10**log10_repetitions)
    basename = f"fidelities_Q{num_qubits:02d}_F{num_frags}_S{log10_repetitions:.2f}.txt"
    filename = os.path.join(data_dir, basename)

    # determine which trial to start on
    trial_start = 0
    if os.path.isfile(filename):
        print(f"data file exists: {filename}")
        print("using the number of lines to determine starting trial number")
        with open(filename, "r") as file:
            for line, _ in enumerate(file):
                pass
            trial_start = line

    # if we are starting fresh, print a file header
    else:
        with open(filename, "w") as file:
            file.write("# full, direct, likely fidelities\n")

    for trial in range(trial_start, num_trials):
        print(f"trial: {trial}")
        np.random.seed(trial)

        # construct circuit and compute fidelities
        circuit, cuts = circuit_ansatz.random_clustered_circuit(num_qubits, num_frags)
        full_fidelity, direct_fidelity, likely_fidelity = compute_fidelities.get_fidelities(
            circuit, cuts, repetitions
        )
        with open(filename, "a") as file:
            file.write(f"{full_fidelity} {direct_fidelity} {likely_fidelity}\n")


##########################################################################################
# collect data

# simulate circuits with different qubit numbers
for qubit_num in qubit_nums:
    for frag_num in frag_nums:
        print(f"frag_num : {frag_num}, qubit_num : {qubit_num}")
        collect_data(qubit_num, frag_num, log10_repetitions_default)

# simulate circuits with different shot numbers
for log10_repetitions in log10_shot_nums:
    for frag_num in frag_nums:
        print(f"frag_num: {frag_num}, log10_repetitions : {log10_repetitions}")
        collect_data(qubit_num_default, frag_num, log10_repetitions)
