#!/usr/bin/env python3
"""
Script to collect the data plotted in the maximum likelihood fragment tomography paper.

Author: Michael A. Perlin (github.com/perlinm)
"""
import os

import numpy as np

from qsplit_mlft import circuit_ansatz, compute_fidelities

##########################################################################################
# data collection methods


def collect_data(
    num_frags: int,
    num_qubits: int = 18,
    log10_repetitions: float = 6,
    num_trials: int = 100,
    data_dir: str = "./data",
) -> None:
    """Collect simulation data for the given parameters.

    Args:
        num_frags: number of circuit fragments in the clustered random unitary circuit
        num_qubits: total number of qubits in the circuit
        log10_repetitions: total number of repetitions (shots) to sample
        num_trials: number of random circuit instances to average over
        data_dir: directory in which to save simulation data
    """
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    print(num_frags, num_qubits, log10_repetitions, num_trials, data_dir)
    return

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
if __name__ == "__main__":
    # collect data in the QSPLIT-MLFT paper

    # fragment numbers to simulate
    frag_nums = [2, 3, 4]

    # simulate circuits with different qubit numbers
    for qubit_num in range(8, 21):
        for frag_num in frag_nums:
            print(f"frag_num : {frag_num}, qubit_num : {qubit_num}")
            collect_data(frag_num, num_qubits=qubit_num)

    # simulate circuits with different shot numbers
    for log10_repetitions in np.arange(4.5, 7.3, 0.25):
        for frag_num in frag_nums:
            print(f"frag_num: {frag_num}, log10_repetitions : {log10_repetitions}")
            collect_data(frag_num, log10_repetitions=log10_repetitions)
