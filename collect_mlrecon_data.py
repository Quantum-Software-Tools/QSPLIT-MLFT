#!/usr/bin/env python3

# author: Michael A. Perlin (github.com/perlinm)

import os, numpy, itertools, qiskit

import circuit_cutter
import mlrecon_methods as ml

seed = 0
simulation_backend = "qasm_simulator"
monitor_jobs = False
data_dir = "./mlrecon_data/"

circuit_type = "clustered"
layers = 1

shots = 10**6
trials = 100

qubit_nums = list(range(6,21))
frag_nums = list(range(2,5))

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

##########################################################################################
# build and cut a circuit
##########################################################################################

def get_filename(qubit_num, frag_num):
    return data_dir + f"{circuit_type}_L{layers}_F{frag_num}_Q{qubit_num}.txt"

# number of fragment variants for given numbers of incident measurement / preparation cuts
def variants(cuts):
    return 4**cuts["prep"] * 3**cuts["meas"]

# "naive" fix to the a "direct" recombined probability distribution
def naive_fix(dist):
    norm = sum( value for value in dist.values() if value >= 0 )
    return { bits : value / norm for bits, value in dist.items() if value >= 0 }

# copute the fidelity of two probability distributions
def fidelity(dist, actual_dist):
    qubits = int(numpy.log2(actual_dist.size)+0.5)
    fidelity = sum( numpy.sqrt(prob * actual_dist[int(bits,2)])
                    for bits, prob in dist.items() )**2
    return fidelity.real if fidelity.imag == 0 else fidelity

for qubit_num, frag_num in itertools.product(qubit_nums, frag_nums):
    print(f"qubit_num, frag_num: {qubit_num}, {frag_num}")

    if qubit_num < 2*frag_num:
        print("skipping because qubit_num < 2 * frag_num")
        continue

    filename = get_filename(qubit_num, frag_num)

    # determine which trial to start on
    trial_start = 0
    if os.path.isfile(filename):
        print(f"data file exists: {filename}")
        print("using the number of lines to determine starting trial number")
        with open(filename, "r") as file:
            for line, _ in enumerate(file): pass
            trial_start = line

    # if we are starting fresh, print a file header
    else:
        with open(filename, "w") as file:
            file.write(f"# full, direct, likely fidelities\n")

    for trial in range(trial_start, trials):
        print(f"trial: {trial}")

        # construct a circuit, and cut it into fragments
        circuit, cuts = ml.build_circuit_with_cuts(circuit_type, layers,
                                                   qubit_num, frag_num, seed + trial)
        fragments, wire_path_map = circuit_cutter.cut_circuit(circuit, cuts)

        fragment_cuts = ml.fragment_cuts(frag_num, wire_path_map)
        total_variants = sum( variants(cuts) for cuts in fragment_cuts )

        # get the actual probability distribution for the full circuit
        actual_dist = abs(ml.get_statevector(circuit))**2

        # get a simulated probability distribution for the full circuit
        circuit.cregs.append(qiskit.ClassicalRegister(qubit_num))
        circuit.measure(range(qubit_num),range(qubit_num))
        full_circuit_result = ml.run_circuits(circuit, shots, backend = simulation_backend,
                                              monitor_jobs = monitor_jobs)
        full_circuit_dist = {}
        for part in full_circuit_result:
            for bits, counts in part.get_counts(circuit).items():
                if bits not in full_circuit_dist:
                    full_circuit_dist[bits] = 0
                full_circuit_dist[bits] += counts / shots

        # simulate fragments, build fragment models, and recombine fragment models
        frag_data = ml.collect_fragment_data(fragments, wire_path_map,
                                             shots = shots // total_variants,
                                             tomography_backend = simulation_backend,
                                             monitor_jobs = monitor_jobs)
        direct_models = ml.direct_fragment_model(frag_data)
        likely_models = ml.maximum_likelihood_model(direct_models)

        direct_recombined_dist = ml.recombine_fragment_models(direct_models, wire_path_map)
        likely_recombined_dist = ml.recombine_fragment_models(likely_models, wire_path_map)

        direct_recombined_dist = naive_fix(direct_recombined_dist)

        # compute and store fidelities
        full_circuit_fidelity = fidelity(full_circuit_dist, actual_dist)
        direct_fidelity = fidelity(direct_recombined_dist, actual_dist)
        likely_fidelity = fidelity(likely_recombined_dist, actual_dist)
        with open(filename, "a") as file:
            file.write(f"{full_circuit_fidelity} {direct_fidelity} {likely_fidelity}\n")
