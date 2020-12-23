#!/usr/bin/env python3

# author: Michael A. Perlin (github.com/perlinm)

import os, numpy, qiskit

import circuit_cutter
import mlrecon_methods as ml

data_dir = "./mlrecon_data/"
trials = 100 # number of random circuit variants to average over

# fragment numbers to use
frag_nums = [ 2, 3, 4 ]

# number of qubits and shots for fixed-shot trials
qubit_nums = range(6,21)
log10_shots_default = 6

# number of qubits and shots for fixed-qubit trials
log10_shot_nums = numpy.arange(4.5,7.3,0.25)
qubit_num_default = 18

# general options : CHANGE AT YOUR OWN PERIL
seed = 0 # random number seed
simulation_backend = "qasm_simulator"
monitor_jobs = False
circuit_type = "clustered"
layers = 1

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

##########################################################################################
# data collection methods

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

def collect_data(qubit_num, frag_num, log10_shots):

    if qubit_num < 2*frag_num:
        print("skipping because qubit_num < 2 * frag_num")
        return

    shots = int(10**log10_shots)
    filename = data_dir + f"{circuit_type}_Q{qubit_num:02d}_F{frag_num}_S{log10_shots:.2f}.txt"

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

##########################################################################################
# collect data

for frag_num in frag_nums:

    # simulate circuits with different qubit numbers
    for qubit_num in qubit_nums:
        print(f"frag_num : {frag_num}, qubit_num : {qubit_num}")
        collect_data(qubit_num, frag_num, log10_shots_default)

    # simulate circuits with different shot numbers
    for log10_shots in log10_shot_nums:
        print(f"frag_num: {frag_num}, log10_shots : {log10_shots}")
        collect_data(qubit_num_default, frag_num, log10_shots)
