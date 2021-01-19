#!/usr/bin/env python3

# author: Michael A. Perlin (github.com/perlinm)

import itertools, numpy, qiskit

print(qiskit.__version__)

import circuit_cutter
import mlrecon_methods as ml

numpy.set_printoptions(linewidth = 200)

shots = 10**6

# should be "GHZ", "cascade", "dense", or "clustered"
circuit_type = "clustered"
layers = 1 # number of gate layers

qubits = 9 # number of qubits
frag_num = 3 # number of fragments

simulation_backend = "qasm_simulator"

seed = 0
print_circuits = True

##########################################################################################
# build and cut a circuit
##########################################################################################

circuit, cuts = ml.build_circuit_with_cuts(circuit_type, layers, qubits, frag_num, seed)
print('circuit:')
print(circuit)
print('cuts:')
print(cuts)

fragments, wire_path_map = circuit_cutter.cut_circuit(circuit, cuts)
fragment_cuts = ml.fragment_cuts(frag_num, wire_path_map)

def variants(cuts):
    return 4**cuts["prep"] * 3**cuts["meas"]
total_variants = sum( variants(cuts) for cuts in fragment_cuts )

if print_circuits:
    print("total circuit:")
    print(circuit)
    print()
    for idx, fragment in enumerate(fragments):
        print(f"fragment {idx}:")
        print(fragment)
        print()
    print("fragment_index, prep_cuts, meas_cuts:")
    for frag_idx, frag_cuts in enumerate(fragment_cuts):
        print(frag_idx, frag_cuts["prep"], frag_cuts["meas"])
    print()
    print("total number of fragment variants:", total_variants)
    print("total number of shots:", ( shots // total_variants ) * total_variants)
    print()

# list of all possible measurement outcomes (bitstrings)
all_bits = [ "".join(bits) for bits in itertools.product(["0","1"], repeat = qubits) ]

# get the actual state / probability distribution for the full circuit
actual_state = ml.get_statevector(circuit)
actual_dist = { "".join(bits) : abs(amp)**2
                for bits, amp in zip(all_bits, actual_state)
                if amp != 0 }

# get a simulated probability distribution for the full circuit
circuit.measure_all()
full_circuit_result = ml.run_circuits(circuit, shots, backend = simulation_backend)
full_circuit_dist = {}
for part in full_circuit_result:
    for bits, counts in part.get_counts(circuit).items():
        if bits not in full_circuit_dist:
            full_circuit_dist[bits] = 0
        full_circuit_dist[bits] += counts / shots

##########################################################################################
# simulate fragments and recombine results to reconstruct the outputs of the full circuit
##########################################################################################

# simulate fragments, build fragment models, and recombine fragment models
frag_data = ml.collect_fragment_data(fragments, wire_path_map,
                                     shots = shots // total_variants,
                                     tomography_backend = simulation_backend)
direct_models = ml.direct_fragment_model(frag_data)
likely_models = ml.maximum_likelihood_model(direct_models)

direct_recombined_dist = ml.recombine_fragment_models(direct_models, wire_path_map)
likely_recombined_dist = ml.recombine_fragment_models(likely_models, wire_path_map)

def naive_fix(dist):
    norm = sum( value for value in dist.values() if value >= 0 )
    return { bits : value / norm for bits, value in dist.items() if value >= 0 }

direct_recombined_dist = naive_fix(direct_recombined_dist)

def fidelity(dist):
    fidelity = sum( numpy.sqrt(actual_dist[bits] * dist[bits], dtype = complex)
                    for bits in all_bits
                    if actual_dist.get(bits) and dist.get(bits) )**2
    return fidelity.real if fidelity.imag == 0 else fidelity

direct_fidelity = fidelity(direct_recombined_dist)
likely_fidelity = fidelity(likely_recombined_dist)
full_circuit_fidelity = fidelity(full_circuit_dist)

if qubits <= 5:
    print()
    print("actual probability distribution:")
    for bits in all_bits:
        try: print(bits, actual_dist[bits])
        except: None

    print()
    print("'direct' recombined probability distribution:")
    for bits in all_bits:
        try: print(bits, direct_recombined_dist[bits])
        except: None

    print()
    print("'likely' recombined probability distribution:")
    for bits in all_bits:
        try: print(bits, likely_recombined_dist[bits])
        except: None

    print()
    print("full circuit probability distribution:")
    for bits in all_bits:
        try: print(bits, full_circuit_dist[bits])
        except: None

print()
print("'direct' distribution fidelity:", direct_fidelity)
print("'likely' distribution fidelity:", likely_fidelity)
print("full circuit fidelity:", full_circuit_fidelity)
