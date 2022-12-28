#!/usr/bin/env python3
"""
Demo script for circuit cutting, fragment tomography, and maximum-likelihood corrections.

Author: Michael A. Perlin (github.com/perlinm)
"""
import collections
from typing import Dict, Iterable, Tuple

import circuit_ansatz
import cirq
import cutting_methods as cm
import numpy as np
import numpy.typing as npt


def get_fidelity(
    approx_dist: Dict[Tuple[int, ...], float], exact_dist: npt.NDArray[np.float_]
) -> float:
    """
    Compute the fidelity between two classical probability distributions.

    The first distribution is represented by a dictionary mapping bitstrings to their probability of
    measurement, while the second distribution is represented by an array that is indexed by
    bitstrings.  Neither distribution is assumed to be normalized.
    """
    overlap = sum(np.sqrt(prob * exact_dist[bits]) for bits, prob in approx_dist.items())
    norms = sum(prob for prob in approx_dist.values()) * exact_dist.sum()
    return overlap**2 / norms


def get_fidelities(
    circuit: cirq.Circuit,
    cuts: Iterable[Tuple[int, cirq.Qid]],
    repetitions: int,
) -> Tuple[float, float, float]:
    """
    Compute the fidelities of reconstructing the output of a circuit using
    1. full circuit execution,
    2. the original circuit cutting method in arXiv:1904.00102, and
    3. the maximum likelyhood fragment tomography method in arXiv:2005.12702.
    """
    qubit_order = sorted(circuit.all_qubits())
    num_qubits = len(qubit_order)

    # compute the actual probability distribution over measurement outcomes for the circuit
    actual_probs = np.abs(cirq.final_state_vector(circuit, qubit_order=qubit_order)) ** 2
    actual_probs.shape = (2,) * num_qubits  # reshape into array indexed by bitstrings

    # get a probability distribution over measurement outcomes by sampling
    circuit_samples = np.random.choice(
        range(actual_probs.size), size=repetitions, p=actual_probs.ravel()
    )
    full_circuit_probs = {
        tuple(int(bit) for bit in bin(outcome)[2:].zfill(num_qubits)): counts / repetitions
        for outcome, counts in collections.Counter(circuit_samples).items()
    }

    # cut the circuit, and perform fragment tomography to build fragment models
    fragments = cm.cut_circuit(circuit, cuts)
    num_variants = sum(
        4 ** len(fragment.quantum_inputs) * 3 ** len(fragment.quantum_outputs)
        for fragment in fragments.values()
    )
    repetitions_per_variant = repetitions // num_variants
    tomo_data = cm.perform_fragment_tomography(fragments, repetitions=repetitions_per_variant)
    direct_models = cm.build_fragment_models(tomo_data)
    likely_models = cm.corrected_fragment_models(direct_models)

    # recombine fragments to infer the distribution over measurement outcomes for the full circuit
    direct_probs = cm.recombine_fragment_models(direct_models, qubit_order=qubit_order)
    likely_probs = cm.recombine_fragment_models(likely_models, qubit_order=qubit_order)

    # apply "naive" corrections to the "direct" probability distribution: throw out negative values
    direct_probs = {
        bistring: probability for bistring, probability in direct_probs.items() if probability >= 0
    }

    full_fidelity = get_fidelity(full_circuit_probs, actual_probs)
    direct_fidelity = get_fidelity(direct_probs, actual_probs)
    likely_fidelity = get_fidelity(likely_probs, actual_probs)
    return full_fidelity, direct_fidelity, likely_fidelity


if __name__ == "__main__":
    num_qubits = 16
    num_clusters = 2
    repetitions = 10**6

    # construct and a random clustered circuit, and identify where it should be cut
    circuit, cuts = circuit_ansatz.random_clustered_circuit(num_qubits, num_clusters)

    # compute and print fidelities
    full_fidelity, direct_fidelity, likely_fidelity = get_fidelities(circuit, cuts, repetitions)
    print("full circuit fidelity:", full_fidelity)
    print("direct fidelity:", direct_fidelity)
    print("likely fidelity:", likely_fidelity)
