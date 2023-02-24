"""
Random clustered circuit for testing circuit cutting methods.

Author: Michael A. Perlin (github.com/perlinm)
"""
from typing import Iterable, Iterator, Tuple

import cirq


def random_clustered_circuit(
    num_qubits: int, num_clusters: int, num_layers: int = 1
) -> Tuple[cirq.Circuit, Iterable[Tuple[int, cirq.Qid]]]:
    """Construct a random clustered circuit, and identify where it should be cut."""
    assert num_qubits >= num_clusters
    assert num_layers >= 1

    clusters = [
        cirq.NamedQubit.range(
            num_qubits // num_clusters + int(cluster_index < num_qubits % num_clusters),
            prefix=f"c_{cluster_index}_",
        )
        for cluster_index in range(num_clusters)
    ]

    def intra_cluster_ops() -> Iterator[cirq.Operation]:
        for cluster in clusters:
            unitary = cirq.testing.random_unitary(2 ** len(cluster))
            yield cirq.MatrixGate(unitary).on(*cluster)

    def inter_cluster_ops() -> Iterator[cirq.Operation]:
        for cluster_a, cluster_b in zip(clusters[:-1], clusters[1:]):
            unitary = cirq.testing.random_unitary(4)
            qubits = cluster_a[-1], cluster_b[0]
            yield cirq.MatrixGate(unitary).on(*qubits)

    circuit = cirq.Circuit()
    for _ in range(num_layers):
        circuit += intra_cluster_ops()
        circuit += inter_cluster_ops()
    circuit += intra_cluster_ops()

    cuts = [
        (moment_index, cluster[-1])
        for moment_index in range(1, 2 * num_layers + 1)
        for cluster in clusters[:-1]
    ]

    return circuit, cuts
