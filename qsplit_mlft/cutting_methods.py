"""
A collection of methods for cutting and recombining circuits.

Author: Michael A. Perlin (github.com/perlinm)
"""
import collections
import functools
import itertools
import operator
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import cirq
import numpy as np
import numpy.typing as npt
import quimb.tensor as qtn
import scipy

BitString = Tuple[int, ...]
PrepBasis = Literal["Pauli", "SIC"]
PrepState = Literal["Z+", "Z-", "X+", "X-", "Y+", "Y-", "S0", "S1", "S2", "S3"]
MeasBasis = Literal["Z", "X", "Y"]

PrepStates = Tuple[PrepState, ...]
MeasBases = Tuple[MeasBasis, ...]

DEFAULT_PREP_BASIS: PrepBasis = "SIC"
PAULI_OPS: MeasBases = ("Z", "X", "Y")


@functools.lru_cache(maxsize=None)
def prep_state_to_proj(prep_state: PrepState) -> npt.NDArray[np.complex_]:
    """Convert a string into a projector onto the state of a qubit (flattened into a 1-D array)."""
    if prep_state == "Z+" or prep_state == "S0":
        vec = np.array([1, 0])
    elif prep_state == "Z-":
        vec = np.array([0, 1])
    elif prep_state == "X+":
        vec = np.array([1, 1]) / np.sqrt(2)
    elif prep_state == "X-":
        vec = np.array([1, -1]) / np.sqrt(2)
    elif prep_state == "Y+":
        vec = np.array([1, 1j]) / np.sqrt(2)
    elif prep_state == "Y-":
        vec = np.array([1, -1j]) / np.sqrt(2)
    elif prep_state in ["S1", "S2", "S3"]:
        corner_index = int(prep_state[1]) - 1
        azimuthal_angle = 2 * np.pi * corner_index / 3
        vec = np.array([1, np.exp(1j * azimuthal_angle) * np.sqrt(2)]) / np.sqrt(3)
    else:
        raise ValueError(f"state not recognized: {prep_state}")
    return np.outer(vec, vec.conj()).ravel()


def get_prep_states(prep_basis: PrepBasis) -> PrepStates:
    """
    Convert a string that specifies a tomographically complete basis of qubit states into a list of
    stings that specify the individual states.
    """
    states: PrepStates
    if prep_basis == "Pauli":
        states = ("Z+", "Z-", "X+", "X-", "Y+", "Y-")
    elif prep_basis == "SIC":
        states = ("S0", "S1", "S2", "S3")
    else:
        raise ValueError(f"tomographic basis not recognized: {prep_basis}")
    return states


####################################################################################################
# cutting a circuit into fragments


class Fragment:
    """
    Data structure for representing a fragment of a cut-up circuit.

    Fragments input/output qubits are partitioned into "circuit inputs", "circuit outputs",
    "quantum inputs", and "quantum outputs".
    - Circuit inputs/outputs = degrees of freedom at the beginning/end of the full (uncut) circuit.
      - Circuit inputs always start in |0>.
      - Circuit outputs are always measured in the computational basis.
    - Quantum inputs/outputs = degrees of freedom adjacent to a cut in the full circuit.
    Quantum inputs/outputs are specified by a dictionary that maps a qubit to a cut_name.
    """

    def __init__(
        self,
        circuit: cirq.AbstractCircuit,
        quantum_inputs: Optional[Mapping[cirq.Qid, str]] = None,
        quantum_outputs: Optional[Mapping[cirq.Qid, str]] = None,
    ) -> None:
        self.circuit = cirq.Circuit(circuit)
        self.quantum_inputs = quantum_inputs or {}
        self.quantum_outputs = quantum_outputs or {}
        assert all(qubit in circuit.all_qubits() for qubit in self.quantum_inputs)
        assert all(qubit in circuit.all_qubits() for qubit in self.quantum_outputs)
        self.circuit_outputs = sorted(circuit.all_qubits() - set(self.quantum_outputs))


def cut_circuit(
    circuit: cirq.AbstractCircuit, cuts: Iterable[Tuple[int, cirq.Qid]]
) -> Dict[str, Fragment]:
    """
    Cut a circuit into fragments.

    Strategy: rename qubits downstream of every cut in the circuit, then factorize circuit into
    separable subcircuits, and collect subcircuits and qubit routing data into Fragment objects.

    Args:
        - circuit: the circuit to be cut.
        - cuts: an iterable (e.g. tuple or list) cuts.  Each cut is specified by a
          (moment_index, qubit), such that all operations addressing the given qubit at or after the
          given moment_index are considered "downstream" of the cut.

    Returns:
        - fragments: dictionary that maps a fragment_key to a fragment.
    """
    circuit = cirq.Circuit(circuit)

    # keep track of all quantum inputs/outputs with dictionaries that map a qubit to a cut_name
    quantum_inputs: Dict[cirq.Qid, str] = {}
    quantum_outputs: Dict[cirq.Qid, str] = {}

    # map that keeps track of where qubits get routed through cuts
    initial_to_final_qubit_map: Dict[cirq.Qid, cirq.Qid] = {}

    cut_index = 0
    for cut_moment_index, cut_qubit in sorted(cuts):
        cut_name = f"cut_{cut_index}"

        # identify the qubits immediately before and after the cut
        old_qubit = initial_to_final_qubit_map.get(cut_qubit) or cut_qubit
        new_qubit = cirq.NamedQubit(cut_name)

        if old_qubit not in circuit[:cut_moment_index].all_qubits():
            # this is a "trivial" cut; there are no operations upstream of it
            continue

        # rename the old_qubit to the new_qubit in all operations downstream of the cut
        replacements = []  # ... to make to the circuit
        for moment_index, moment in enumerate(circuit[cut_moment_index:], start=cut_moment_index):
            for old_op in moment:
                if old_qubit in old_op.qubits:
                    new_op = old_op.transform_qubits({old_qubit: new_qubit})
                    replacements.append((moment_index, old_op, new_op))
                    break

        if not replacements:
            # this is a "trivial" cut; there are no operations downstream of it
            continue

        circuit.batch_replace(replacements)
        initial_to_final_qubit_map[cut_qubit] = new_qubit
        quantum_outputs[old_qubit] = quantum_inputs[new_qubit] = cut_name
        cut_index += 1

    # Factorize the circuit into independent subcircuits, and collect these subcircuits together
    # with their quantum inputs/outputs into Fragment objects.
    fragments = {}
    for fragment_index, subcircuit_moments in enumerate(circuit.factorize()):
        subcircuit = cirq.Circuit(subcircuit_moments)
        fragment_qubits = subcircuit.all_qubits()
        fragments[f"fragment_{fragment_index}"] = Fragment(
            subcircuit,
            {qubit: cut for qubit, cut in quantum_inputs.items() if qubit in fragment_qubits},
            {qubit: cut for qubit, cut in quantum_outputs.items() if qubit in fragment_qubits},
        )
    return fragments


####################################################################################################
# performing fragment tomography


"""
Fragment tomomography data is collected into a dictionary with type signature
  'Dict[BitString, ConditionalFragmentData]',
where the BitString key is a measurement outcome at the circuit output of the fragment, and
  'ConditionalFragmentData = Dict[Tuple[PrepStates, MeasBases, BitString], float]'
maps a (prepared_state, measurement_basis, measurement_outcome) at the quantum inputs/outputs to a
probability (float) that...
(1) the previously specified 'BitString's at the circuit/quantum outputs are measured, when
(2) preparing the given prepared_state at the quantum input, and
(3) measuring in the given measurement_basis at the quantum output.
"""
ConditionalFragmentData = Dict[Tuple[PrepStates, MeasBases, BitString], float]


class FragmentTomographyData:
    """Data structure for storing data collected from fragment tomography."""

    def __init__(
        self,
        fragment: Fragment,
        tomography_data: Dict[BitString, ConditionalFragmentData],
        prep_basis: PrepBasis,
    ) -> None:
        self.fragment = fragment
        self.data = tomography_data
        self.prep_basis = prep_basis

    def substrings(self) -> Iterator[BitString]:
        """Iterate over all measurement outcomes at the circuit outputs of this fragment."""
        yield from self.data.keys()

    def condition_on(self, substring: BitString) -> ConditionalFragmentData:
        """
        Get the ConditionalFragmentData associated with a fixed measurement outcome at this
        fragment's circuit outputs.
        """
        return self.data[substring]


def perform_fragment_tomography(
    fragments: Dict[str, Fragment],
    prep_basis: PrepBasis = DEFAULT_PREP_BASIS,
    repetitions: Optional[int] = None,
) -> Dict[str, FragmentTomographyData]:
    """Perform fragment tomography on a collection of fragments."""
    num_variants = sum(
        len(prep_basis) ** len(fragment.quantum_inputs)
        * len(PAULI_OPS) ** len(fragment.quantum_outputs)
        for fragment in fragments.values()
    )
    repetitions_per_variant = repetitions // num_variants
    return {
        fragment_key: perform_single_fragment_tomography(
            fragment, prep_basis, repetitions_per_variant
        )
        for fragment_key, fragment in fragments.items()
    }


def perform_single_fragment_tomography(
    fragment: Fragment,
    prep_basis: PrepBasis = DEFAULT_PREP_BASIS,
    repetitions_per_variant: Optional[int] = None,
    seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
) -> FragmentTomographyData:
    """
    Perform fragment tomography on the given fragment, using the specified tomographically complete
    basis of qubit states (which specifies which states to prepare at a fragment's quantum inputs).
    1. For a specified tomographically complete basis of qubit states, prepare all distinct
       combinations of quantum inputs.
    2. For each choice of prepared_state at the quantum inputs, measure quantum outputs in all
       distict combinations of Pauli bases.
    2. For each choice of (prepared_state, measurement_basis), compute a probability distribution
       over BitString measurement outcomes at the output qubits of the fragment.
    3. Split observed outcomes into the bitstrings at circuit outputs vs. quantum outputs, and
       organize all data into a FragmentTomographyData object.

    WARNING: this method makes no guarantee of efficiency.
    """
    quantum_inputs = list(fragment.quantum_inputs.keys())
    quantum_outputs = list(fragment.quantum_outputs.keys())
    circuit_outputs = fragment.circuit_outputs
    qubit_order = circuit_outputs + quantum_outputs
    num_qubits = len(qubit_order)
    if repetitions_per_variant:
        measurement = cirq.measure_each(*qubit_order)
        simulator = cirq.Simulator(seed=seed)

    tomography_data: Dict[BitString, ConditionalFragmentData] = collections.defaultdict(dict)
    for prep_states in itertools.product(get_prep_states(prep_basis), repeat=len(quantum_inputs)):
        circuit_with_prep = prep_state_ops(prep_states, quantum_inputs) + fragment.circuit

        if not repetitions_per_variant:
            # construct the state at the end of the fragment when preparing the prep_states
            prep_fragment_state = cirq.final_state_vector(
                circuit_with_prep, qubit_order=qubit_order
            )

        for meas_bases in itertools.product(PAULI_OPS, repeat=len(quantum_outputs)):
            # construct sub-circuit to measure in the 'meas_bases'
            meas_ops = meas_basis_ops(meas_bases, quantum_outputs)

            if not repetitions_per_variant:
                # get exact probability distribution over measurement outcomes
                final_state = cirq.final_state_vector(
                    cirq.Circuit(meas_ops),
                    initial_state=prep_fragment_state,
                    qubit_order=qubit_order,
                )
                probabilities = np.reshape(abs(final_state) ** 2, (2,) * num_qubits)

                # collect exact probabilities into the tomography_data object
                for circuit_outcome in itertools.product([0, 1], repeat=len(circuit_outputs)):
                    for quantum_outcome in itertools.product([0, 1], repeat=len(quantum_outputs)):
                        conditions = (prep_states, meas_bases, quantum_outcome)
                        probability = probabilities[circuit_outcome + quantum_outcome]
                        tomography_data[circuit_outcome][conditions] = probability

            else:  # simulate sampling from the true probability distribution
                full_circuit = circuit_with_prep + meas_ops + measurement
                results = simulator.run(full_circuit, repetitions=repetitions_per_variant)
                outcome_counter = results.multi_measurement_histogram(keys=qubit_order)

                # collect measurement outcomes into the tomography_data object
                for outcome, counts in outcome_counter.items():
                    circuit_outcome = outcome[: len(circuit_outputs)]
                    quantum_outcome = outcome[len(circuit_outputs) :]
                    # Record the fraction of times we observed this measurement outcome with the
                    # given prep_states/meas_bases.
                    conditions = (prep_states, meas_bases, quantum_outcome)
                    tomography_data[circuit_outcome][conditions] = counts / repetitions_per_variant

    # identify the cut indices at quantum inputs/outputs and return a FragmentTomographyData object
    return FragmentTomographyData(fragment, tomography_data, prep_basis)


def prep_state_ops(prep_states: PrepStates, qubits: Iterable[cirq.Qid]) -> Iterator[cirq.Operation]:
    """Return a circuit that prepares the given state on the given qubits (assumed to be in |0>)."""
    for prep_state, qubit in zip(prep_states, qubits):
        if prep_state == "Z+" or prep_state == "S0":
            continue
        elif prep_state == "Z-":
            yield cirq.X.on(qubit)
        elif prep_state == "X+":
            yield cirq.H.on(qubit)
        elif prep_state == "X-":
            yield cirq.X.on(qubit)
            yield cirq.H.on(qubit)
        elif prep_state == "Y+":
            yield cirq.H.on(qubit)
            yield cirq.S.on(qubit)
        elif prep_state == "Y-":
            yield cirq.H.on(qubit)
            yield cirq.inverse(cirq.S).on(qubit)
        elif prep_state in ["S1", "S2", "S3"]:
            polar_angle = 2 * np.arccos(1 / np.sqrt(3))  # cos(polar_angle/2) = 1/sqrt(3)
            yield cirq.ry(polar_angle).on(qubit)
            corner_index = int(prep_state[1]) - 1
            if corner_index != 0:
                azimuthal_angle = 2 * np.pi * corner_index / 3
                yield cirq.rz(azimuthal_angle).on(qubit)
        else:
            raise ValueError(f"state not recognized: {prep_state}")


def meas_basis_ops(meas_bases: MeasBases, qubits: Iterable[cirq.Qid]) -> Iterator[cirq.Operation]:
    """Return operations that map the given Pauli measurment basis onto the computational basis."""
    for basis, qubit in zip(meas_bases, qubits):
        if basis == "X":
            yield cirq.H.on(qubit)
        elif basis == "Y":
            yield cirq.inverse(cirq.S).on(qubit)
            yield cirq.H.on(qubit)


####################################################################################################
# building fragment models from fragment tomography data


class FragmentModel:
    """
    Data structure for representing a quantitative model of a fragment.

    Each fragment can be represented by a Choi matrix from the fragment's quantum inputs to its
    quantum outputs + circuit outputs.  This full Choi matrix is block diagonal, where each "block"
    is obtained by projecting onto a measurement outcome at the circuit outputs.  Model data is thus
    collected into a dictionary that maps a bitstring at the circuit outputs to a block of the Choi
    matrix.  Each block is, in turn, represented by a tensor whose indices are in one-to-one
    correspondence with the quantum inputs + quantum outputs of the fragment.
    """

    def __init__(self, fragment: Fragment, data: Dict[BitString, qtn.Tensor]) -> None:
        self.fragment = fragment
        self.data = data

    def substrings(self) -> Iterator[BitString]:
        """Iterate over measurement outcomes at the circuit outputs of this fragment."""
        yield from self.data.keys()

    def block(self, substring: BitString) -> qtn.Tensor:
        """Return the tensor in a single block of this fragment's Choi matrix."""
        return self.data[substring]

    def blocks(self) -> Iterator[Tuple[BitString, qtn.Tensor]]:
        """Iterate over all blocks of this fragment's Choi matrix."""
        yield from self.data.items()

    def num_blocks(self) -> int:
        return len(self.data)


def build_fragment_models(
    fragment_tomography_data_dict: Dict[str, FragmentTomographyData],
    *,
    rank_cutoff: float = 1e-8,
) -> Dict[str, FragmentModel]:
    """Convert a collection of fragment tomography data into a collection of fragment models."""
    return {
        fragment_key: build_single_fragment_model(
            frag_tomo_data, fragment_key=fragment_key, rank_cutoff=rank_cutoff
        )
        for fragment_key, frag_tomo_data in fragment_tomography_data_dict.items()
    }


def build_single_fragment_model(
    fragment_tomography_data: FragmentTomographyData,
    *,
    fragment_key: Optional[str] = None,
    rank_cutoff: float = 1e-8,
) -> FragmentModel:
    """Convert fragment tomography data into a fragment model."""
    data = {
        substring: build_conditional_fragment_model(fragment_tomography_data, substring)
        for substring in fragment_tomography_data.substrings()
    }
    return FragmentModel(fragment_tomography_data.fragment, data)


def build_conditional_fragment_model(
    fragment_tomography_data: FragmentTomographyData,
    substring: BitString,
    *,
    fragment_key: Optional[str] = None,
    rank_cutoff: float = 1e-8,
) -> qtn.Tensor:
    """
    Build a reduced Choi matrix (as a qtn.Tensor) that represents a fragment after conditioning on
    a fixed BitString measurement outcome at the fragment's circuit output.

    Args:
        - fragment_tomgraphy_data: the data collected from fagment tomography.
        - substring: a BitString at the circuit outputs of the fragment.
        - fragment_key (optional): a hashable key to identify this fragment.
        - rank_cutoff (optional): see documentation for the 'cond' argument of scipy.linalg.lstsq:
          https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html

    Returns:
        tensor: a qtn.Tensor object representing a reduced Choi matrix for this fragment.
    """
    # Identify the fragment data conditioned on the given circuit output, as well as the total
    # number of quantum inputs/outputs, and the tomographically complete basis used to prepare
    # states at the quantum inputs.
    conditional_fragment_data = fragment_tomography_data.condition_on(substring)
    prep_basis = fragment_tomography_data.prep_basis
    num_inputs = len(fragment_tomography_data.fragment.quantum_inputs)
    num_outputs = len(fragment_tomography_data.fragment.quantum_outputs)

    # Compute the reduced Choi matrix by least squares fitting to tomography data, by solving
    # a linear system of equations, 'A x = b', where
    # - 'x' is the (vectorized) Choi matrix,
    # - 'b' is a vector of probabilities for different measurement outcomes, and
    # - 'A' is a matrix whose rows are the (vectorized) operators whose expectation values (with
    #   respect to the Choi matrix 'x') are the probabilities in 'b'.
    interrogation_operators = interrogation_matrix(num_inputs, num_outputs, prep_basis)  # 'A'
    interrogation_outcomes = [  # 'b'
        conditional_fragment_data.get(condition) or 0
        for condition in condition_iterator(num_inputs, num_outputs, prep_basis)
    ]
    reduced_choi_matrix = scipy.linalg.lstsq(
        interrogation_operators,
        interrogation_outcomes,
        cond=rank_cutoff,
    )[0]

    # Factorize the reduced Choi matrix into tensor factors associated with individual qubit degrees
    # of freedom, and return as a qtn.Tensor object in which each tensor factor (index) is labeled
    # by a corresponding cut_name.
    choi_tensor = reduced_choi_matrix.reshape((4,) * (num_inputs + num_outputs))
    cuts_at_inputs = list(fragment_tomography_data.fragment.quantum_inputs.values())
    cuts_at_outputs = list(fragment_tomography_data.fragment.quantum_outputs.values())
    cut_indices = cuts_at_inputs + cuts_at_outputs
    tags = (fragment_key,) if fragment_key is not None else None
    return qtn.Tensor(choi_tensor, inds=cut_indices, tags=tags)


@functools.lru_cache(maxsize=None)
def interrogation_matrix(
    num_inputs: int,
    num_outputs: int,
    prep_basis: PrepBasis = DEFAULT_PREP_BASIS,
) -> npt.NDArray[np.complex_]:
    """
    Construct a matrix whose rows are the operators (flattened to 1-D arrays) that are interrogated
    by fragment tomography.
    """
    condition_vecs = [
        conditions_to_vec(prep_states, meas_bases, meas_outcome)
        for prep_states, meas_bases, meas_outcome in condition_iterator(
            num_inputs, num_outputs, prep_basis
        )
    ]
    return np.array(condition_vecs)


def conditions_to_vec(
    prep_states: PrepStates,
    meas_bases: MeasBases,
    meas_outcome: BitString,
) -> npt.NDArray[np.complex_]:
    """
    Convert a choice of (prepared_states, measurement_bases, measurement_outcome) at the quantum
    inputs/outputs of a fragment into the operator (flattened to a 1-D array) for a corresponding
    matrix element of that fragment.
    """
    out_strs = [basis + ("+" if bit == 0 else "-") for basis, bit in zip(meas_bases, meas_outcome)]
    out_vecs = [prep_state_to_proj(out_str) for out_str in out_strs]
    inp_vecs = [prep_state_to_proj(inp_str) for inp_str in prep_states]
    out_vec = functools.reduce(np.kron, out_vecs, np.array([1]))
    inp_vec = functools.reduce(np.kron, inp_vecs, np.array([1]))
    return np.kron(inp_vec.conj(), out_vec)


def condition_iterator(
    num_inputs: int,
    num_outputs: int,
    prep_basis: PrepBasis = DEFAULT_PREP_BASIS,
) -> Iterator[Tuple[PrepStates, MeasBases, BitString]]:
    """
    Iterate over all choices of (prepared_states, measurement_bases, measurment_outcome) for the
    quantum inputs/outputs of a fragment with a given number of quantum inputs/outputs (and a given
    choice of tomographically complete basis of qubit states).
    """
    for prep_states in itertools.product(get_prep_states(prep_basis), repeat=num_inputs):
        for meas_bases in itertools.product(PAULI_OPS, repeat=num_outputs):
            for meas_outcome in itertools.product([0, 1], repeat=num_outputs):
                yield prep_states, meas_bases, meas_outcome


####################################################################################################
# applying maximum-likelihood corrections to fragment models


def corrected_fragment_models(
    fragment_models: Dict[str, FragmentModel]
) -> Dict[str, FragmentModel]:
    """
    Apply maximum-likelihood corrections to a collection of fragment models.

    The general strategy, taken from arXiv:1106.5458, is to diagonalize the Choi matrix for each
    fragment, and eliminate negative eigenvalues one by one in order of decreasing magnitude.  Every
    time a negative eigenvalue is eliminated, its value is distributed evenly among all other
    eigenvalues, such that the trace of the Choi matrix is preserved.  This process is repeated
    until the Choi matrix has no more negative eigenvalues.

    TODO: find the maximum-likelihood model for an isometric channel.
    """
    return {
        fragment_key: corrected_single_fragment_model(fragment_model)
        for fragment_key, fragment_model in fragment_models.items()
    }


def corrected_single_fragment_model(fragment_model: FragmentModel) -> FragmentModel:
    """Apply maximum-likelihood corrections to a fragment model."""
    num_inputs = len(fragment_model.fragment.quantum_inputs)
    num_outputs = len(fragment_model.fragment.quantum_outputs)
    num_qubits = num_inputs + num_outputs

    # compute all eigenvalues and eigenvectors of the Choi matrix for this fragment
    eigenvalues = np.empty((fragment_model.num_blocks(), 2**num_qubits))
    eigenvectors = np.empty(
        (fragment_model.num_blocks(), 2**num_qubits, 2**num_qubits), dtype=complex
    )
    for idx, (circuit_outcome, block_tensor) in enumerate(fragment_model.blocks()):
        # construct the Choi matrix for this block
        block_choi_matrix = np.moveaxis(
            block_tensor.data.reshape((2,) * (num_qubits * 2)),
            range(1, 2 * num_qubits, 2),
            range(num_qubits, 2 * num_qubits),
        ).reshape((2**num_qubits,) * 2)
        block_eig_vals, block_eig_vecs = np.linalg.eigh(block_choi_matrix)
        eigenvalues[idx] = block_eig_vals
        eigenvectors[idx] = block_eig_vecs
    eigenvalues = correct_probability_distribution(eigenvalues)

    # Iterate over all blocks of the Choi matrix, and reconstruct them from the corrected
    # eigenvalues and their corresponding eigenvectors.
    corrected_data = {}
    for (circuit_outcome, block_tensor), block_eig_vals, block_eig_vecs in zip(
        fragment_model.blocks(), eigenvalues, eigenvectors
    ):
        corrected_block_choi_matrix = sum(
            val * np.outer(vec, vec.conj()) for val, vec in zip(block_eig_vals, block_eig_vecs.T)
        )
        # convert the corrected Choi matrix back into a tensor
        corrected_block_tensor_data = np.moveaxis(
            corrected_block_choi_matrix.reshape((2,) * (num_qubits * 2)),
            range(num_qubits, 2 * num_qubits),
            range(1, 2 * num_qubits, 2),
        ).reshape((4,) * num_qubits)
        corrected_data[circuit_outcome] = qtn.Tensor(
            corrected_block_tensor_data, inds=block_tensor.inds, tags=block_tensor.tags
        )

    return FragmentModel(fragment_model.fragment, corrected_data)


def correct_probability_distribution(probabilities: npt.NDArray[float]) -> npt.NDArray[float]:
    """Apply maximum-likelihood corrections to a classical probability distribution.

    Eliminate negative probabilities one by one in order of decreasing magnitude, and distribute
    their values among all other probabilities.  Method taken from arXiv:1106.5458.
    """
    prob_order = np.argsort(probabilities.ravel())
    sorted_probabilities = probabilities.ravel()[prob_order]
    for idx, val in enumerate(sorted_probabilities):
        if val >= 0:
            break
        sorted_probabilities[idx] = 0
        num_vals_remaining = probabilities.size - idx - 1
        sorted_probabilities[idx + 1 :] += val / num_vals_remaining
    inverse_sort = np.arange(probabilities.size)[np.argsort(prob_order)]
    corrected_probabilities = sorted_probabilities[inverse_sort]
    corrected_probabilities.shape = probabilities.shape
    return corrected_probabilities


####################################################################################################
# recombining fragment models


def recombine_fragment_models(
    fragment_models: Dict[str, FragmentModel],
    qubit_order: Optional[Sequence[cirq.Qid]] = None,
) -> Dict[BitString, float]:
    """
    Recombine fragment models into a probability distribution over BitString measurement outcomes
    for the full, un-cut circuit.
    """
    recombined_distribution = {}

    fragments = {
        fragment_index: fragment_model.fragment
        for fragment_index, fragment_model in fragment_models.items()
    }
    contraction_path = get_contraction_path(fragment_models)
    outcome_combiner = get_outcome_combiner(fragments, qubit_order)

    # loop over all choices of bitstrings at the circuit outputs of fragments
    frag_keys = list(fragment_models.keys())
    frag_circuit_outputs = [
        list(frag_model.substrings()) for frag_model in fragment_models.values()
    ]
    for fragment_substrings in itertools.product(*frag_circuit_outputs):
        # Combine measurement outcomes on fragments into a measurement outcome on
        # the full, uncut circuit.
        measurement_outcomes = dict(zip(frag_keys, fragment_substrings))
        combined_measurement_outcome = outcome_combiner(measurement_outcomes)

        # collect all fragment tensors into a tensor network, and contract it
        tensors = [
            fragment_models[frag_key].block(substring)
            for frag_key, substring in measurement_outcomes.items()
        ]
        network = qtn.TensorNetwork(tensors)
        recombined_distribution[combined_measurement_outcome] = network.contract(
            optimize=contraction_path
        ).real

    return recombined_distribution


def get_contraction_path(fragment_models: Dict[str, FragmentModel]) -> List[Tuple[int, int]]:
    """Compute a tensor network contraction path for the given fragment models."""
    tensors = [next(iter(model.blocks()))[1] for model in fragment_models.values()]
    return qtn.TensorNetwork(tensors).contraction_path()


def get_outcome_combiner(
    fragments: Dict[str, Fragment],
    qubit_order: Optional[Sequence[cirq.Qid]] = None,
) -> Callable[[Dict[str, BitString]], BitString]:
    """
    Construct a function that that combines substrings at the circuit outputs of fragments.

    Args:
        - fragments: the fragments that need to be recombined, stored in a dictionary that maps a
          fragment_key to a Fragment.
        - qubit_order (optional): the order of the qubits in the reconstructed circuit.  If a qubit
          order is not provided, this defaults to qubit_order = sorted(reconstructed_circuit_qubits)

    Returns:
        - outcome_combiner: a function that recombines measurement outcomes at fragments into a
          measurment outcome for the full, uncut circuit.
            Args:
                - fragment_outcomes: a dictionary that maps a fragment_key to a measurement outcome
                  (BitString) at the circuit outputs of the corresponding fragment.
            Returns:
                - recombined_circuit_output: a recombined measurement outcome (BitString).
    """
    # collect some data about qubits and cuts
    circuit_qubits: List[cirq.Qid] = []  # all qubits addressed by the reconstructed circuit
    qubit_to_cut: Dict[cirq.Qid, str] = {}  # map from a qubit before a cut to the cut_name
    cut_to_qubit: Dict[str, cirq.Qid] = {}  # map from a cut_name to the qubit after the cut
    for fragment in fragments.values():
        circuit_qubits.extend(fragment.circuit.all_qubits() - set(fragment.quantum_inputs))
        qubit_to_cut.update(fragment.quantum_outputs)
        cut_to_qubit.update({cut: qubit for qubit, cut in fragment.quantum_inputs.items()})

    # construct a map that tracks where each qubit gets routed through the fragments
    initial_to_final_qubit_map = {}
    for circuit_qubit in circuit_qubits:
        initial_to_final_qubit_map[circuit_qubit] = circuit_qubit
        while (qubit := initial_to_final_qubit_map[circuit_qubit]) in qubit_to_cut:
            initial_to_final_qubit_map[circuit_qubit] = cut_to_qubit[qubit_to_cut[qubit]]

    # identify the order of the "final" qubits at the ends of fragments
    if qubit_order is None:
        qubit_order = sorted(circuit_qubits)
    final_qubit_order = [initial_to_final_qubit_map.get(qubit) or qubit for qubit in qubit_order]

    # Identify the permutation that needs to be applied to the concatenation of fragment substrings
    # in order to get the bits of the combined string in the right order.
    fragment_keys = list(fragments.keys())
    fragment_outputs = [fragments[key].circuit_outputs for key in fragment_keys]
    contacenated_qubits = functools.reduce(operator.add, fragment_outputs)
    bit_permutation = [contacenated_qubits.index(qubit) for qubit in final_qubit_order]

    def outcome_combiner(fragment_substrings: Dict[str, BitString]) -> BitString:
        """
        Combine measurement outcomes at the circuit outputs of fragments into an overall
        measurement outcome for the full, uncut circuit.
        """
        substrings = [fragment_substrings[key] for key in fragment_keys]
        concatenated_substring = functools.reduce(operator.add, substrings)
        return tuple(concatenated_substring[index] for index in bit_permutation)

    return outcome_combiner
