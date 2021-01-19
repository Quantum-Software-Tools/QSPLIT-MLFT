#!/usr/bin/env python3

# author: Michael A. Perlin (github.com/perlinm)

import ast, itertools, numpy, scipy, qiskit, tensornetwork
from qiskit.tools import monitor

prep_state_keys = { "Pauli" : [ "Zp", "Zm", "Xp", "Yp" ],
                    "SIC" : [ "S0", "S1", "S2", "S3" ] }
meas_state_keys = { "Pauli" : [ "Zp", "Zm", "Xp", "Xm", "Yp", "Ym" ],
                    "SIC" : [ "S0", "S1", "S2", "S3" ] }

# get the quantum state prepared by a circuit
def get_statevector(circuit):
    simulator = qiskit.Aer.get_backend("statevector_simulator")
    sim_job = qiskit.execute(circuit, simulator)
    return sim_job.result().get_statevector(circuit)

# run circuits and get the resulting probability distributions
def run_circuits(circuits, shots, backend = "qasm_simulator",
                 max_hardware_shots = 8192, monitor_jobs = False):

    # get results from a single run
    def _results(shots, backend):
        tomo_job = qiskit.execute(circuits, backend = backend, shots = shots)
        if monitor_jobs: qiskit.tools.monitor.job_monitor(tomo_job)
        return tomo_job.result()

    if type(backend) is str:
        # if the backend is a string, simulate locally with a qiskit Aer backend
        backend = qiskit.Aer.get_backend(backend)
        backend._configuration.max_shots = shots
        return [ _results(shots, backend) ]

    else:
        # otherwise, we're presumably running on hardware,
        #   so only run as many shots at a time as we're allowed
        max_shot_repeats = shots // max_hardware_shots
        shot_remainder = shots % max_hardware_shots
        shot_sequence = [ max_hardware_shots ] * max_shot_repeats \
                      + [ shot_remainder ] * ( shot_remainder > 0 )
        return [ _results(shots, backend) for shots in shot_sequence ]

# convert a statevector into a density operator
def to_projector(vector):
    return numpy.outer(numpy.conjugate(vector), vector)

# fidelity of two density operators
def state_fidelity(rho, sigma):
    assert(rho.ndim in [ 1, 2 ])
    assert(sigma.ndim in [ 1, 2 ])
    if rho.ndim == 1: rho = to_projector(rho)
    if sigma.ndim == 1: sigma = to_projector(sigma)

    sqrt_rho = scipy.linalg.sqrtm(rho)
    return abs(numpy.trace(scipy.linalg.sqrtm(sqrt_rho @ sigma @ sqrt_rho)))**2

# relative entropy S( P | Q ) \equiv tr( P log(P/Q) ) in bits
# interpretation: information lost by using Q (the "approximate" distribution)
#                 to approximate P (the "actual" distribution)
def relative_entropy(actual_dist, approx_dist):
    def _term(bits):
        actual_val = actual_dist[bits]
        if actual_val == 0:
            return 0

        approx_val = approx_dist[bits]
        if approx_val is None:
            return numpy.nan

        return actual_val * numpy.log(actual_val/approx_val) / numpy.log(2)
    return sum( _term(bits) for bits in actual_dist.keys() )

# Euclidean distance between two channels
def channel_distance(rho, sigma):
    rho /= rho.trace()
    sigma /= sigma.trace()
    return numpy.linalg.matrix_power(rho-sigma,2).trace()

##########################################################################################
# methods for peforming process tomography and organizing associated data
##########################################################################################

# identify all stitches in a cut-up circuit, in dictionary format:
#   { <exit wire> : <init wire> }
def identify_stitches(wire_path_map):
    circuit_wires = list(wire_path_map.keys())
    stitches = {}
    for wire in circuit_wires:
        # identify all init/exit wires in the path of this wire
        init_wires = wire_path_map[wire][1:]
        exit_wires = wire_path_map[wire][:-1]
        # add the stitches in this path
        stitches.update({ exit_wire : init_wire
                          for init_wire, exit_wire in zip(init_wires, exit_wires) })
    return stitches

# identify preparation / meauserment qubits for all fragments
def identify_frag_targets(wire_path_map):
    stitches = identify_stitches(wire_path_map)
    frag_targets = {}
    for meas_frag_qubit, prep_frag_qubit in stitches.items():
        meas_frag, meas_qubit = meas_frag_qubit
        prep_frag, prep_qubit = prep_frag_qubit
        for frag in [ meas_frag, prep_frag ]:
            if frag not in frag_targets:
                frag_targets[frag] = { "meas" : (), "prep" : () }
        frag_targets[meas_frag]["meas"] += (meas_qubit,)
        frag_targets[prep_frag]["prep"] += (prep_qubit,)
    return frag_targets

# perform partial quantum tomography on a circuit and return the corresponding raw data
def partial_tomography(circuit, prep_qubits, meas_qubits, shots, prep_basis,
                       tomography_backend = "qasm_simulator", monitor_jobs = False):
    if prep_qubits == None: perp_qubits = []
    if meas_qubits == None: meas_qubits = []
    if prep_qubits == "all": prep_qubits = circuit.qubits
    if meas_qubits == "all": meas_qubits = circuit.qubits
    total_qubit_num = len(circuit.qubits)

    # convert qubit objects to qubit indices (i.e. in a quantum register)
    def _qubit_index(qubit):
        if type(qubit) is int: return qubit
        else: return qubit.index
    prep_qubits = list(map(_qubit_index, prep_qubits))
    meas_qubits = list(map(_qubit_index, meas_qubits))

    # define preparation states and measurement bases
    prep_states = prep_state_keys[prep_basis]
    meas_ops = [ "Z", "X", "Y" ]

    # collect preparation / measurement labels for all circuit variants
    def _reorder(label, positions):
        return tuple( label[positions.index(qq)] for qq in range(len(label)) )
    def _full_label(part_label, part_qubits, pad):
        qubit_order = list(part_qubits) + [ qq for qq in range(total_qubit_num)
                                            if qq not in part_qubits ]
        full_pad = [pad] * ( total_qubit_num - len(part_qubits) )
        return _reorder(list(part_label) + full_pad, qubit_order)
    def _full_prep_label(prep_label):
        return _full_label(prep_label, prep_qubits, prep_states[0])
    def _full_meas_label(meas_label):
        return _full_label(meas_label, meas_qubits, "Z")

    prep_labels = list(itertools.product(prep_states, repeat = len(prep_qubits)))
    meas_labels = list(itertools.product(meas_ops, repeat = len(meas_qubits)))

    # define full preparation / measurment labels on *all* qubits
    full_prep_labels = list(map(_full_prep_label, prep_labels))
    full_meas_labels = list(map(_full_meas_label, meas_labels))

    # collect circuit variants for peforming tomography
    get_tomo_circuits = qiskit.ignis.verification.tomography.process_tomography_circuits
    tomo_circuits = get_tomo_circuits(circuit, circuit.qubits,
                                      prep_basis = prep_basis,
                                      prep_labels = full_prep_labels,
                                      meas_labels = full_meas_labels)

    return run_circuits(tomo_circuits, shots, tomography_backend, monitor_jobs = monitor_jobs)

# organize raw tomography data into a dictionary of dictionaries,
#   mapping (1) bitstrings on the "final" qubits
#       --> (2) prepared / measured state labels
#       --> (3) observed counts, i.e.
# { <final bitstring> : { <prepared_measured_states> : <counts> } }
def organize_tomography_data(raw_data_collection, prep_qubits, meas_qubits, prep_basis):
    def _qubit_index(qubit):
        if type(qubit) is int: return qubit
        else: return qubit.index
    prep_qubits = list(map(_qubit_index, prep_qubits))
    meas_qubits = list(map(_qubit_index, meas_qubits))

    # split a bitstring on all qubits into:
    #   (1) a bitstring on the "middle" qubits that are associated with a cut, and
    #   (2) a bitstring on the "final" qubits that are *not* associated with a cut
    def _split_bits(bits):
        mid_bits = "".join([ bits[len(bits)-idx-1] for idx in meas_qubits ])
        fin_bits = "".join([ bit for pos, bit in enumerate(bits)
                             if len(bits)-pos-1 not in meas_qubits ])
        return mid_bits, fin_bits

    organized_data = {}
    for raw_data in raw_data_collection:
        for result in raw_data.results:
            name = result.header.name
            meas_counts = raw_data.get_counts(name)
            full_prep_label, full_meas_label = ast.literal_eval(name)

            prep_label = tuple( full_prep_label[qubit] for qubit in prep_qubits )
            meas_label = tuple( full_meas_label[qubit] for qubit in meas_qubits )
            for bits, counts in meas_counts.items():
                meas_bits, final_bits = _split_bits(bits)
                meas_state = tuple( basis + ( "p" if outcome == "0" else "m" )
                                    for basis, outcome in zip(meas_label, meas_bits) )
                count_label = ( prep_label, meas_state )
                if final_bits not in organized_data:
                    organized_data[final_bits] = {}
                organized_data[final_bits][count_label] = counts

    # add zero count data for output strings with missing prep/meas combinations
    prep_labels = itertools.product(prep_state_keys[prep_basis], repeat = len(prep_qubits))
    meas_states = itertools.product(meas_state_keys["Pauli"], repeat = len(meas_qubits))
    count_labels = list(itertools.product(prep_labels, meas_states))
    for bits in organized_data.keys():
        if len(organized_data[bits]) == len(count_labels): continue
        for count_label in count_labels:
            if count_label not in organized_data[bits]:
                organized_data[bits][count_label] = 0

    return organized_data

# perform process tomography on all fragments and return the corresponding data
def collect_fragment_data(fragments, wire_path_map, shots,
                          tomography_backend = "qasm_simulator",
                          prep_basis = "SIC", monitor_jobs = False):
    frag_targets = identify_frag_targets(wire_path_map)
    frag_raw_data = [ partial_tomography(fragment,
                                         frag_targets[idx].get("prep"),
                                         frag_targets[idx].get("meas"),
                                         shots = shots, prep_basis = prep_basis,
                                         tomography_backend = tomography_backend,
                                         monitor_jobs = monitor_jobs)
                      for idx, fragment in enumerate(fragments) ]

    return [ organize_tomography_data(raw_data,
                                      frag_targets[idx].get("prep"),
                                      frag_targets[idx].get("meas"),
                                      prep_basis = prep_basis)
             for idx, raw_data in enumerate(frag_raw_data) ]

##########################################################################################
# methods for building maximum likelihood models of a circuit
##########################################################################################

# convert string label to a matrix
def label_to_matrix(label):
    if label == "I": return numpy.eye(2)

    bases = qiskit.ignis.verification.tomography.basis
    if label[0] in [ "X", "Y", "Z" ]:
        matrix = bases.paulibasis.pauli_preparation_matrix
        if len(label) == 1:
            return matrix(label+"p") - matrix(label+"m")
        else:
            return matrix(label)
    if label[0] == "S":
        return bases.sicbasis.sicpovm_preparation_matrix(label)

    raise ValueError(f"label not recognized: {label}")

# convert a tuple of preparation / measurement labels into a choi matrix element
def target_labels_to_matrix(targets):
    try:
        prep_labels, meas_labels = targets["prep"], targets["meas"]
    except:
        prep_labels, meas_labels = targets
    prep_matrix = numpy.array(1)
    meas_matrix = numpy.array(1)
    for label in prep_labels:
        prep_matrix = numpy.kron(label_to_matrix(label), prep_matrix)
    for label in meas_labels:
        meas_matrix = numpy.kron(label_to_matrix(label), meas_matrix)
    return numpy.kron(prep_matrix.T, meas_matrix)

# use tomography data to build a "naive" model (i.e. choi matrix) for a circuit fragment.
# `tomography_data` should be a dictionary of dictionaries, mapping
#   <bitstring on "final" (classical) outputs of fragment>
#   --> <preparation / measurement labels>
#   --> <number of counts>
def direct_fragment_model(tomography_data, discard_poor_data = False, rank_cutoff = 1e-8):
    # if we were given a list of data sets, build a model for each data set in the list
    if type(tomography_data) is list:
        return [ direct_fragment_model(data_set, discard_poor_data, rank_cutoff)
                 for data_set in tomography_data ]

    # build a block-diagonal choi matrix from experiment data,
    #   where each block corresponds to a unique bitstring
    #   on the "final" outputs of a fragent
    choi_matrix = {}
    for final_bits, fixed_bit_data in tomography_data.items():
        prep_meas_states, state_counts = zip(*fixed_bit_data.items())
        prep_labels, meas_labels = zip(*prep_meas_states)
        prep_qubit_num = len(prep_labels[0])
        meas_qubit_num = len(meas_labels[0])
        if discard_poor_data:
            # if our system of equations defining this block of the choi matrix
            #   is underdetermined, don't bother fitting
            degrees_of_freedom = 4**( prep_qubit_num + meas_qubit_num )
            if len(fixed_bit_data) < degrees_of_freedom:
                print(f"discarding {sum(state_counts)} counts that define" +
                      " an underdetermined system of equations")
                continue

        # total number of cut qubits
        cut_qubit_num = prep_qubit_num + meas_qubit_num

        # trace of the choi matrix we're fitting
        choi_trace = sum(state_counts) / ( 2**prep_qubit_num * 3**meas_qubit_num )

        # collect data for fitting procedure, in which we will find a vector choi_fit
        #   that minimizes | state_matrix.conj() @ choi_fit - state_counts |
        state_matrix = numpy.array([ target_labels_to_matrix(states).flatten()
                                     for states in prep_meas_states ] +
                                   [ numpy.eye(2**cut_qubit_num).flatten() ])
        state_counts = numpy.array(list(state_counts) + [ choi_trace ])

        # TODO: add count-adjusted weights to fitting procedure
        choi_fit = scipy.linalg.lstsq(state_matrix.conj(), state_counts,
                                      cond = rank_cutoff)[0]

        # save the fitted choi matrix
        choi_matrix[final_bits] = choi_fit.reshape(2**cut_qubit_num,2**cut_qubit_num)

    return choi_matrix

# find the closest nonnegative choi matrix to a "naive" one (see arXiv:1106.5458)
def maximum_likelihood_model(choi_matrix):
    # if we were given a list of models,
    #   then build maximum likelihood model for each data set in the list
    if type(choi_matrix) is list:
        return [ maximum_likelihood_model(mat) for mat in choi_matrix ]

    # diagonalize each block of the choi matrix
    choi_eigs = {}
    choi_vecs = {}
    for final_bits, choi_block in choi_matrix.items():
        choi_eigs[final_bits], choi_vecs[final_bits] = scipy.linalg.eigh(choi_block)

    # find the eigenvalues of the closest nonnegative choi matrix
    all_eigs = numpy.concatenate(list(choi_eigs.values()))
    eig_order = numpy.argsort(all_eigs)
    sorted_eigs = all_eigs[eig_order]
    dim = len(sorted_eigs)
    for idx in range(dim):
        val = sorted_eigs[idx]
        if val >= 0: break
        sorted_eigs[idx] = 0
        sorted_eigs[idx+1:] += val / ( dim - (idx+1) )
    reverse_order = numpy.arange(dim)[numpy.argsort(eig_order)]
    all_eigs = sorted_eigs[reverse_order]

    # organize eigenvalues back into their respective blocks
    num_blocks = len(choi_eigs)
    block_size = dim // num_blocks
    all_eigs = numpy.reshape(all_eigs, (num_blocks, block_size))
    choi_eigs = { bits : all_eigs[idx,:]
                  for idx, bits in enumerate(choi_eigs.keys())
                  if not numpy.count_nonzero(all_eigs[idx,:]) == 0 }

    # reconstruct choi matrix from eigenvalues / eigenvectors
    return { bits : sum( val * to_projector(choi_vecs[bits][:,idx])
                         for idx, val in enumerate(vals) if val > 0 )
             for bits, vals in choi_eigs.items() }

##########################################################################################
# methods for recombining fragment models
##########################################################################################

# recombine fragment data by inserting a complete basis of operators
def _recombine_using_insertions(frag_models, wire_path_map):
    frag_num = len(frag_models)
    stitches = identify_stitches(wire_path_map)
    frag_targets = identify_frag_targets(wire_path_map)

    # identify permutation to apply to recombined fragment output
    final_bit_pieces = [ list(choi.keys()) for choi in frag_models ]
    bit_permutation = united_axis_permutation(wire_path_map)

    combined_dist = {}
    for stitch_ops in itertools.product(["I","Z","X","Y"], repeat = len(stitches)):
        frag_ops = { idx : { "prep" : {} , "meas" : {} }
                     for idx in range(frag_num) }
        for stitch_op, stitch_qubits in zip(stitch_ops, stitches.items()):
            meas_frag_qubit, prep_frag_qubit = stitch_qubits
            meas_frag, meas_qubit = meas_frag_qubit
            prep_frag, prep_qubit = prep_frag_qubit
            meas_idx = frag_targets[meas_frag]["meas"].index(meas_qubit)
            prep_idx = frag_targets[prep_frag]["prep"].index(prep_qubit)
            frag_ops[meas_frag]["meas"][meas_idx] = stitch_op
            frag_ops[prep_frag]["prep"][prep_idx] = stitch_op

        frag_ops = [ frag_ops[idx] for idx in range(frag_num) ]
        def _ops_to_labels(ops_dict):
            labels = {}
            labels["prep"] \
                = tuple( ops_dict["prep"][idx] for idx in range(len(ops_dict["prep"])) )
            labels["meas"] \
                = tuple( ops_dict["meas"][idx] for idx in range(len(ops_dict["meas"])) )
            return labels

        frag_labels = list(map(_ops_to_labels, frag_ops))
        frag_mats = list(map(target_labels_to_matrix, frag_labels))

        for frag_bits in itertools.product(*final_bit_pieces):
            joined_bits = "".join(frag_bits[::-1])
            final_bits = "".join([ joined_bits[idx] for idx in bit_permutation ])
            frag_vals = [ mat.flatten().conj() @ choi[bits].flatten()
                          for choi, bits, mat
                          in zip(frag_models, frag_bits, frag_mats) ]
            val = numpy.product(frag_vals).real
            try:
                combined_dist[final_bits] += val
            except:
                combined_dist[final_bits] = val

    return combined_dist

# recombine fragment data by building and contracting tensor networks
def _recombine_using_networks(frag_models, wire_path_map):
    frag_num = len(frag_models)
    stitches = identify_stitches(wire_path_map)
    frag_targets = identify_frag_targets(wire_path_map)

    # identify permutation to apply to recombined fragment output
    final_bit_pieces = [ list(choi.keys()) for choi in frag_models ]
    bit_permutation = united_axis_permutation(wire_path_map)

    combined_dist = {}
    for frag_bits in itertools.product(*final_bit_pieces):
        joined_bits = "".join(frag_bits[::-1])
        final_bits = "".join([ joined_bits[idx] for idx in bit_permutation ])

        nodes = {}
        for choi, bits, idx in zip(frag_models, frag_bits, range(frag_num)):
            matrix = choi[bits]
            qubits =  (len(bin(matrix.shape[0]))-3)
            tensor = matrix.reshape((2,)*2*qubits)

            prep_qubits = len(frag_targets[idx]["prep"])
            prep_axes_bra = reversed(range(prep_qubits))
            meas_axes_ket = reversed(range(prep_qubits,qubits))
            prep_axes_ket = reversed(range(qubits,qubits+prep_qubits))
            meas_axes_bra = reversed(range(qubits+prep_qubits,2*qubits))
            prep_axes = numpy.array(list(zip(prep_axes_bra, prep_axes_ket))).flatten()
            meas_axes = numpy.array(list(zip(meas_axes_ket, meas_axes_bra))).flatten()
            tensor = tensor.transpose(list(prep_axes) + list(meas_axes))
            tensor = tensor.reshape((4,)*qubits)

            nodes[idx] = tensornetwork.Node(tensor)

        for meas_frag_qubit, prep_frag_qubit in stitches.items():
            meas_frag, meas_qubit = meas_frag_qubit
            prep_frag, prep_qubit = prep_frag_qubit

            meas_qubit_idx = frag_targets[meas_frag]["meas"].index(meas_qubit)
            prep_qubit_idx = frag_targets[prep_frag]["prep"].index(prep_qubit)

            prep_axis = prep_qubit_idx
            meas_axis = len(frag_targets[meas_frag]["prep"]) + meas_qubit_idx
            nodes[meas_frag][meas_axis] ^ nodes[prep_frag][prep_axis]

        val = tensornetwork.contractors.greedy(nodes.values()).tensor.real
        try:
            combined_dist[final_bits] += val
        except:
            combined_dist[final_bits] = val

    return combined_dist

def recombine_fragment_models(*args, method = "network", **kwargs):
    if method == "network":
        recombination_method = _recombine_using_networks
    elif method == "insertion":
        recombination_method = _recombine_using_insertions
    else:
        raise ValueError("recombination method {method} not recognized")
    combined_dist = recombination_method(*args, **kwargs)
    combined_norm = sum(combined_dist.values())
    return { bits : val / combined_norm for bits, val in combined_dist.items() }

##########################################################################################
# TODO: cleanup all of the code below, which is currently just borrowed from old codes.
# when recombining fragment results, we concatenate fragments' "final" bitstrings.
# the methods below are used to determine the permutation we need to apply
#   to the concatenated bitstring in order to put the bits in the "correct" order, i.e.
#   in which these bits appear in the full, uncut ciruit
##########################################################################################

# read a wire path map to identify init / exit wires for all fragments
def _identify_init_exit_wires(wire_path_map, num_fragments):
    # collect all exit/init wires in format ( frag_index, wire )
    all_init_wires = set()
    all_exit_wires = set()

    # loop over all paths to identify init/exit wires
    for path in wire_path_map.values():
        all_init_wires.update(path[1:])
        all_exit_wires.update(path[:-1])

    # collect init/exit wires within each fragment, sorting them to fix their order
    init_wires = tuple([ { wire for idx, wire in all_init_wires if idx == frag_idx }
                         for frag_idx in range(num_fragments) ])
    exit_wires = tuple([ { wire for idx, wire in all_exit_wires if idx == frag_idx }
                         for frag_idx in range(num_fragments) ])
    return init_wires, exit_wires

# return the order of output wires in a united distribution
def _united_wire_order(wire_path_map, frag_wires):
    _, exit_wires = _identify_init_exit_wires(wire_path_map, len(frag_wires))
    return [ ( frag_idx, wire )
             for frag_idx, wires in enumerate(frag_wires)
             for wire in wires if wire not in exit_wires[frag_idx] ]

# return a dictionary mapping fragment output wires to the output wires of a circuit
def _frag_output_wire_map(wire_path_map):
    return { path[-1] : wire for wire, path in wire_path_map.items() }

# determine the permutation of tensor factors taking an old wire order to a new wire order
# old/new_wire_order are lists of wires in an old/desired order
# wire_map is a dictionary identifying wires in old_wire_order with those in new_wire_order
def _axis_permutation(old_wire_order, new_wire_order, wire_map = None):
    if wire_map is None: wire_map = { wire : wire for wire in old_wire_order }
    output_wire_order = [ wire_map[wire] for wire in old_wire_order ]
    wire_permutation = [ output_wire_order.index(wire) for wire in new_wire_order ]
    return [ len(new_wire_order) - 1 - idx for idx in wire_permutation ][::-1]

# get lists of wires in the circuit and its fragments
def _get_all_wires(wire_path_map):
    circuit_wires = list(wire_path_map.keys())
    all_frag_wires = set( wire for path in wire_path_map.values() for wire in path )
    frag_num = max( frag_wire[0] for frag_wire in all_frag_wires ) + 1
    frag_wires = [ sorted([ frag_wire[1]
                            for frag_wire in all_frag_wires
                            if frag_wire[0] == idx ], key = lambda qq : qq.index)
                   for idx in range(frag_num) ]
    return circuit_wires, frag_wires

# get the permutation to apply to the tensor factors of a united distribution
def united_axis_permutation(wire_path_map):
    circuit_wires, frag_wires = _get_all_wires(wire_path_map)
    output_wires = _united_wire_order(wire_path_map, frag_wires)
    output_wire_map = _frag_output_wire_map(wire_path_map)
    return _axis_permutation(output_wires, circuit_wires, output_wire_map)

##########################################################################################
# circuit construction methods
##########################################################################################

def add_qubit_markers(circuit):
    for qq in range(len(circuit.qubits)):
        circuit.rz(qq, circuit.qubits[qq])

def random_unitary(qubits):
    return qiskit.quantum_info.random.random_unitary(2**qubits)

# construt a circuit that prepares a multi-qubit GHZ state
def ghz_circuit(qubits):
    qreg = qiskit.QuantumRegister(qubits, "q") # quantum register
    circuit = qiskit.QuantumCircuit(qreg) # initialize a trivial circuit

    # add trivial operations that help us read circuits
    add_qubit_markers(circuit)
    circuit.barrier()

    # the GHZ circuit itself
    circuit.h(circuit.qubits[0])
    for qq in range(len(circuit.qubits)-1):
        circuit.cx(circuit.qubits[qq], circuit.qubits[qq+1])

    # add trivial operations that help us read circuits
    circuit.barrier()
    add_qubit_markers(circuit)

    return circuit

# construct a cascade circuit of random local 2-qubit gates
def random_cascade_circuit(qubits, layers, seed = None):
    if seed is not None: numpy.random.seed(seed)
    qreg = qiskit.QuantumRegister(qubits, "q")
    circuit = qiskit.QuantumCircuit(qreg)

    # add trivial operations that help us read circuits
    add_qubit_markers(circuit)
    circuit.barrier()

    # the random unitary circuit itself
    for layer in range(layers):
        for qq in range(len(circuit.qubits)-1):
            circuit.append(random_unitary(2), [ qreg[qq], qreg[qq+1] ])

    # add trivial operations that help us read circuits
    circuit.barrier()
    add_qubit_markers(circuit)

    return circuit

# construct a dense circuit of random local 2-qubit gates
def random_dense_circuit(qubits, layers, seed = None):
    if seed is not None: numpy.random.seed(seed)
    qreg = qiskit.QuantumRegister(qubits, "q")
    circuit = qiskit.QuantumCircuit(qreg)

    # add trivial operations that help us read circuits
    add_qubit_markers(circuit)
    circuit.barrier()

    # the random unitary circuit itself
    for layer in range(layers):
        for odd_links in range(2):
            for qq in range(odd_links, qubits-1, 2):
                circuit.append(random_unitary(2), [ qreg[qq], qreg[qq+1] ])

    # add trivial operations that help us read circuits
    circuit.barrier()
    add_qubit_markers(circuit)

    return circuit

# construct a dense circuit of random local 2-qubit gates
def random_clustered_circuit(qubits, layers, cluster_connectors, seed = None):
    if seed is not None: numpy.random.seed(seed)
    qreg = qiskit.QuantumRegister(qubits, "q")
    circuit = qiskit.QuantumCircuit(qreg)

    # add trivial operations that help us read circuits
    add_qubit_markers(circuit)
    circuit.barrier()

    clusters = len(cluster_connectors)+1
    boundaries = [ -1 ] + cluster_connectors + [ qubits ]

    def intra_cluster_gates():
        for cc in range(clusters):
            cluster_qubits = qreg[ boundaries[cc]+1 : boundaries[cc+1]+1 ]
            circuit.append(random_unitary(len(cluster_qubits)), cluster_qubits)

    def inter_cluster_gates():
        for cc in range(clusters-1):
            connecting_qubits = qreg[ boundaries[cc+1] : boundaries[cc+1]+2 ]
            circuit.append(random_unitary(2), connecting_qubits)

    # the random unitary circuit itself
    intra_cluster_gates()
    for _ in range(layers):
        inter_cluster_gates()
        intra_cluster_gates()

    # add trivial operations that help us read circuits
    circuit.barrier()
    add_qubit_markers(circuit)

    return circuit

# build a named circuit with cuts
def build_circuit_with_cuts(circuit_type, layers, qubits, fragments, seed = 0):

    cut_qubits = [ qubits*ff//fragments-1 for ff in range(1,fragments) ]

    if circuit_type == "GHZ":
        circuit = ghz_circuit(qubits)
        cuts = [ (circuit.qubits[idx * qubits//fragments], 2)
                 for idx in range(1,fragments) ]

    elif circuit_type == "cascade":
        circuit = random_cascade_circuit(qubits, layers, seed)
        cuts = [ (circuit.qubits[qubit], 1 + loc)
                 for qubit in cut_qubits
                 for loc in range(1,2*layers) ]

    elif circuit_type == "dense":
        circuit = random_dense_circuit(qubits, layers, seed)
        cuts = [ (circuit.qubits[qubit], 1 + loc)
                 for qubit in cut_qubits
                 for loc in range(1,2*layers) ]

    elif circuit_type == "clustered":
        circuit = random_clustered_circuit(qubits, layers, cut_qubits, seed)
        cuts = [ (circuit.qubits[qubit], 2 + 2*layer + side)
                 for qubit in cut_qubits
                 for layer in range(layers)
                 for side in [ 0, 1 ] ]

    else:
        raise TypeError("circuit type not recognized: {circuit_type}")

    return circuit, cuts

def fragment_cuts(frag_num, wire_path_map):
    fragment_cuts = [ { "prep" : 0, "meas" : 0 } for _ in range(frag_num) ]
    for cut_meas, cut_prep in identify_stitches(wire_path_map).items():
        fragment_cuts[cut_meas[0]]["meas"] += 1
        fragment_cuts[cut_prep[0]]["prep"] += 1
    return fragment_cuts
