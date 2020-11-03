# Circuit cutting with maximum likelihood fragment tomography

This repository contains the codes used for numerical experiments in [this work](https://arxiv.org/abs/2005.12702) on circuit cutting.  *Disclaimer: I (Michael A. Perlin) apologize in advance for my pedestrian presentation of the contents in this repository.  I am merely a physicist, with little training in "proper" code maintenance and documentation.*

In a nutshell, circuit cutting is compiler-level quantum computing a technique for reducing the size of quantum circuits and mitigating the buildup of quantum errors.  The benefits of circuit cutting come at the cost of a classical computing overhead that is exponential in the number of "cuts" that are made to a circuit.  This technique is therefore best suited for circuits with a "clustered" structure that allows them to be split into sub-circuits using a small number of "cuts".

The basic idea behind circuit cutting is to
1. "cut" a quantum circuit into sub-circuits, called *fragments*,
2. run the fragments (and minor variants thereof) on quantum hardware, and then
3. recombine fragment data via classical post-processing to reconstruct the output of the original circuit.

Our main contributions to circuit cutting are summarized in the abstract of our paper:

> We introduce maximum likelihood fragment tomography (MLFT) as an improved circuit cutting technique for running "clustered" quantum circuits on quantum devices with a limited number of qubits.  In addition to minimizing the classical computing overhead of circuit cutting methods, MLFT finds the most likely probability distribution for the output of a quantum circuit, given the measurement data obtained from the circuit's fragments.  Unlike previous circuit cutting methods, MLFT guarantees that all reconstructed probability distributions are strictly non-negative and normalized.  We demonstrate the benefits of MLFT for accurately estimating the output of a fragmented quantum circuit with numerical experiments on random unitary circuits.  Finally, we provide numerical evidence and theoretical arguments that circuit cutting can estimate the output of a clustered circuit with higher fidelity than full circuit execution, thereby motivating the use of circuit cutting as a standard tool for running clustered circuits on quantum hardware.

### Contents

The contents of this repository are as follows (all codes written in Python 3):

* `circuit_cutting.py`: this file contains methods to cut a quantum circuit (represented by a Qiskit `QuantumCircuit` object) into fragments.
* `mlrecon_methods.py`: this file contains the primary methods used for maximum likelihood fragment tomography, as well as a tensor-network-based method for recombining fragment models to reconstruct the "full" (pre-cut) circuit output.  Also included here are methods to construct some simple quantum circuits that are amenable to circuit cutting, such as the "clustered random unitary circuits" used in our paper.
* `mlrecon_demo.py`: this is a "demo" file for using the methods in `circuit_cutter.py` and `mlrecon_methods.py`.  By default, this file will build a clustered random unitary circuit, and compare the fidelity of estimating this circuit's output using
  1. full circuit execution
  2. the [original](https://journals.aps.org/prl/accepted/cf075YabH641a287f098406380a7b05df8764bce0) circuit cutting method (also on the [arXiv](https://arxiv.org/abs/1904.00102))
  3. our circuit cutting method (maximum likelihood fragment tomography).
* `collect_mlrecon_data.py`: this script simulates circuits with varying qubit, fragment, and shot numbers, and computes the fidelity with which these circuits' outputs can be estimated using the same methods as in the demo script (`mlrecon_demo.py`).  This is the script that was used to collect all simulation data for our paper.
* `plot_mlrecon_data.py`: this script plots the data collected by `collect_mlrecon_data.py` to make the last figure in our paper.

The primary dependencies of this repository are [Qiskit](https://qiskit.org/) v0.12.0, [TensorNetwork](https://github.com/google/TensorNetwork) v0.4.1, as well as some standard Python packages such as NumPy, SciPy, and NetworkX.  The dependency on a (very) outdated version of Qiskit (v0.12.0) is largely due to breaking changes in how Qiskit represents quantum circuits in the backend, which is central to the methods in `circuit_cutting.py`.  If there is demand, I may update these codes for compatibility with newer versions of Qiskit in the future.
