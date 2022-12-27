# Circuit cutting with maximum likelihood fragment tomography

This repository contains the codes used for numerical experiments in [this work](https://www.nature.com/articles/s41534-021-00390-6) (also on the [arXiv](https://arxiv.org/abs/2005.12702)) on circuit cutting.

In a nutshell, circuit cutting is a compiler-level quantum computing a technique for reducing the size of quantum circuits and mitigating the buildup of quantum errors.  The benefits of circuit cutting come at the cost of a classical computing overhead that is exponential in the number of "cuts" that are made to a circuit.  This technique is therefore best suited for circuits with a "clustered" structure that allows them to be split into subcircuits using a small number of "cuts".

The basic idea behind circuit cutting is to
1. "cut" a quantum circuit into subcircuits, called *fragments*,
2. run the fragments (and minor variants thereof) on quantum hardware, and then
3. recombine fragment data via classical post-processing to reconstruct the output of the original circuit.  In the present work, "output" means "probability distribution over the outcomes of measurement in the computational basis".

Our main contributions to circuit cutting are summarized in the abstract of our paper:

> We introduce maximum likelihood fragment tomography (MLFT) as an improved circuit cutting technique for running "clustered" quantum circuits on quantum devices with a limited number of qubits.  In addition to minimizing the classical computing overhead of circuit cutting methods, MLFT finds the most likely probability distribution for the output of a quantum circuit, given the measurement data obtained from the circuit's fragments.  We demonstrate the benefits of MLFT for accurately estimating the output of a fragmented quantum circuit with numerical experiments on random unitary circuits.  Finally, we show that circuit cutting can estimate the output of a clustered circuit with higher fidelity than full circuit execution, thereby motivating the use of circuit cutting as a standard tool for running clustered circuits on quantum hardware.

### Contents

All codes in this repository are written in Python 3, version `>=3.8.13`.  Python package requirements are specified in `requirements.txt`.

The contents of this repository are as follows:

* `circuit_ansatz.py`: this file contains a method to construct the "random clustered circuit" used in [our paper](https://www.nature.com/articles/s41534-021-00390-6), and to identify the locations at which this circuit should be cut.
* `cutting_methods.py`: this file contains the primary methods used for (a) cutting a circuit into fragments, (b) performing maximum-likelihood fragment tomography, and (c) recombining fragment models to reconstruct the probability distribution over measurement outcomes for the full, uncut circuit.
* `compute_fidelities.py`: this file can be considered a "demo" script for the methods in `circuit_ansatz.py` and `cutting_methods.py`.  By default, running this file will build a clustered random unitary circuit, and compute the fidelity of estimating this circuit's output using
  1. full circuit execution,
  2. the [original](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.150504) circuit cutting method (also on the [arXiv](https://arxiv.org/abs/1904.00102)), and
  3. the method from [our paper](https://www.nature.com/articles/s41534-021-00390-6), maximum likelihood fragment tomography.
* `collect_data.py`: this script simulates circuits with varying qubit, fragment, and shot (or repetition) numbers, and computes the fidelity with which these circuits' outputs can be estimated using the same methods as in `compute_fidelities.py`.  This script collects all simulation data for [our paper](https://www.nature.com/articles/s41534-021-00390-6).
* `plot_data.py`: this script plots the data collected by `collect_data.py` to make the simulation figures in [our paper](https://www.nature.com/articles/s41534-021-00390-6).
