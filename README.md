# Qubit Routing with Reinforcement Learning (RL)

This repository contains code for a qubit routing procedure that makes use of RL. Originally developed by Matteo G. Pozzi as part of a Master's Thesis for the University of Cambridge Computer Laboratory. A paper on the subject is available here: https://arxiv.org/abs/2007.15957.

## Intro to Qubit Routing

Before quantum circuits can be executed on quantum architectures, they must be be modified to satisfy the contrains of the target topology. Specifically, a quantum architecture has a connectivity graph, consisting of physical qubits ("nodes") and links between them. (Logical) qubits inhabit the nodes, and two-qubit gates may only occur between qubits on adjacent nodes in the topology. SWAP gates must be inserted to move the qubits and satisfy such constraints - this process is known as "routing".

This project uses RL to perform the task of routing qubits. For more details, please see the above paper.


## Module structure

- The `environments` directory contains classes that represent different types of quantum architecture. These are called "environments", in the RL sense of the word - specifically, they are responsible for generating a new state from a state-action pair, and delivering a reward. The simplest is the "grid" environment, but there are some unique real-world quantum architectures as well, such as the IBM Q20 Tokyo. There is also a `PhysicalEnvironment` class that is responsible for simulating a (_routed_) quantum circuit on a given target architecture, for the purpose of verifying that the hardware constraints are indeed satisfied.
- The `benchmarks` directory contains a series of benchmarks used for the thesis and subsequent paper. In general, to obtain results for different architectures or routing methods, simply import the correct environment and `schedule_swaps` function for the architecture and method you would like to test, respectively.
- The `agents` and `annealers` directories contain the code for the actual RL method, as well as some helper functions for training the models and performing routing.
- The `other_systems` directory contains code for routing with other existing methods, such as Qiskit's `StochasticSwap`. These files simply wrap external library calls into a common format, for easy benchmarking.
- The `realistic_test_set` directory contains a series of `.qasm` files that were used in the paper to benchmark the different routing methods.
- The `utils` directory contains a series of utilities that were useful throughout the thesis, as well as some static heuristics (which were used in the initial phase of the thesis) and a simple implementation of a PER memory tree.

Please note: the "single state" agent was briefly trialled in the thesis but was not used in the paper. The "paired state" agent remains the recommended choice, and is the one evaluated in the paper.


## Python package versions

This code requires **Python 3.7**, as well as specific versions of a few libraries in order to run. I believe there have been some minor changes to those libraries in recent times, and I have not yet had the chance to update the code in response. I recommend installing the relevant packages by running `pip install -r requirements.txt`, or you could even try fixing the code yourself to be compatible with the latest versions.


## Disclaimer

This code has not yet been properly documented. It is the result of cleaning up the original project repository (which was significantly larger and more messy), so apologies in advance if some files have ended up in the wrong places. Please do get in touch if you have any issues running or understanding the code, and I will try my best to help out.
