# Qubit Routing with Reinforcement Learning (RL)

This repository contains code for a qubit routing procedure that makes use of RL. Originally developed by Matteo G. Pozzi as part of a Master's Thesis for the University of Cambridge Computer Laboratory. A paper on the subject is currently in the process of being published.

## Intro to Qubit Routing

Before quantum circuits can be executed on quantum architectures, they must be be modified to satisfy the contrains of the target topology. Specifically, a quantum architecture has of a connectivity graph, consisting of physical qubits ("nodes") and links between them. (Logical) qubits inhabit the nodes, and two-qubit gates may only occur between qubits on adjacent nodes in the topology. SWAP gates must be inserted to move qubits and satisfy such constraints - this process is known as "routing".


