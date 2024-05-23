Code for implementing Synaptic-RA model, demonstrating how the model evolves to form an attractor manifold (equilibrium point, travelling wave solution) under two types of inputs (endogenous and exogenous inputs).

Environmental requirements: Python 3.8.10

Install dependencies: numpy, scipy, matplotlib, seaborn

Defined methods: relu1, topology_dis, piece, gaussian_wave, dx1, dx2 et al.

Three folders: Synaptic-RA and Synaptic-RA3 are for information integration; Synaptic-RA2 is for information tracking.
Both Synaptic-RA2 and Synaptic-RA3 use 16 neurons, and Synaptic-RA use 128 neurons. The overall connection of the model is consistent, only the parameters change. 
