# qiskit-gcode-path-optimization
First place, CWRU Quantum Computing Club Hackathon, Fall 2022.

## Objective
Use a combination of quantum and classical optimization techniques to produce optimal G-Code paths for modern laser cutters.

## Methodology
1. Use an off-the-shelf library to generate G-Code from the SVG
2. Replace the *one small* part of the library that approximates a solution to TSP with something that interfaces with a quantum computer
    - Generate a complete weighted graph with NetworkX
    - Use Qiskit's algorithms module to convert the graph to a **q**uadratic **u**nconstrained **b**inary **o**ptimization problem
    - Use a variational quantum eigensolver to solve the QUBO
3. Use the result of the optimization problem to construct faster G-Code paths.
