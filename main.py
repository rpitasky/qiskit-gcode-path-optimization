import numpy as np

from qiskit import Aer
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Tsp
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import QuantumInstance
from qiskit_optimization.converters import QuadraticProgramToQubo

from qiskit.providers.ibmq import IBMQ, least_busy

from qiskit.algorithms import VQE

CLASSICAL_METHODS = False
SIMULATING = True
SEED = 10598
REPS = 5
IBMQ_TOKEN = "<put your token here>"

# Only available on x86_64 Linux systems with qiskit-aer-gpu installed.
USE_GPU = False


def get_eigensolver(num_qubits, backend=None):
    """
    Gets the eigensolver with the given number of qubits
    """

    if CLASSICAL_METHODS:
        return NumPyMinimumEigensolver()
    else:
        quantum_instance = QuantumInstance(backend, seed_simulator=SEED, seed_transpiler=SEED)

        # solve ising problem using vqe
        spsa = SPSA(maxiter=300)
        ansatz = TwoLocal(num_qubits, "ry", "cz", reps=REPS, entanglement="linear")
        vqe = VQE(ansatz, optimizer=spsa, quantum_instance=quantum_instance)

        return vqe


def get_backend():
    if SIMULATING:
        b = Aer.get_backend("qasm_simulator")
        if USE_GPU:
            b.set_options(device='GPU')

        return b
    else:
        IBMQ.save_account(IBMQ_TOKEN)
        return least_busy(
            IBMQ.load_account().backends(
                filters=lambda x: not x.configuration().simulator
            )
        )


# Below are things that are just modified from svg_to_gcode to be more efficient for our problem

NAMESPACES = {'svg': 'http://www.w3.org/2000/svg'}
SVG = """<svg xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg" version="1.1" fill="none" stroke="none" stroke-linecap="square" stroke-miterlimit="10" width="960" height="720"><g xmlns="http://www.w3.org/2000/svg" clip-path="url(#p.0)"><path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m193.979 280.34384l0 0c0 -82.28609 66.706055 -148.99214 148.99213 -148.99214l0 0c39.515167 0 77.411896 15.697342 105.35333 43.638794c27.941467 27.941437 43.638794 65.838165 43.638794 105.35335l0 0c0 82.28607 -66.706055 148.99213 -148.99213 148.99213l0 0c-82.28607 0 -148.99213 -66.706055 -148.99213 -148.99213z" fill-rule="evenodd"/><path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m193.979 280.34384l0 0c0 -21.491272 66.706055 -38.91339 148.99213 -38.91339l0 0c82.28607 0 148.99213 17.42212 148.99213 38.91339l0 0c0 21.491272 -66.706055 38.91339 -148.99213 38.91339l0 0c-82.28607 0 -148.99213 -17.42212 -148.99213 -38.91339z" fill-rule="evenodd"/><path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m230.46457 375.81104l0 0c0 -4.1933594 11.8845825 -8.214966 33.039276 -11.180084c21.154694 -2.965149 49.84662 -4.630951 79.763885 -4.630951l0 0c62.29944 0 112.80313 7.078827 112.80313 15.811035l0 0c0 8.732178 -50.503693 15.811005 -112.80313 15.811005l0 0c-62.29947 0 -112.80316 -7.078827 -112.80316 -15.811005z" fill-rule="evenodd"/><path stroke="#000000" stroke-width="1.0" stroke-linejoin="round" stroke-linecap="butt" d="m230.46457 184.87665l0 0c0 -4.193344 11.8845825 -8.214951 33.039276 -11.180084c21.154694 -2.965149 49.84662 -4.630951 79.763885 -4.630951l0 0c62.29944 0 112.80313 7.078842 112.80313 15.811035l0 0c0 8.732178 -50.503693 15.81102 -112.80313 15.81102l0 0c-62.29947 0 -112.80316 -7.078842 -112.80316 -15.81102z" fill-rule="evenodd"/></g></svg>"""

from xml.etree import ElementTree
from typing import List

from svg_to_gcode.svg_parser import Path, Transformation
from svg_to_gcode.geometry import Curve, Chain, LineSegmentChain
from svg_to_gcode import TOLERANCES


# from svg_to_gcode.svg_parser import parse_string
def parse_root(root: ElementTree.Element, canvas_height=None) -> List[Curve]:
    if canvas_height is None:
        canvas_height = float(root.get("height"))

    curves = []

    # Draw elements (Depth-first search)
    for element in list(root):
        transformation = Transformation()

        transform = element.get('transform')
        if transform:
            transformation.add_transform(transform)
        # If the current element is opaque and visible, draw it
        if element.tag == "{%s}path" % NAMESPACES["svg"]:
            line_chain = LineSegmentChain()

            path = Path(element.attrib['d'], canvas_height, True, transformation)

            for curve in path.curves:
                approx = LineSegmentChain.line_segment_approximation(curve)
                if approx.length() == 0:
                    pass
                try:
                    for c in approx:
                        if c.length() > TOLERANCES['approximation']:
                            line_chain.append(c)
                except:
                    curves.extend([line_chain])
                    line_chain = LineSegmentChain()
                    for c in approx:
                        if c.length() > TOLERANCES['approximation']:
                            line_chain.append(c)
            curves.extend([line_chain])
        # Continue the recursion
        curves.extend(parse_root(element, canvas_height=canvas_height))
    return curves


# Extend the Compiler class to use our quantum algorithm
import typing

import networkx as nx

from svg_to_gcode.compiler import Compiler, interfaces
from svg_to_gcode.geometry import Curve, Line, LineSegmentChain


# One hell of a name, really.
class QuantumLaserOptimizer(Compiler):
    def append_curves(self, curves: [typing.Type[Curve]]):
        chains = []
        for line_chain in curves:
            if line_chain.length() != 0:
                chains.append([[line_chain.point(0), line_chain.point(1)], line_chain])

        G = nx.Graph()
        G.add_nodes_from([0 for i in range(len(chains))])

        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                [a, b] = chains[i][0]

                distance = abs(a - b)
                G.add_edge(i, j, weight=distance)

        tsp = Tsp(G)

        qp = tsp.to_quadratic_program()
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        qubitOp, offset = qubo.to_ising()

        es = get_eigensolver(qubitOp.num_qubits, backend=get_backend())

        result = es.compute_minimum_eigenvalue(qubitOp)

        solution = np.hstack(tsp.interpret(tsp.sample_most_likely(result.eigenstate)))

        for index in solution:
            self.append_line_chain(chains[int(index)][1])


def main():
    root = ElementTree.fromstring(SVG)
    curves = parse_root(root)

    print("Parsed svg with " + str(len(curves)) + " curves.")

    compiler = QuantumLaserOptimizer(interfaces.Gcode, movement_speed=1000, cutting_speed=300, pass_depth=5)
    compiler.append_curves(curves)

    compiler.compile_to_file("out.gcode", passes=2)


if __name__ == "__main__":
    main()
