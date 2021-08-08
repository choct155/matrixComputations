import numpy as np
import tensorflow as tf
from sympy import solve
from sympy.abc import a, b, c, d, e, f, g, I
from sympy.core.add import Add
from sympy.core.symbol import Symbol
from typing import List, Dict
from dataclasses import dataclass, field
from functools import reduce
from IPython.display import display, Image
from matrixcomp.ch1_2 import solve_for_x

@dataclass
class CircuitNode:
    label: str
    resistor_exprs: List[Add]
    current_symbol: Symbol
    node_expr: Add = field(default_factory=Add)

    def __post_init__(self) -> None:
        has_c_symbol: Dict[int, bool] = {
            idx: le.has(self.current_symbol) for idx, le in enumerate(self.resistor_exprs)
        }
        if not all(has_c_symbol.values()):
            missing_symbol: Dict[int, bool] = {
                idx: test_val for idx, test_val in has_c_symbol.items() if not test_val
            }
            msg: str = f"""
            All resistor expressions must contain the current symbol!
            The following do not:
            {missing_symbol}
            """
            raise ValueError(msg)

        current_expressions: List[Add] = CircuitNode.solve_for_current(self.resistor_exprs)
        self.node_expr: Add = CircuitNode.combine_resistor_exprs(current_expressions)


    """takes the input circuit node and creates a list of expressions for each current into that node"""
    @staticmethod
    def solve_for_current(resistor_exprs: List[Add]) -> List[Add]:
        out: List[Add] = list(map(lambda le: solve(le, I)[0], resistor_exprs))
        return out

    """takes the list of current expressions for a node and combines like variables"""
    @staticmethod
    def combine_resistor_exprs(resistor_exprs: List[Add]) -> Add:
        out: Add = reduce(lambda first, second: first + second, resistor_exprs)
        return out

    """user input: initial expressions for each node"""
    # x1: CircuitNode = CircuitNode("x1", [0.5*(a - b) - I, 0.1*(a - d) - I], I)
    # x2: CircuitNode = CircuitNode("x2", [0.5*(b - a) - I, 0.2*(b - c) - I, .5*(b - e) - I], I)
    # x3: CircuitNode = CircuitNode("x3", [0.2*(c - b) - I, 0.1*(c - f) - I], I)
    # x4: CircuitNode = CircuitNode("x4", [0.1*(d - a) - I, 0.1*(d - e) - I, .2*(d - g) - I], I)
    # x5: CircuitNode = CircuitNode("x5", [0.5*(e - b) - I, 0.1*(e - d) - I, .5*(e - f) - I, .1*(e - 9) - I], I)
    # x6: CircuitNode = CircuitNode("x6", [0.5*(f - e) - I, 0.1*(f - c) - I, .2(f - 0) - I], I)
    # x7: CircuitNode = CircuitNode("x7", [0.2*(g - d) - I, 0.2*(g - 9) - I], I)

    """user input: initial list of nodes"""
    # x_all: Circuit = [x1, x2, x3, x4, x5, x6, x7]

@dataclass
class Circuit:
    #input for initial circuit is a list of CircuitNodes
    circuit_expr: List[CircuitNode]
    x_matrix: tf.Tensor

    def __post_init__(self) -> None:
       empty_dict: Dict[Add, float] = Circuit.create_empty_circuit_dict(self.circuit_expr)
       a_array: List[List[float]] = Circuit.create_circuit_array(self.circuit_expr, empty_dict)
       b_list: List[float] = Circuit.create_b_array(self.circuit_expr)
       a_matrix: tf.Tensor = Circuit.convert_to_tensor(a_array)
       b_matrix: tf.tensor = Circuit.convert_to_tensor(b_list)
       self.x_matrix: tf.tensor = solve_for_x(a_matrix, b_matrix)

    """creates an empty dict which accounts for all variables"""
    @staticmethod
    def create_empty_circuit_dict(input_circuit: "Circuit") -> Dict[Add, float]:
        #determine number of keys needed
        number_of_keys: int = len(input_circuit)
        #create alphabetical list of keys
        key_list: List = []
        alpha = 'a'
        for i in range(0, number_of_keys):
            key_list.append(alpha)
            alpha = chr(ord(alpha) + 1)
        #initialize dict to return {a: 0, b: 0, c: 0, d: 0, e: 0, f: 0, g: 0}
        empty_dict: Dict[Add, float] = dict.fromkeys(key_list, 0.0)
        return empty_dict

    """takes the expression for a node and creates a list with each node's variables, with 0 as the value for any
    unaccounted-for keys in the row"""
    @staticmethod
    def create_circuit_row(input_circuit_node: CircuitNode, empty_dict: Dict[Add, float]) -> Dict[Add, float]:
        row_dict: Dict[Add, float] = input_circuit_node.node_expr.as_coefficients_dict()
        updated_row: Dict[Add, float] = empty_dict.update(row_dict)
        return updated_row

    """maps over the input list of CircuitNodes to create an array of just the values in the row dicts"""
    @staticmethod
    def create_circuit_array(input_circuit: "Circuit", empty_dict: Dict[Add, float]) -> List[List[float]]:
        output_array: List[List[float]] = map(input_circuit, lambda x: Circuit.create_circuit_row(x, empty_dict).values)
        return output_array

    """finds the value in a node expression without a coefficient"""
    @staticmethod
    def find_non_coefficients(input_node: CircuitNode) -> float:
        output_number: float = 0.0
        # want 0 returned if there is no value in the node, but also not sure if this atoms(Number) works the way it says it does
        # https://www.geeksforgeeks.org/python-sympy-atoms-method/
        output_number: float = input_node.node_expr.atoms(Number)
        return output_number

    "creates the b_array with the non-coefficient values"
    @staticmethod
    def create_b_array(input_circuit: "Circuit") -> List[float]:
        out: List[float] = map(input_circuit, lambda x: Circuit.find_non_coefficients(x))
        return out

    """converts created array to tensor"""
    @staticmethod
    def convert_to_tensor(input) -> tf.Tensor:
        return tf.convert_to_tensor(input)