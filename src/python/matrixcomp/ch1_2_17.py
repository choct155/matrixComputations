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

    @staticmethod
    def solve_for_current(resistor_exprs: List[Add]) -> List[Add]:
        """takes the input circuit node and creates a list of expressions for each current into that node"""
        out: List[Add] = list(map(lambda le: solve(le, I)[0], resistor_exprs))
        return out

    @staticmethod
    def combine_resistor_exprs(resistor_exprs: List[Add]) -> Add:
        """takes the list of current expressions for a node and combines like variables"""
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
    # x_all: Circuit = Circuit([x1, x2, x3, x4, x5, x6, x7])

class Circuit:

    def __init__(self, circuit_expr: List[CircuitNode], x_matrix: tf.Tensor) -> None:
        self.circuit_expr: List[CircuitNode] = circuit_expr
        empty_dict: Dict[Symbol, float] = Circuit.create_empty_circuit_dict(self.circuit_expr)
        full_array: List[List[float]] = Circuit.create_circuit_array(self.circuit_expr, empty_dict)
        a_array: List[List[float]] = Circuit.create_a_array(full_array)
        b_list: List[float] = Circuit.create_b_array(full_array)
        a_matrix: tf.Tensor = Circuit.convert_to_tensor(a_array)
        b_matrix: tf.tensor = Circuit.convert_to_tensor(b_list)
        self.x_matrix: tf.Tensor = solve_for_x(a_matrix, b_matrix)

    @staticmethod
    def create_empty_circuit_dict(input_circuit: List[CircuitNode]) -> Dict[Symbol, float]:
        """creates an empty dict which accounts for all variables"""
        get_symbols: Callable[[CircuitNode], Set[Symbol]] = lambda cn: set(cn.node_expr.as_coefficients_dict().keys())
        all_symbols: List[Set[Symbol]] = map(get_symbols, input_circuit)
        symbol_set: Set[Symbol] = reduce(lambda s1, s2: s1.union(s2), all_symbols)
        #initialize dict to return {a: 0, b: 0, c: 0, d: 0, e: 0, f: 0, g: 0, 1: 0}
        empty_dict: Dict[Symbol, float] = dict.fromkeys(symbol_set, 0.0)
        return empty_dict

    @staticmethod
    def create_circuit_array(input_circuit: List[CircuitNode], empty_dict: Dict[Symbol, float]) -> List[Dict[Symbol, float]]:
        """maps over the input list of CircuitNodes to create list of row dicts"""
        #get_row_dict: Callable[[CircuitNode], Dict[Symbol, float]] = lambda cn: empty_dict.update(cn.node_expr.as_coefficients_dict())
        #output_array_dicts: List[Dict[Symbol, float]] = map(get_row_dict, input_circuit)
        partial_row_dict: Callable[CircuitNode] = lambda cn: cn.node_expr.as_coefficients_dict()
        full_row_dict: Callable[Dict[Symbol, float]] = lambda d1: d1.update(empty_dict) or d1
        partial_array_dicts: List[Dict[Symbol, float]] = map(partial_row_dict, input_circuit)
        full_array_dicts: List[Dict[Symbol, float]] = map(full_row_dict, partial_array_dicts)
        #output_array: List[List[float]] = map(lambda x: x.values(), full_array_dicts)
        return full_array_dicts

    @staticmethod
    def create_b_array(input_array: List[Dict[Symbol, float]]) -> List[float]:
        b_array: List[float] = [key[1] for key in input_array]
        return b_array

    @staticmethod
    def create_a_array(input_array: List[Dict[Symbol, float]]) -> List[List[float]]:
        a_dict_list: List[Dict[Symbol, float]] = map(lambda x: x.pop(1), input_array)
        a_array: List[List[float]] = map(lambda a: a.values(), a_dict_list)
        return a_array

    # @staticmethod
    # def create_a_b_arrays(input_array: List[List[float]]) -> (List[List[float]], List[float]):
    #     """separates a and b arrays"""
    #     b: List[float] = [r.pop(7) for r in input_array]
    #     return (input_array, b)

    @staticmethod
    def convert_to_tensor(input) -> tf.Tensor:
        """converts created array to tensor"""
        return tf.convert_to_tensor(input)