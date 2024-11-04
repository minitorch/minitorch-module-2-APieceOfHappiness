from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    new_vals1 = [val + epsilon if i == arg else val for i, val in enumerate(vals)]
    new_vals2 = [val - epsilon if i == arg else val for i, val in enumerate(vals)]
    return (f(*new_vals1) - f(*new_vals2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    sorted_list = []
    visited_vars = set()

    def sort_variables(variable: Variable) -> None:
        if variable.unique_id in visited_vars:
            return None
        visited_vars.add(variable.unique_id)

        parents = variable.parents
        for parent in parents:
            if parent.is_constant() is False:
                sort_variables(parent)
        sorted_list.append(variable)

    sort_variables(variable)
    return sorted_list[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_vars = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}
    # print('WOOOOOW' * 10)
    for var in sorted_vars:
        if var.unique_id not in derivatives:
            derivatives[var.unique_id] = 0
        deriv = derivatives[var.unique_id]
        # print(f'---> {var}, {deriv}')
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for parent_var, grad in var.chain_rule(deriv):
                if parent_var.unique_id not in derivatives:
                    derivatives[parent_var.unique_id] = 0
                derivatives[parent_var.unique_id] += grad


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
