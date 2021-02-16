import random
from typing import Callable, List


def get_all_possible_expression_addresses(expr, current_address=None):
    if current_address is None:
        current_address = []

    addresses = [current_address]

    for node_idx, node_expr in enumerate(expr.args):
        current_node_address = [*current_address, node_idx]
        addresses = [*addresses, *get_all_possible_expression_addresses(node_expr, current_node_address)]

    return addresses


def get_expression_part(expr, address):
    for num in address:
        expr = expr.args[num]
    return expr


def modify_expression(expr, modifier, address):
    if not address:
        modified_part = modifier(expr)
        return modified_part if modified_part != None else expr
    if len(address) == 1:
        largs = list(expr.args)
        modified_part = modifier(largs[address[0]])
        if modified_part != None:
            largs[address[0]] = modifier(largs[address[0]])
        else:
            largs.pop(address[0])

        try:
            return expr.func(*largs)
        except:
            return expr.func(*largs, 2)
    else:
        largs = list(expr.args)
        largs[address[0]] = modify_expression(expr.args[address[0]], modifier, address[1:])
        new = expr.func(*largs)
    return new

def test_modify_expression_prod():
    from sympy import symbols
    balanced_accuracy = symbols('balanced_accuracy')
    b = symbols('b')
    d = symbols('d')
    s = symbols('s')
    modify_expression(b**(d*s)*balanced_accuracy**balanced_accuracy + d, lambda expr: None, [1,1,1])