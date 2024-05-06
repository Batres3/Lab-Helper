# All sympy stuff is hidden here for faster code parsing
from typing import Callable, Union
from sympy import Symbol, parse_expr, lambdify, diff, Mul, Add, Pow, sqrt, latex, init_printing

init_printing()

def list_swap(l: list, x, y):
    if x not in l or y not in l:
        ValueError("elements not in given list!")
    el = [l.index(x), l.index(y)]
    l[el[0]], l[el[1]] = l[el[1]], l[el[0]]
text = ["ab", "a", "b", "bc", "c"]

def sort_string_list(l: list[str]) -> list[str]: 
    out = l
    for x in out:
        for y in out:
            if x in y and out.index(x) < out.index(y):
                list_swap(text, x, y)
    return out

def get_function(function: str, vars: list[str], consts: list[str] = [], error_symbol: str = r"\Delta") -> Union[Callable, str, Callable, str]:
    func = function
    all_text = sort_string_list(vars + consts)
    hashes = [str(hash(var)) for var in all_text]
    symbols = [Symbol(var) for var in all_text]
    symbols_vars = [Symbol(var) for var in vars] # Only variables
    for var, h in zip(all_text, hashes):
         func = func.replace(var, h)
    for var, h in zip(all_text, hashes):
         func = func.replace(h, f"Symbol('{var}')")
    func = parse_expr(func)
    out_func = lambdify(symbols,func, "numpy")

    errors = [Symbol(err) for err in [error_symbol + " " + var for var in vars]]
    lambdify_errs = [Symbol(err) for err in [a + "_err" for a in vars]] # Necessary because variables with "\Delta" are not interpreted properly
    err_func_latex = 0
    err_func_lambdify = 0
    for i in range(len(errors)): 
        err_func_latex = Add(err_func_latex, Pow(Mul(errors[i], diff(func, symbols_vars[i]), evaluate= False), 2, evaluate=False))
        err_func_lambdify = Add(err_func_lambdify, Pow(Mul(lambdify_errs[i], diff(func, symbols_vars[i]), evaluate= False), 2, evaluate=False))
    err_func_latex = sqrt(err_func_latex)
    out_err_func = lambdify(symbols + lambdify_errs, err_func_lambdify, "numpy")
    return out_func, latex(func), out_err_func, latex(err_func_latex)
