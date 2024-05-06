from sympy_enclosure import get_function
from IPython.display import display, Latex, Markdown
from pandas import Series, DataFrame
from typing import Union

class HelperNew:
    def __init__(self, function: str,vars_in: list[str] = [], const_in: list[str] = [], error_mark: str = r"\Delta"):
        for var in vars_in + const_in:
            if var not in function:
                raise ValueError(f"{var} not in function")
        self._vars = vars_in
        self._consts = const_in
        self._function, self._function_latex, self._error_function, self._error_function_latex = get_function(function, vars_in, const_in, error_mark)

    def get_inputs(self) -> list[str]:
        return list(self._function.__code__.co_varnames)
    def get_error_inputs(self, include_regular_inputs: bool = True, clean: bool = False) -> list[str]:
        errs_only = [e for e in self._error_function.__code__.co_varnames if e not in self.get_inputs()] 
        if clean:
            errs_only = [a.split("_err")[0] for a in self.get_error_inputs(include_regular_inputs=False)]
        if include_regular_inputs:
            return self.get_inputs() + errs_only
        else:
            return errs_only
    def _find_missing_indices(self, d: Series | DataFrame) -> list[str]:
        indices = list(d.index) if isinstance(d, Series) else list(d.columns)
        return [a for a in self.get_inputs() if a not in indices]
    def _find_missing_error_indices(self, d: Series | DataFrame) -> tuple[list[str], str]:
        indices = list(d.index) if isinstance(d, Series) else list(d.columns)
        errors = self.get_error_inputs(include_regular_inputs=False, clean=False)
        possible_error_symbols = ["_err", "err", "Δ"]
        error_symbol = ""
        for symbol in possible_error_symbols:
            if any(symbol in e for e in indices):
                error_symbol = symbol
                break
        if error_symbol == "":
            raise ValueError(f"No accepted error symbols ({possible_error_symbols}) in Series/DataFrame")
        errors = [e + error_symbol if error_symbol != "Δ" else error_symbol + e for e in errors]
        return [e for e in self.get_inputs() + errors if e not in indices], error_symbol
        

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Series) or isinstance(args[0], DataFrame):
            data = args[0]
            missing, error_symbol = self._find_missing_indices(data)
            if missing:
                raise ValueError(f"Required inputs {missing} not in Series/DataFrame")
            inputs = data[self.get_inputs()].values
        elif isinstance(args, tuple):
            if len(args) != len(self.get_inputs()):
                raise ValueError(f"Number of inputs ({len(args)}) does not match required number of inputs ({len(self.get_inputs())}, {self.get_inputs()})")
            inputs = args
        return self._function(*inputs)

    def error(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Series) or isinstance(args[0], DataFrame):
            data = args[0]
            missing,  = self._find_missing_error_indices(data)
            if missing:
                raise ValueError(f"Required inputs {missing} not in Series/DataFrame")
            inputs = data[self.get_inputs()].values
        elif isinstance(args, tuple):
            if len(args) != len(self.get_inputs()):
                raise ValueError(f"Number of inputs ({len(args)}) does not match required number of inputs ({len(self.get_inputs())}, {self.get_inputs()})")
            inputs = args


    def __repr__(self):
        print("Variables:")
        display(Markdown(", ".join([f"${var}$" for var in self._vars])))
        print("Constants:")
        display(Markdown(", ".join([f"${var}$" for var in self._consts])))
        print("Function:")
        display(Latex(f"${self._function_latex}$"))
        print("Error Function:")
        display(Latex(f"${self._error_function_latex}$"))
        return ""

a = HelperNew("a**2 + b", vars_in=["a", "b"])
df = DataFrame()
df["a"] = [1, 2]
df["b"] = [1, 1]
df["aerr"] = [1, 1]
df["berr"] = [1, 1]
print(a.error(df))
