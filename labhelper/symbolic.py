from .sympy_enclosure import get_function
from .units import Quantity, remove_units
from IPython.display import display, Latex, Markdown
from pandas import Series, DataFrame
from pandas.api.types import is_list_like
from typing import Union

def identify_error_symbol(indices: list[str] | str, possible_symbols: list[str]) -> str:
    if isinstance(indices, str):
        indices = [indices]
    error_symbol = ""
    for symbol in possible_symbols:
        if any(symbol in e for e in indices):
            error_symbol = symbol
            break
    return error_symbol

class Helper:
    _possible_error_symbols = ["_err", "err", "Δ"]
    def __init__(self, function: str,vars_in: list[str] = [], const_in: list[str] = [], error_mark: str = r"\Delta") -> None:
        for var in vars_in + const_in:
            if var not in function:
                raise ValueError(f"{var} not in function")
        self._vars = vars_in
        self._consts = const_in
        self._values = {} # TODO: mabye switch this to DataFrame?
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
        for key, value in self._values.items():
            if key not in indices:
                d[key] = value
        indices = list(d.index) if isinstance(d, Series) else list(d.columns)
        missing = [a for a in self.get_inputs() if a not in indices]
        return missing

    def _find_missing_error_indices(self, d: Series | DataFrame) -> tuple[list[str], list[str]]:
        indices = list(d.index) if isinstance(d, Series) else list(d.columns)
        # Rename possible error symbols to "_err" for consistency
        error_symbol = identify_error_symbol(indices, Helper._possible_error_symbols)
        if error_symbol != "":
            error_indices = [e for e in indices if error_symbol in e]
            valid_error_indices = [e.replace(error_symbol, "", 1) + "_err" for e in error_indices]
            rename_map = {invalid: valid for invalid, valid in zip(error_indices, valid_error_indices)}
            d.rename(columns=rename_map, inplace=True)
            indices = list(d.index) if isinstance(d, Series) else list(d.columns)
        
        for key, value in self._values.items():
            if key not in indices:
                d[key] = value
        indices = list(d.index) if isinstance(d, Series) else list(d.columns)
        return [a for a in self.get_error_inputs() if a not in indices]

    def _validate_key(self, keys) -> list[str]:
        if isinstance(keys, str):
            keys = [keys]
        invalid_types = []
        for key in keys:
            if not isinstance(key, str):
                invalid_types.append(type(key))
        if invalid_types:
            raise ValueError(f"Indexing is only allowed through strings, not {invalid_types}")
        
        if all(key in self.get_inputs() for key in keys):
            return keys
        
        symbol = identify_error_symbol(keys, Helper._possible_error_symbols)
        if symbol == "":
            raise ValueError(f"Key is not valid. Reminder: possible error symbols are {Helper._possible_error_symbols}")
        non_errors = [key for key in keys if symbol not in key]
        errors = [key.replace(symbol, "", 1) for key in keys if key not in non_errors]
        valid_errors = [key + "_err" for key in errors]
        final = non_errors + valid_errors
        for key in final:
            if key not in self.get_error_inputs():
                raise ValueError(f"Key {key} is not one of {self.get_error_inputs()}")
        return final
        
    def __getitem__(self, keys):
        keys = self._validate_key(keys)
        not_set = [key for key in keys if key not in self._values.keys()]
        if not_set:
            raise ValueError(f"Key {not_set} has not been set")
        
        return [self._values[key] for key in keys] if len(keys) > 1 else self._values[keys[0]]
    def __setitem__(self, keys, values):
        keys = self._validate_key(keys)
        if not is_list_like(values):
            values = [values]
        self._values |= {key: value for key, value in zip(keys, values)}

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Series) or isinstance(args[0], DataFrame):
            data = args[0]
            missing = self._find_missing_indices(data)
            if missing:
                raise ValueError(f"Required inputs {missing} not in Series/DataFrame")
            inputs = data[self.get_inputs()].values.T
        elif isinstance(args, tuple):
            if len(args) != len(self.get_inputs()):
                raise ValueError(f"Number of inputs ({len(args)}) does not match required number of inputs ({len(self.get_inputs())}, {self.get_inputs()})")
            inputs = args
        return self._function(*inputs)

    def error(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Series) or isinstance(args[0], DataFrame):
            data = args[0]
            missing = self._find_missing_error_indices(data)
            if missing:
                raise ValueError(f"Required inputs {missing} not in Series/DataFrame")
            inputs = data[self.get_error_inputs()].map(remove_units).values.T # Remove units for now since unit**1/2 is not supported
            final = self._error_function(*inputs) # In SI units
            units = self._function(*data[self.get_inputs()].iloc[0].values)
            if isinstance(units, Quantity):
                units, expected_units = units.units, units.expected_units
                print(final)
                final = [Quantity(value=a, units=units, expected_units=expected_units) for a in final] if is_list_like(final) else Quantity(value=final, units=units, expected_units=expected_units)
            return final
        elif isinstance(args, tuple):
            if len(args) != len(self.get_error_inputs()):
                raise ValueError(f"Number of inputs ({len(args)}) does not match required number of inputs ({len(self.get_error_inputs())}, {self.get_error_inputs()})")
            inputs = args
            return self._error_function(*inputs)


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