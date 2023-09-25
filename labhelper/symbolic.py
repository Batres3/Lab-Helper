import sympy as sp
from IPython.display import display

sp.init_printing()

class Helper:
    """
    Helper class made to simplify the process of making a function programatically and to calculate the error function associated to the given function.
    
    In order to input LaTex symbols (such as greek letters, letters with subscripts or superscripts, such as t_0, or any other symbol which is valid in LaTex)
    the input must be written as it would be in LaTex: "ɑ_0" would be "\\alpha_{0}" (in the case of green letters the backslash "\\" can be ommited -> "\\alpha_{0}" = "alpha_{0}")
    
    MAKE SURE THAT THE PROVIDED VARIABLES IN vars_in, const_in MATCH EXACTLY WITH THE VARIABLES AND CONSTANTS IN THE FUNCTION
    """
    # For Readers looking at the documentation from source code keep in mind \\ is equivalent to \ because of the manner in which python parses docstrings
    def __init__(self, func_str: str, vars_in: list[str] , const_in: list[str] | None = None, error_mark: str = r"\Delta ") -> None:
        self.vars_text = vars_in
        self.vars = [sp.Symbol(e) for e in vars_in]
        if const_in != None:
            self.consts_text = const_in
            self.consts = [sp.Symbol(e) for e in const_in]
        else:
            self.consts = []
        for var in self.vars_text + self.consts_text:
            func_str = func_str.replace(var, "Symbol('" + var + "')")

        self.function = sp.parse_expr(func_str)
        
        if not all(item in self.function.atoms(sp.Symbol) for item in self.vars + self.consts):
            raise ValueError("The given variables and/or constants are not the same as in the given function")

        self.error_mark = error_mark
        self.errors_text = [self.error_mark + a for a in vars_in]
        self.errors = [sp.Symbol(a) for a in self.errors_text]
        self.error_function = None
    
    def function_input_check(self):
        """
        Shows all inputs to the function
        """
        display(self.function.atoms(sp.Symbol))

    def display_data(self):
        """
        Prints the function, constants and variables
        """
        print("Variables:")
        display(self.vars)
        print("Constants:")
        display(self.consts)
        print("Function:")
        display(self.function)
        if self.error_function != None:
            print("Error Function:")
            display(self.error_function)
    
    def calculate_error_function(self):
        """
        Calculates the error function of the function provided during class creation following the formula:

        Delta f(x1,...,xn) = sqrt((Delta x1 * df/dx1)^2 + ... + (Delta xn * df/dxn)^2)
        """
        self.error_function = 0
        for i in range(len(self.vars)): 
            a = sp.Mul(self.errors[i], sp.diff(self.function, self.vars[i]), evaluate= False)
            b = sp.Pow(a, 2, evaluate=False)
            self.error_function = sp.Add(self.error_function, b)
        self.error_function = sp.sqrt(self.error_function)
        
    def evaluate_function(self, subs: dict, as_float: bool = False):
        """
        Evaluates the function with the given values, which must be provided in the form of a Dictionary, for instance:

        let f(A,b,C,d) = (A*b)/(C*d) then to evaluate f at (0.5, 2, 3, -1) the input "subs" must be as follows:

        evaluate_function({"A": 0.5, "b": 2, "C": 3, "d": -1})

        Not all variables have to be supplied, for instance, for the above example the following input is valid:

        evaluate_function({"A": 0.5, "C": 3}) This will leave "b" and "d" as variables
        """
        if not all(item in self.consts_text + self.vars_text for item in subs.keys()):
            raise ValueError("The given variables and/or constants are not the same as in the given function")
        return float(self.function.evalf(subs=subs)) if as_float else self.function.evalf(subs=subs)
    
    def evaluate_error_function(self, subs: dict, as_float: bool = False):
        """
        Evaluates the error function with the given values, which must be provided in the form of a Dictionary, for instance:

        let Delta f(A,Delta A,B,Delta B) = sqrt((Delta A * B)^2 + (Delta B * A)^2) then to evaluate f at (0.5, 2, 3, -1) the input "subs" must be as follows:

        evaluate_function({"A": 0.5, "\\Delta A": 2, "C": 3, "\\Delta B": -1})

        Not all variables have to be supplied, for instance, for the above example the following input is valid:

        evaluate_function({"A": 0.5, "C": 3}) This will leave "\\Delta A" and "\\Delta B" as variables
        """
        if self.error_function == None:
            self.calculate_error_function()
        
        if not all(item in self.consts_text + self.vars_text + self.errors_text for item in subs.keys()):
            raise ValueError("The given variables and/or constants are not the same as in the error function")
        return float(self.error_function.evalf(subs=subs)) if as_float else self.error_function.evalf(subs=subs)
    
    def solve_error_function_for_variable(self, variable_to_solve: str, function_value = None, rest_of_variables: dict = None, symbolically = False):
        """
        Calculates the value the given variable "variable_to_solve" must have in order to make the error function equal "function_value"
        
        If "symbolically" is set to True, the only required input is "variable_to_solve"

        Useful for checking where the differences between experimental and theoretical values arise from, as it can give the value that the
        error must take in order to match the experimental value
        """
        if symbolically:
            function_value = sp.Symbol(self.error_mark + "f")
            if variable_to_solve not in self.vars_text + self.consts_text + self.errors_text:
                raise ValueError("value of variable_to_solve is not a parameter of error function")
            return sp.solve(sp.Equality(self.error_function, function_value), sp.Symbol(variable_to_solve))
        else:
            if rest_of_variables == None:
                raise ValueError("must provide rest_of_variables parameter, or set symbolically to True")
            if not all(item in self.consts_text + self.vars_text + self.errors_text for item in rest_of_variables.keys()):
                raise ValueError("The given variables and/or constants are not the same as in the error function")
            return sp.solve(sp.Equality(self.error_function.evalf(subs=rest_of_variables), function_value), sp.Symbol(variable_to_solve))
        
    def solve_error_function_for_all_variables(self, function_value = None, rest_of_variables: dict = None, symbolically: bool = False):
        """
        Solves the error function for each of the variables (x1, x2, ...) separately
        and returns either the required value for each variable in order to make the function == function_value
        (assuming the rest of the variables are as specified in rest_of_variables, ALL variables must be provided if a numerical result is desired)
        or the expression to get said result if "symbolically" is set to True
        """
        if symbolically:
            function_value = sp.Symbol(self.error_mark + "f")
            for error in self.errors:
                print("---------------")
                display(error)
                print("=\n")
                display(sp.solve(sp.Equality(self.error_function, function_value), error))
                print("---------------")
        else:
            if rest_of_variables == None or function_value == None:
                raise ValueError("must provide rest_of_variables and function_value parameters")
            if not all(item in self.consts_text + self.vars_text + self.errors_text for item in rest_of_variables.keys()):
                raise ValueError("The given variables and/or constants are not the same as in the error function")
            for var in self.vars:
                dict_cpy = rest_of_variables.copy()
                dict_cpy.pop(var.name)
                print("---------------")
                display(var)
                print("=\n")
                display(sp.solve(sp.Equality(self.error_function.evalf(subs=dict_cpy), function_value), var))
                print("---------------")
