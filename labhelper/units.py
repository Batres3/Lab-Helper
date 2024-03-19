from numbers import Number
from enum import IntEnum
from typing import Callable
from fractions import Fraction
from math import sqrt, prod, ceil, floor, trunc, log

def prime_factorization_int(n: int) -> list[tuple[int, int]]:
    if n == 1:
        return []
    elif n == 0:
        return []

    fac = []
    powers = []
    count = 0
    while n % 2 == 0:
        count += 1
        n = n // 2
    fac.append(2) if count > 0 else 0
    powers.append(count) if count > 0 else 0
    # n must be odd at this point
    # so a skip of 2 ( i = i + 2) can be used
    for i in range(3,int(sqrt(n))+1,2):
        # while i divides n , add i to the list
        count = 0
        while n % i== 0:
            count += 1
            n = n // i
        fac.append(i) if count > 0 else 0
        powers.append(count) if count > 0 else 0
    # Condition if n is a prime
    # number greater than 2
    if n > 2:
        fac.append(n)
        powers.append(1) 
    return [(factor, power) for factor, power in zip(fac, powers)]


def prime_factorization(n: float) -> list[tuple[int, int]]:
    positive_factors = []
    negative_factors = []

    top, bottom = Fraction(n).limit_denominator().as_integer_ratio()
    positive_factors = prime_factorization_int(top)
    negative_factors = [(factor, -power) for factor, power in prime_factorization_int(bottom)]
    final = positive_factors + negative_factors
    final.sort()
    return final

def custom_factors(n: float, custom_factors: list[float]):
    num_factors = prime_factorization(n)
    factors_check = [prime_factorization(custom) for custom in custom_factors]
    final = []
    for fac, custom in zip(factors_check, custom_factors):
        if all(prime_fac[0] in [e[0] for e in num_factors] for prime_fac in fac):
        #if all(prime_fac[0] == prime_n[0] and abs(prime_fac[1]) <= abs(prime_n[1]) for prime_fac, prime_n in zip(fac, num_factors)):
            matching = [a for a in num_factors if a[0] in [b[0] for b in fac]]
            not_matching = [a for a in num_factors if a not in matching]
            power = min([prime_n[1]//prime_fac[1] for prime_fac, prime_n in zip(fac, matching)], key=abs)
            num_factors = [(prime_n[0], prime_n[1] - prime_fac[1]*power) for prime_fac, prime_n in zip(fac, matching) if prime_n[1] - prime_fac[1]*power != 0] + not_matching
            final.append((custom, power))
    return final + num_factors

class DefaultUnits(IntEnum):
    none = 1
    meter = 2
    second = 3
    kilogram = 5
    kelvin = 7
    ampere = 11
    mol = 13
    candela = 17

    def __str__(self):
        match self:
            case DefaultUnits.none:
                return ""
            case DefaultUnits.meter:
                return "m"
            case DefaultUnits.second:
                return "s"
            case DefaultUnits.kilogram:
                return "kg"
            case DefaultUnits.kelvin:
                return "K"
            case DefaultUnits.ampere:
                return "A"
            case DefaultUnits.mol:
                return "mol"
            case DefaultUnits.candela:
                return "cd"

class Quantity:
    _SI_map: dict[int, str] = {2: "m", 3:"s", 5:"kg", 7:"K", 11:"A", 13:"mol", 17:"cd"}
    def __init__(self, value: Number = 1, units: float = 1, expected_units: list = [], custom_string: str = "", expect_self: bool = False) -> None:
        self.units = units
        self.value = value
        self.expected_units: list[Quantity] = expected_units
        self.unit_map = Quantity._SI_map
        self.custom_string: str = custom_string
        if expect_self:
            self.expected_units = [self]

    def _units_to_strings(self) -> tuple[Number, str]:
        units_vals = self.units
        value = self.value
        custom_map = self.unit_map | {unit.units:unit.custom_string for unit in self.expected_units}
        units = []
        factors = custom_factors(units_vals, [e.units for e in self.expected_units])
        for unit in self.expected_units:
            try:
                power = [e[1] for e in factors if e[0] == unit.units][0]
            except:
                continue
            value /= unit.value**power

        units = [f"{custom_map.get(e[0])}^{e[1]}" if e[1] != 1 else f"{custom_map.get(e[0])}" for e in factors]
        return value, " * ".join(units)

    def _set_custom_string(self, text: str) -> None:
        self.custom_string = text

    def _get_expected_units(self, other, division: bool = False) -> list | None:
        if self.expected_units[0].units == DefaultUnits.none or other.expected_units[0].units == DefaultUnits.none:
            unit1, unit2 = self.expected_units[0], other.expected_units[0]
            newUnit = Quantity(value=unit1.value*unit2.value, units=unit1.units*unit2.units)
            newUnit.custom_string = unit1.custom_string + unit2.custom_string
            return [newUnit]
        final_self = [a for a in self.expected_units if a.units not in [e.units for e in other.expected_units]]
        return final_self + other.expected_units
    
    def __str__(self, significant_digits: int = 3): 
        val, units = self._units_to_strings() 
        modifier = "%." + str(significant_digits) + "g"
        return f"{modifier % val} {units}"

    def __repr__(self):
        return str(self)

    # Multiplication
    
    def __mul__(self, other):
        if isinstance(other, Number):
            return Quantity(value=self.value*other, units=self.units, expected_units=self.expected_units, custom_string=self.custom_string)
        if isinstance(other, Quantity):
            return Quantity(value=self.value*other.value, units=self.units*other.units, expected_units=self._get_expected_units(other))

    def __rmul__(self, other):
        if isinstance(other, Number):
            return Quantity(value=self.value*other, units=self.units, expected_units=self.expected_units, custom_string=self.custom_string)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return Quantity(value=self.value/other, units=self.units, custom_string=self.custom_string, expected_units=self.expected_units)
        if isinstance(other, Quantity):
            return Quantity(value=self.value/other.value, units=self.units/other.units, expected_units=self._get_expected_units(other))

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return Quantity(value=other/self.value, units=1/self.units, custom_string=self.custom_string, expected_units=self.expected_units)

    def __floordiv__(self, other):
        if isinstance(other, Number):
            return Quantity(value=self.value//other, units=self.units, custom_string=self.custom_string, expected_units=self.expected_units)
        if isinstance(other, Quantity):
            return Quantity(value=self.value//other.value, units=self.units/other.units, expected_units=self._get_expected_units(other))
    def __rfloordiv__(self, other):
        if isinstance(other, Number):
            return Quantity(value=other//self.value, units=1/self.units, custom_string=self.custom_string, expected_units=self.expected_units)
    def __mod__(self, other):
        if isinstance(other, Number):
            return Quantity(value=self.value%other, units=self.units, custom_string=self.custom_string, expected_units=self.expected_units)
        if isinstance(other, Quantity):
            return Quantity(value=self.value%other.value, units=self.units/other.units, expected_units=self._get_expected_units(other))
    def __rmod__(self, other):
        if isinstance(other, Number):
            return Quantity(value=other%self.value, units=1/self.units, custom_string=self.custom_string, expected_units=self.expected_units)

    def __pow__(self, other):
        if isinstance(other, int):
            return Quantity(value=self.value**other, units=self.units**other, expected_units=self.expected_units)

    # Comparison
    def __lt__(self, other):
        if isinstance(other, Quantity):
            if other.units == self.units:
                return self.value < other.value
            else:
                raise ValueError("Units do not match!")
        else:
            raise ValueError("Comparison is only valid between Quantities")
    def __gt__(self, other):
        return not self < other
    def __eq__(self, other):
        if isinstance(other, Quantity):
            if other.units == self.units:
                return self.value == other.value
            else:
                raise ValueError("Units do not match!")
        else:
            raise ValueError("Comparison is only valid between Quantities")
    def __le__(self, other):
        return self < other or self == other
    def __ge__(self, other):
        return self > other or self == other
    def __ne__(self, other):
        return not self == other

    
    # Addition
    def __neg__(self):
        return Quantity(value=-self.value, units=self.units, custom_string=self.custom_string, expected_units=self.expected_units)
    def __add__(self, other):
        if isinstance(other, Quantity):
            if self.units != other.units:
                raise ValueError("Units do not match!")
            return Quantity(value=self.value+other.value, units=self.units, expected_units=self._get_expected_units(other))
        else:
            raise ValueError("Quantities can only be added with other quantities")
    def __radd__(self, other):
        if not isinstance(other, Quantity):
            raise ValueError("Quantities can only be added with other quantities")
    def __sub__(self, other):
        return self.__add__(-other)

    def __float__(self):
        return float(self.value)
    
    # Rounding

    def __round_general__(self, method: Callable, decimals: int | None = None):
        value = self.value
        value /= prod([e.value for e in self.expected_units])
        rounded = method(value, decimals) if decimals is not None else method(value)
        value = rounded * prod([e.value for e in self.expected_units])
        return Quantity(value=value, units=self.units, custom_string=self.custom_string, expected_units=self.expected_units)
    def __round__(self, decimals = 0): return self.__round_general__(round, decimals)
    def __ceil__(self): return self.__round_general__(ceil)
    def __floor__(self): return self.__round_general__(floor)
    def __trunc__(self): return self.__round_general__(trunc)

    # For numpy support:

    def sqrt(self): return sqrt(self.value)
    def rint(self): return self.__round__()
    def log(self): return log(self.value)
    
def work_on_units(func):
    def wrapper(*args, **kwargs):
        if args:
            args = list(args)
            arg = args.pop(0)
            if isinstance(arg, Quantity):
                final = func(arg, *args, **kwargs) if args else func(arg, **kwargs)
            elif isinstance(arg, list):
                final = []
                for a in arg:
                    final.append(wrapper(a, *args, **kwargs) if args else wrapper(a, **kwargs))
            else:
                final = arg
            return final
        else:
            arg = kwargs.pop(list(kwargs.keys())[0])
            if isinstance(arg, Quantity):
                final = func(arg, **kwargs) if args else func(arg)
            elif isinstance(arg, list):
                final = []
                for a in arg:
                    final.append(wrapper(a, **kwargs) if args else wrapper(a, **kwargs))
            else:
                final = arg
            return final
    return wrapper
@work_on_units
def remove_units(x: Quantity):
    val = x.value
    for unit in x.expected_units:
        val /= unit.value
    return val

@work_on_units
def to_SI(x: Quantity):
    return Quantity(x.value, x.units, [], custom_string = x.custom_string)

@work_on_units
def to_units(x, units):
    if not isinstance(units, list):
        units = [units]
    final = []
    for unit in units:
        final += unit.expected_units
    expected_units = final
    return Quantity(x.value, x.units, expected_units, x.custom_string)
