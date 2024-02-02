from numbers import Number
from enum import Enum, IntEnum
from math import sqrt
from fractions import Fraction
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


def prime_factorization(n: float, custom_factors: list[float] | None = None) -> list[tuple[int, int]]:
    positive_factors = []
    negative_factors = []
    if custom_factors is not None:
        for factor in custom_factors:
            count = 0
            print(n % factor)
            while n % factor == 0:
                count += 1
                n = n//factor
            positive_factors.append((factor, count))

    top, bottom = Fraction(n).limit_denominator().as_integer_ratio()
    positive_factors = prime_factorization_int(top)
    negative_factors = [(factor, -power) for factor, power in prime_factorization_int(bottom)]
    return positive_factors + negative_factors

class Prefix(Enum):
    QUETTA = 1e30
    RONNA  = 1e27
    YOTTA  = 1e24
    ZETTA  = 1e21
    EXA    = 1e18
    PETA   = 1e15
    TERA   = 1e12
    GIGA   = 1e9
    MEGA   = 1e6
    KILO   = 1e3
    HECTO  = 1e2
    DECA   = 1e1
    NONE   = 1
    DECI   = 1e-1
    CENTI  = 1e-2
    MILI   = 1e-3
    MICRO  = 1e-6
    NANO   = 1e-9
    PICO   = 1e-12
    FEMTO  = 1e-15
    ATTO   = 1e-18
    ZEPTO  = 1e-21
    YOCTO  = 1e-24
    RONTO  = 1e-27
    QUECTO = 1e-30
    
    def __mul__(self, other):
        if isinstance(other, Number):
            return other * self.value
        if isinstance(other, Prefix):
            try:
                return Prefix(self.value * other.value)
            except:
                return self.value * other.value
    def __rmul__(self, other):
        if isinstance(other, Number):
            return other * self.value
    def __truediv__(self, other):
        if isinstance(other, Number):
            return self.value / other
        if isinstance(other, Prefix):
            try:
                return Prefix(self.value / other.value)
            except:
                return self.value / other.value
    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return other / self.value
    
    def __float__(self):
        return float(self.value)
    def __int__(self):
        return int(self.value)

    def __str__(self):
        match self:
            case Prefix.QUETTA:
                return "Q"
            case Prefix.RONNA:
                return "R"
            case Prefix.YOTTA:
                return "Y"
            case Prefix.ZETTA: 
                return "Z"
            case Prefix.EXA: 
                return "E"
            case Prefix.PETA:  
                return "P"
            case Prefix.TERA:  
                return "T"
            case Prefix.GIGA:  
                return "G"
            case Prefix.MEGA:  
                return "M"
            case Prefix.KILO:  
                return "k"
            case Prefix.HECTO: 
                return "h"
            case Prefix.DECA:  
                return "da"
            case Prefix.NONE:  
                return ""
            case Prefix.DECI:  
                return "d"
            case Prefix.CENTI: 
                return "c"
            case Prefix.MILI:  
                return "m"
            case Prefix.MICRO: 
                return "µ"
            case Prefix.NANO:  
                return "n"
            case Prefix.PICO:  
                return "p"
            case Prefix.FEMTO: 
                return "f"
            case Prefix.ATTO:  
                return "a"
            case Prefix.ZEPTO: 
                return "z"
            case Prefix.YOCTO: 
                return "y"
            case Prefix.RONTO: 
                return "r"
            case Prefix.QUECTO:
                return "q"

class Unit(Enum):
    NONE = 1
    METER = 2
    SECOND = 3
    KILOGRAM = 5
    KELVIN = 7
    AMPERE = 11
    MOL = 13
    CANDELA = 17
    HERTZ = 1 / SECOND
    NEWTON = METER * KILOGRAM * (1/SECOND**2)
    def __mul__(self, other):
        if isinstance(other, Number):
            return Quantity(value=other, units=self)
        if isinstance(other, Prefix):
            return Quantity(value=other.value, units=self)
        if isinstance(other, Unit):
            return Quantity(units=Unit(self.value * other.value))
    def __rmul__(self, other):
        if isinstance(other, Number):
            return Quantity(value=other, units=self)
        if isinstance(other, Prefix):
            return Quantity(value=other.value, units=self)
    def __truediv__(self, other):
        if isinstance(other, Number):
            return Quantity(1/other, (self.value))
        if isinstance(other, BasicUnit):
            return Quantity(unit=(self.value)) / Quantity(unit=(other))
    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return Quantity(other, (float(1/self)))
        if isinstance(other, BasicUnit):
            return Quantity(unit=(other)) / Quantity(unit=(self.value))

    def __int__(self):
        return int(self.value)
    def __float__(self):
        return float(self.value)

    def __str__(self):
        match self:
            case Unit.METER:
                return "m"
            case Unit.SECOND:
                return "s"
            case Unit.KILOGRAM:
                return "kg"
            case Unit.KELVIN:
                return "K"
            case Unit.AMPERE:
                return "A"
            case Unit.MOL:
                return "mol"
            case Unit.CANDELA:
                return "cd"

class DefaultUnits(IntEnum):
    meter = 2
    second = 3
    kilogram = 5
    kelvin = 7
    ampere = 11
    mol = 13
    candela = 17

    def __str__(self):
        match self:
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
    def __init__(self, value: Number = 1, units: float = 1, expected_units: list | None = None) -> None:
        self.units = units
        self.value = value
        self.expected_units: list[Quantity] | None = expected_units
        self.custom_string: str | None = None

    def _units_to_strings(self) -> tuple[Number, str]:
        if self.expected_units is None:
            factors = prime_factorization(self.units)
            value = self.value
            units = [ f"{str(DefaultUnits(factor))}^{power}" if power != 1 else f"{str(DefaultUnits(factor))}" for factor, power in factors]
        else:
            self._validate_units()
            units_vals = self.units
            value = self.value
            units = []
            for unit in self.expected_units:
                units_vals /= unit.units
                value /= unit.value
                units += unit.custom_string
            remaining_factors = prime_factorization(units_vals)
            units += [f"{str(DefaultUnits(factor))}^{power}" if power != 1 else f"{str(DefaultUnits(factor))}" for factor, power in remaining_factors]
        return value, " * ".join(units)

    def _set_custom_string(self, text: str) -> None:
        self.custom_string = text

    def _get_expected_units(self, other, division: bool = False) -> list | None:
        if self.expected_units is not None:
            if other.expected_units is not None:
                return self.expected_units + other.expected_units
            else:
                return self.expected_units
        else:
            if other.expected_units is not None:
                return other.expected_units
            else:
                return None

    def _validate_units(self) -> None:
        if self.expected_units is None:
            raise ValueError("expected_units are not specified!")
        units = self.units
        for unit in self.expected_units:
            if units % unit.units != 0:
                raise ValueError("expected_units do not match with given units!")
            if unit.custom_string is None:
                raise ValueError("custom_string not specified!")
            units /= unit.units
        
    def __str__(self): 
        val, units = self._units_to_strings() 
        return f"{val} {units}"

    def __mul__(self, other):
        if isinstance(other, Number):
            return Quantity(value=self.value*other, units=self.units, expected_units=self.expected_units)
        if isinstance(other, Quantity):
            return Quantity(value=self.value*other.value, units=self.units*other.units, expected_units=self._get_expected_units(other))

    def __rmul__(self, other):
        if isinstance(other, Number):
            return Quantity(value=self.value*other, units=self.units, expected_units=self.expected_units)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return Quantity(value=self.value/other, units=self.units)
        if isinstance(other, Quantity):
            return Quantity(value=self.value/other.value, units=self.units/other.units, expected_units=self._get_expected_units(other))

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            if self.expected_units is None:
                return Quantity(value=other/self.value, units=1/self.units) 
            else:
                return Quantity(value=other/self.value, units=1/self.units, expected_units=[unit**-1 for unit in self.expected_units])

    def __pow__(self, other):
        if isinstance(other, int):
            if self.expected_units is None:
                return Quantity(value=self.value**other, units=self.units**other)
            else:
                return Quantity(value=self.value**other, units=self.units**other, expected_units=self.expected_units)


meter = Quantity(1, DefaultUnits.meter)
second = Quantity(1, DefaultUnits.second)
kilogram = Quantity(1, DefaultUnits.kilogram)
gram = Quantity(1e-3, DefaultUnits.kilogram)
kelvin = Quantity(1, DefaultUnits.kelvin)
ampere = Quantity(1, DefaultUnits.ampere)
mol = Quantity(1, DefaultUnits.mol)
candela = Quantity(1, DefaultUnits.candela)

newton = kilogram*meter/second**2
newton._set_custom_string("N")
newton.expected_units = [newton]
#test = 10/newton
#print(newton**2)
print(prime_factorization((5*2/3**2)**2, [5*2/3**2]))
