from numbers import Number
from enum import IntEnum
from fractions import Fraction
from math import sqrt

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
    def __init__(self, value: Number = 1, units: float = 1, expected_units: list = [], custom_string: str = "") -> None:
        self.units = units
        self.value = value
        self.expected_units: list[Quantity] = expected_units
        self.unit_map = Quantity._SI_map
        self.custom_string: str = custom_string

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
        if (self.units == DefaultUnits.none or other.units == DefaultUnits.none):
            newUnit = Quantity(value=self.value*other.value, units=self.units*other.units)
            newUnit.custom_string = self.custom_string + other.custom_string
            if self.units == DefaultUnits.none:
                try:
                    other.expected_units[0] = newUnit
                except:
                    other.expected_units = [newUnit]
            else:
                try:
                    self.expected_units[0] = newUnit
                except:
                    self.expected_units = [newUnit]
        return list(set(self.expected_units + other.expected_units))
    
    def to_SI(self):
        self.expected_units = []
        return self
    
    def to_units(self, units):
        if not isinstance(units, list):
            units = [units]
        self.expected_units = units
        return self

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
            return Quantity(value=other/self.value, units=1/self.units, expected_units=[unit**-1 for unit in self.expected_units])

    def __pow__(self, other):
        if isinstance(other, int):
            return Quantity(value=self.value**other, units=self.units**other, expected_units=self.expected_units)

    def __float__(self):
        return self.value

# ------------ DEFINITIONS ---------------

meter = Quantity(1, DefaultUnits.meter, custom_string="m")
second = Quantity(1, DefaultUnits.second, custom_string="s")
kilogram = Quantity(1, DefaultUnits.kilogram, custom_string="kg")
gram = Quantity(1e-3, DefaultUnits.kilogram, custom_string="g")
kelvin = Quantity(1, DefaultUnits.kelvin, custom_string="K")
ampere = Quantity(1, DefaultUnits.ampere, custom_string="A")
mol = Quantity(1, DefaultUnits.mol, custom_string="mol")
candela = Quantity(1, DefaultUnits.candela, custom_string="cd")

quetta = Quantity(1e30, units=DefaultUnits.none, custom_string="Q")
ronna  = Quantity(1e27, units=DefaultUnits.none, custom_string="R")
yotta  = Quantity(1e24, units=DefaultUnits.none, custom_string="Y")
zetta  = Quantity(1e21, units=DefaultUnits.none, custom_string="Z")
exa    = Quantity(1e18, units=DefaultUnits.none, custom_string="E")
peta   = Quantity(1e15, units=DefaultUnits.none, custom_string="P")
tera   = Quantity(1e12, units=DefaultUnits.none, custom_string="T")
giga   = Quantity(1e9, units=DefaultUnits.none, custom_string="G")
mega   = Quantity(1e6, units=DefaultUnits.none, custom_string="M")
kilo   = Quantity(1e3, units=DefaultUnits.none, custom_string="k")
hecto  = Quantity(1e2, units=DefaultUnits.none, custom_string="h")
deca   = Quantity(1e1, units=DefaultUnits.none, custom_string="da")
deci   = Quantity(1e-1, units=DefaultUnits.none, custom_string="d")
centi  = Quantity(1e-2, units=DefaultUnits.none, custom_string="c")
mili   = Quantity(1e-3, units=DefaultUnits.none, custom_string="m")
micro  = Quantity(1e-6, units=DefaultUnits.none, custom_string="µ")
nano   = Quantity(1e-9, units=DefaultUnits.none, custom_string="n")
pico   = Quantity(1e-12, units=DefaultUnits.none, custom_string="p")
femto  = Quantity(1e-15, units=DefaultUnits.none, custom_string="f")
atto   = Quantity(1e-18, units=DefaultUnits.none, custom_string="a")
zepto  = Quantity(1e-21, units=DefaultUnits.none, custom_string="z")
yocto  = Quantity(1e-24, units=DefaultUnits.none, custom_string="y")
ronto  = Quantity(1e-27, units=DefaultUnits.none, custom_string="r")
quecto = Quantity(1e-30, units=DefaultUnits.none, custom_string="q")

def __define_unit(unit, symbol) -> None:
    unit._set_custom_string(symbol)
    unit.expected_units = [unit]

hertz = second**-1
__define_unit(hertz, "Hz")
newton = kilogram*meter*second**-2
__define_unit(newton, "N")
pascal = newton/meter**2
__define_unit(pascal, "Pa")
joule = newton*meter
__define_unit(joule, "J")
watt = joule/second
__define_unit(watt, "W")
coulomb = ampere*second
__define_unit(coulomb, "C")
volt = joule/coulomb
__define_unit(volt, "V")
farad = coulomb/volt
__define_unit(farad, "F")
ohm = volt/ampere
__define_unit(ohm, "Ω")
siemens = ohm**-1
__define_unit(siemens, "S")
weber = volt*second
__define_unit(weber, "Wb")
tesla = weber/meter**2
__define_unit(tesla, "T")
henry = weber/ampere
__define_unit(henry, "H")
# TODO: ºC

lumen = 1*candela
__define_unit(lumen, "lm")
lux = lumen/meter**2
__define_unit(lux, "lx")
becquerel = second**-1
__define_unit(becquerel, "Bq")
gray = joule/kilogram
__define_unit(gray, "Gy")
sievert = joule/kilogram
__define_unit(sievert, "Sv")
katal = mol*second**-1
__define_unit(katal, "kat")