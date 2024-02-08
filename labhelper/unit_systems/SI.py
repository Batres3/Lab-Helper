from . import __define_unit
from .prefixes import *
from ..units import Quantity, DefaultUnits

meter = Quantity(1, DefaultUnits.meter, custom_string="m", expect_self=True)
second = Quantity(1, DefaultUnits.second, custom_string="s", expect_self=True)
kilogram = Quantity(1, DefaultUnits.kilogram, custom_string="kg", expect_self=True)
gram = Quantity(1e-3, DefaultUnits.kilogram, custom_string="g", expect_self=True)
kelvin = Quantity(1, DefaultUnits.kelvin, custom_string="K", expect_self=True)
ampere = Quantity(1, DefaultUnits.ampere, custom_string="A", expect_self=True)
mol = Quantity(1, DefaultUnits.mol, custom_string="mol", expect_self= True)
candela = Quantity(1, DefaultUnits.candela,custom_string="cd", expect_self=True)

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
electronvolt = 1.6e-19*joule
__define_unit(electronvolt, "eV")
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
