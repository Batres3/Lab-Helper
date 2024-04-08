from . import __define_unit
from .prefixes import *
from ..units import Quantity

meter = Quantity(1, 2, custom_string="m", expect_self=True)
second = Quantity(1, 3, custom_string="s", expect_self=True)
kilogram = Quantity(1, 5, custom_string="kg", expect_self=True)
gram = Quantity(1e-3, 5, custom_string="g", expect_self=True)
kelvin = Quantity(1, 7, custom_string="K", expect_self=True)
ampere = Quantity(1, 11, custom_string="A", expect_self=True)
mol = Quantity(1, 13, custom_string="mol", expect_self= True)
candela = Quantity(1, 17,custom_string="cd", expect_self=True)

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
