from . import __define_unit
from .prefixes import *
from ..units import Quantity, DefaultUnits

centimetre = Quantity(1e-2, DefaultUnits.meter, custom_string="cm", expect_self=True)
gram = Quantity(1e-3, DefaultUnits.kilogram, custom_string="g", expect_self=True)
second = Quantity(1, DefaultUnits.second, custom_string="s", expect_self=True)
gal = centimetre/second**2
__define_unit(gal, "Gal")
dyne = gram*centimetre/second**2
__define_unit(dyne, "dyn")
erg = gram * centimetre**2/second**2
__define_unit(erg, "erg")
barye = gram/(centimetre*second**2)
__define_unit(barye, "Ba")
poise = gram/(centimetre*second)
__define_unit(poise, "P")
stokes = centimetre**2/second
__define_unit(stokes, "St")
kayser = centimetre**-1
__define_unit(kayser, "K")

