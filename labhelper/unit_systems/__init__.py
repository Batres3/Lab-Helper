def __define_unit(unit, symbol) -> None:
    unit._set_custom_string(symbol)
    unit.expected_units = [unit]

