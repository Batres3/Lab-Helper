import pandas as pd
from pandas._config import display
import pyperclip as pc
import re
from .symbolic import identify_error_symbol, Helper
from numpy import zeros
from .units import Quantity, remove_units

def df_switch_columns(df: pd.DataFrame, column1, column2):
    """
    Returns a new DataFrame with the required columns switched (DOES NOT MODIFY ORIGINAL DATAFRAME)
    """
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df

def df_switch_rows(df: pd.DataFrame, row1, row2):
    """
    Returns a new DataFrame with the required rows switched (DOES NOT MODIFY ORIGINAL DATAFRAME), not efficient for very large DataFrames (>1000 rows)
    """
    ids = df.index.tolist()
    a, b = ids.index(row1), ids.index(row2)
    ids[a], ids[b] = ids[b], ids[a]
    df = df.reindex(ids)
    return df

def df_create(columns, indices) -> pd.DataFrame:
    """
    Returns a new DataFrame with the specified columns and indexes,
    these can be given as a list of names (columns -> ["Input 1", "Input 2"], rows -> ["Experiment 1", "Experiment 2"])
    or as a number of columns or indexes.
    It is valid to supply a list for the columns and a number for the indices, and vice versa.
    """
    if type(columns) == int and type(indices) == int:
        return pd.DataFrame(columns=range(columns), index=range(indices)).fillna(0)
    elif type(columns) == int and type(indices) == list:
        return pd.DataFrame(columns=range(columns), index=indices).fillna(0)
    elif type(columns) == list and type(indices) == int:
        return pd.DataFrame(columns=columns, index=range(indices)).fillna(0)
    elif type(columns) == list and type(indices) == list:
        return pd.DataFrame(columns=columns, index=indices).fillna(0)
    else:
        raise TypeError("Only integers or lists are supported!")

def copy_to_clipboard(var: str):
    pc.copy(var)

def get_value_error_pairs(df: pd.DataFrame, possible_error_symbols: list[str] = Helper._possible_error_symbols) -> list[tuple[str, str]]:
    cols = df.columns
    error_symbol = identify_error_symbol(cols, possible_error_symbols)
    if not error_symbol: return []
    error_cols = [e for e in cols if error_symbol in e]
    matching = [e.replace(error_symbol, "") for e in error_cols]
    unmatched = [e for e in matching if e not in cols]
    if unmatched:
        raise ValueError(f"There are unmatched error columns: {unmatched}")

    return list(zip(matching, error_cols))


def df_to_latex(df: pd.DataFrame, number_of_decimals: int | None = 2, index: bool = False, copy_to_clipboard: bool = True):
    """
    Turns Pandas DataFrame into a LaTeX table, formatted as ||r|...|r|| for the given number of columns in the
    DataFrame (because I like the way it looks), automatically copies result into clipboard if copy_to_clipboard is not set to False
    """
    # Rename columns if all their components are units
    if number_of_decimals == None:
        float_format = None
    else:
        float_format = f"%#.{number_of_decimals}g"

    # Utility functions
    def to_string(x) -> str:
        if isinstance(x, str):
            return x
        if isinstance(x, float) and float_format:
            x = float_format % x
            if x[-1] == ".":
                x = x[:-1]
            return x
        return str(x)

    def rename_col(x):
        if not df[x].map(lambda e: isinstance(e, Quantity)).product(): return x
        new_name = f"{x} ({str(df[x][0]).split(' ', 1)[-1]})"
        df[x] = df[x].map(remove_units)
        return new_name

    df = df.copy()
    df.rename(rename_col, axis=1, inplace=True)
    table_format = "|c" * len(df.columns) + "|"
    # Handle errors
    def match_value_to_error(x):
        value, error = x
        error = to_string(error)
        num_decimals = len(error.split(".")[-1])
        value = f"%.{num_decimals}f" % value
        return f"${value} \\pm {error}$"
    for value, error in get_value_error_pairs(df):
        df[error] = df[error].map(to_string)
        df[value] = df[[value, error]].apply(match_value_to_error, axis=1)
        df.drop(error, axis=1, inplace=True)
    df = df.map(to_string)
    basic_latex = df.to_latex(index=index, column_format=table_format)
    latex = basic_latex.replace(r"\toprule", r"\hline").replace(r"\midrule", r"\hline\hline").replace("\\bottomrule\n", "")
    latex = latex.replace(r"\\", r"\\\hline").replace(r"\\\hline", r"\\", 1)
    # Replace all 1.2e+03 with $1.2\cdot10^{3}$
    latex = re.sub(r"(\d{1,}\.?\d*)e\+?(\-?)0*(\d*)", r"$\1\\cdot10^{\2\3}$",latex)
    return latex

def multiindex_df(superindices: str | list[str], subindices: list[str], num_rows: int = 0) -> pd.DataFrame:
    if not isinstance(superindices, list):
        superindices = [superindices]
    cols = len(subindices) * len(superindices)
    data = zeros([num_rows, cols])
    return pd.DataFrame(data=data, columns=pd.MultiIndex.from_product([superindices, subindices]))

