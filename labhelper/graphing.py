from seaborn import set_style as ss

def set_style():
    ss(style="darkgrid", rc={"axes.grid": True, "xtick.bottom": True, "ytick.left": True, "axes.edgecolor": ".15", "axes.spines.right": False, "axes.spines.top": False})

