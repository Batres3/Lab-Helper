from scipy.stats import t
from scipy.optimize import curve_fit as cfit
from numpy.typing import NDArray
from pandas import Series
import numpy as np
from typing import NamedTuple
from numbers import Number

def random_error_of_mean(std, num_samples, confidence):
    """
    Calculates the random uncertainty of the mean of a set of samples following the formula:

        t_n-1 * (σ_n-1 / √n)
    """
    return std * student_t_n(num_samples - 1, confidence) / np.sqrt(num_samples)

def student_t_n(degrees_of_freedom, confidence):
    """
    Returns the t_n,ɑ/2 coefficient for the given degree of freedom and confidence level.
    Exact same inputs and outputs as the cheat sheet table from year 1
    """
    return t.ppf(1 - (1-confidence)*0.5, degrees_of_freedom)

def coefficient_errors(cov):
    return np.sqrt(np.diag(cov))

def curve_fit(f, xdata, ydata, p0 = None): # simple abstraction
    if isinstance(f, int):
        return np.polyfit(deg=f, x=xdata, y=ydata, cov=True)
    else:
        return cfit(f, xdata, ydata, [np.max(xdata)] * (f.__code__.co_argcount - 1) if not p0 else p0)

# this finds the measurement-related error in a given fitted parameter which is the nth (using python indexing) in the list of fitted parameters being used
def std_fit(f,xdata,ydata,xerr,yerr, n, p0 = None):
    if isinstance(f, int):
        num_vars = f + 1
    else:
        num_vars = f.__code__.co_argcount - 1
    fits=np.zeros((n, num_vars))
    num_data = len(xdata)
    for i in range(n):
        errs = np.random.rand(2, num_data) * 2 - 1 # X and Y are assumed to be the same shape
        errs = np.multiply([xerr, yerr], errs)
        data = errs + [xdata, ydata]
        fittedParameters, _ = curve_fit(f, *data, p0)
        fits[i]=fittedParameters
    return np.std(fits, axis=0)

class FitResults(NamedTuple):
    params: list[float]
    errs: list[float]
    original_errs: list[float]
    boot_errs: list[float]
    rsquared: float
    rmse: float
    pcov: NDArray[np.float32]
    varnames: list[str]
    def __repr__(self):
        final = f"Fit Results:\nR²: {self.rsquared}\nParameters: {list(self.params)}\nRMSE: {self.rmse}\n"
        for name, val, err, original, boot in zip(self.varnames, self.params, self.errs, self.original_errs, self.boot_errs):
            final += f"{name} = {val} ± {err} (original: {original:#.3g}, bootstrap: {boot:#.3g})\n"
        return final

def fit(f, xdata, ydata, xerr = None, yerr = None, p0 = None, num_iter: int = 1000) -> FitResults:
    # type checking
    if isinstance(xerr, Number):
        xerr = np.full((xdata.shape[0],),xerr)
    if isinstance(yerr, Number):
        yerr = np.full((ydata.shape[0],), yerr)
    if f is None:
        f = 1
    polynomial = isinstance(f, int)
    use_errs = xerr is not None and yerr is not None
    fittedParameters, pcov = curve_fit(f, xdata, ydata, p0)
    if polynomial:
        modelPredictions = np.polyval(fittedParameters, xdata)
    else:
        modelPredictions = f(xdata, *fittedParameters)
    absError = modelPredictions - ydata
    rmse = np.sqrt(np.mean(np.square(absError))) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(ydata))
    if use_errs:
        r = std_fit(f,xdata,ydata,xerr,yerr,num_iter, p0)
    else:
        r = np.zeros((pcov.shape[-1]))
    errs = np.sqrt(np.square(r) + np.diag(pcov))
    if polynomial:
        varnames = [f"order {i}" for i in reversed(range(len(fittedParameters)))]
    else:
        varnames = f.__code__.co_varnames[1:]
    return FitResults(fittedParameters, errs, np.sqrt(np.diag(pcov)), r, Rsquared,rmse, pcov, varnames)
