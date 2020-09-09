import numpy as np


def mean_relationship(x, y, bins_values):
    """Mean of bins
    This function takes two corresponding 1D arrays x and y, and calculates mean 
    of y for specific range of x
    """
    sort_ind_x = np.argsort(x)
    x = x[sort_ind_x]
    y = y[sort_ind_x]
    hist, bin_edges = np.histogram(x, bins=bins_values)
    array_end = np.cumsum(hist)
    array_start = np.cumsum(hist) - hist
    y_x = np.zeros(len(array_start))
    y_x_std = np.zeros(len(array_start))
    for i in np.arange(len(array_start)):
        y_x[i] = np.mean(y[array_start[i]:array_end[i]])
        y_x_std[i] = np.std(y[array_start[i]:array_end[i]])
    return y_x, y_x_std


def mean_relationship_twoD(x, y, bins_values):
    """Mean of bins
    This function takes two corresponding 2D arrays x and y, and calculates mean 
    of y for specific range of x
    """
    sort_ind_x = np.argsort(x)
    x = x[sort_ind_x]
    y = y[:, sort_ind_x]
    hist, bin_edges = np.histogram(x, bins=bins_values)
    array_end = np.cumsum(hist)
    array_start = np.cumsum(hist) - hist
    y_x = np.zeros((len(y), len(array_start)))
    for i in np.arange(len(array_start)):
        y_x[:, i] = np.mean(y[:, array_start[i]:array_end[i]], axis=1)
    return y_x
