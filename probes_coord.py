import numpy as np


def dim_calculator():
    """Probe coordinations
    This function outputs coordinations of all the probes on the grid (X, Y) 
    assuming that reference is the upper right fixation point. Distance between 
    neighbouring probes is 4 visual degrees. So, the distance between the first 
    and the last probe in a row or column is 36 visual degrees
    """
    probe_set = np.arange(1, 101)
    X = -36 + ((probe_set - 1) // 10) * 4
    Y = 2 - ((probe_set - 1) % 10) * 4
    dim = np.vstack((X, Y)).T
    return dim


def dim_calculatorP3():
    """Probe coordinations
    This function outputs coordinations of all the probes on the grid (X, Y) 
    assuming that reference is the upper middle fixation point. Distance between 
    neighbouring probes is 4 visual degrees. So, the distance between the first 
    and the last probe in a row or column is 36 visual degrees
    """
    probe_set = np.arange(1, 101)
    X = 20 - 36 + ((probe_set - 1) // 10) * 4
    Y = 2 - ((probe_set - 1) % 10) * 4
    dim = np.vstack((X, Y)).T
    return dim
