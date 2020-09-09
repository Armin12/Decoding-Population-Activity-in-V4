import numpy as np
from probes_coord import dim_calculator


def probe_distance(probe1, probe2):
    """Probes distance
    This function calculates Euclidean distance between two probes in visual 
    degrees
    """
    dim = dim_calculator()
    delta_dim = dim[probe1 - 1] - dim[probe2 - 1]
    dist = np.sqrt(delta_dim[0] ** 2 + delta_dim[1] ** 2)
    return dist


def probe_distances_array(probes_pairs):
    """Distances of all probe pairs
    This function calculates distances between each pair of probes given by 
    probes_pairs
    """
    distance_array = np.zeros(len(probes_pairs), dtype=float)
    for i in np.arange(len(probes_pairs)):
        probe1, probe2 = probes_pairs[i, 0], probes_pairs[i, 1]
        distance_array[i] = probe_distance(probe1, probe2)
    return distance_array


def Probe_eccentricity(probe, dim):
    """Probe eccentricity
    This function calculates the eccentricity of a probe
    """
    probe_ecc = np.sqrt(dim[probe - 1, 0] ** 2 + dim[probe - 1, 1] ** 2)
    return probe_ecc


def Pair_eccentricity(probe_pairs):
    """Probe pair eccentricity
    This function calculates mean eccentricity of two probes for an array of 
    probe pairs
    """
    dim = dim_calculator()
    ecc_pairs = np.zeros(len(probe_pairs))
    for i in np.arange(len(probe_pairs)):
        ecc_pairs[i] = 0.5 * (Probe_eccentricity(probe_pairs[i, 0], dim) + Probe_eccentricity(probe_pairs[i, 1], dim))
    return ecc_pairs


def Magnification_factor(ecc_pairs):
    """Cortical magnification factor
    This function calculates cortical magnification factor at specific 
    eccentricities for V4. Using equation M=3.01E^(-0.9) from Gattass paper, we 
    calculate the cortical magnification factor in V4 corresponding to mean 
    eccentricity of each probe pair.
    """
    mgf_pairs = np.zeros(len(ecc_pairs))
    for i in np.arange(len(ecc_pairs)):
        mgf_pairs[i] = 3.01 * ecc_pairs[i] ** (-0.9)
    return mgf_pairs


def cortical_distance(distance_array, mgf_pairs):
    """Cortical distance
    This function finds cortical distance by multiplying cortical 
    magnification factor and probes distance. We calculate cortical 
    distance between each pair of probes by multiplying their visual separation 
    in degree and corresponding magnification factor.
    """
    cortical_distances = np.multiply(distance_array, mgf_pairs)
    return cortical_distances
