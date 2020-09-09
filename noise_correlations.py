import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn import preprocessing
from mean_bin import mean_relationship_twoD
import itertools


def MUA_noise_correlation_dataset(MUA, stim, probe):
    """
    This function creates MUA response dataset for repeated presentation of a 
    particular stimulus. Responses are then standardized.
    """
    arr_im = MUA[stim == probe]
    noise_dataset = np.sum(arr_im[:, :, 16:22], axis=2)
    noise_dataset = preprocessing.StandardScaler().fit_transform(noise_dataset)
    return noise_dataset


def LFP_noise_correlation_dataset(LFP, stim, probe):
    """
    This function creates LFP response dataset for repeated presentation of a 
    particular stimulus. Responses are then standardized.
    """
    arr_im = LFP[stim == probe]
    noise_dataset = np.sum(arr_im[:, :, 200:225], axis=2)
    noise_dataset = preprocessing.StandardScaler().fit_transform(noise_dataset)
    return noise_dataset


def noise_correlation(noise_dataset, electrode1, electrode2):
    """
    This function calculates Pearson correlation coefficient between two 
    electrodes. Since noise_data is standardized, this function determines
    outliers by comparing the absolute values to 3 and removes trials for which 
    responses of either electrodes is outlier.
    """
    pair_dataset = noise_dataset[:, np.array([electrode1, electrode2])]
    outresponses = np.argwhere(np.absolute(pair_dataset) > 3)
    out_trials = np.unique(outresponses[:, 0])
    pair_dataset = np.delete(pair_dataset, out_trials, 0)
    nc_value, p_val = pearsonr(pair_dataset[:, 0], pair_dataset[:, 1])
    return nc_value


def MUA_all_pairs_noise_correlation(MUA, stim, probe, electrode_set):
    """
    This function calculates MUA noise correlations for all the pairs of 
    electrodes for a particular stimulus.
    """
    noise_dataset = MUA_noise_correlation_dataset(MUA, stim, probe)
    electrode_pairs = list(itertools.combinations(list(electrode_set), 2))
    electrode_pairs = np.array(electrode_pairs, dtype=int)

    nc_array = np.zeros(len(electrode_pairs))
    for i in np.arange(len(electrode_pairs)):
        print(i)
        electrode1, electrode2 = electrode_pairs[i, 0], electrode_pairs[i, 1]
        nc_array[i] = noise_correlation(noise_dataset, electrode1, electrode2)
    return nc_array


def LFP_all_pairs_noise_correlation(LFP, stim, probe, electrode_set):
    """
    This function calculates LFP noise correlations for all the pairs of 
    electrodes for a particular stimulus.
    """
    noise_dataset = LFP_noise_correlation_dataset(LFP, stim, probe)
    electrode_pairs = list(itertools.combinations(list(electrode_set), 2))
    electrode_pairs = np.array(electrode_pairs, dtype=int)

    nc_array = np.zeros(len(electrode_pairs))
    for i in np.arange(len(electrode_pairs)):
        print(i)
        electrode1, electrode2 = electrode_pairs[i, 0], electrode_pairs[i, 1]
        nc_array[i] = noise_correlation(noise_dataset, electrode1, electrode2)
    return nc_array


def MUA_all_probes_all_pairs_noise_correlation(MUA, stim, electrode_set, MUA_responsive_probes):
    """
    This function calculates MUA noise correlations for all electrode pairs 
    and all the stimuli
    """
    electrode_pairs = list(itertools.combinations(list(electrode_set), 2))
    electrode_pairs = np.array(electrode_pairs, dtype=int)
    nc_array = np.zeros((len(np.unique(stim)), len(electrode_pairs)))
    nc_array[:] = np.nan
    for probe in MUA_responsive_probes:
        nc_array[probe - 1, :] = MUA_all_pairs_noise_correlation(MUA, stim, probe, electrode_set)
    return nc_array


def LFP_all_probes_all_pairs_noise_correlation(LFP, stim, electrode_set, LFP_responsive_probes):
    """
    This function calculates LFP noise correlations for all electrode pairs 
    and all the stimuli
    """
    electrode_pairs = list(itertools.combinations(list(electrode_set), 2))
    electrode_pairs = np.array(electrode_pairs, dtype=int)
    nc_array = np.zeros((len(np.unique(stim)), len(electrode_pairs)))
    nc_array[:] = np.nan
    for probe in LFP_responsive_probes:
        nc_array[probe - 1, :] = LFP_all_pairs_noise_correlation(LFP, stim, probe, electrode_set)
    return nc_array


def electrode_dist(plxarray, electrode1, electrode2):
    """
    This function calculates distance between two electrodes on the array
    """
    plxarray = plxarray - 1
    coord1 = np.ravel(np.argwhere(plxarray == electrode1))
    coord2 = np.ravel(np.argwhere(plxarray == electrode2))
    coord = coord1 - coord2
    dist12 = np.sqrt(coord[0] ** 2 + coord[1] ** 2)
    return dist12


def all_pairs_electrodes_distances(plxarray, electrode_set):
    """
    This function calculates distances between all the electrodes on the
    electrode array
    """
    electrode_pairs = list(itertools.combinations(list(electrode_set), 2))
    electrode_pairs = np.array(electrode_pairs, dtype=int)

    electrodes_dist_pairs = np.zeros(len(electrode_pairs))
    for i in np.arange(len(electrode_pairs)):
        print(i)
        electrode1, electrode2 = electrode_pairs[i, 0], electrode_pairs[i, 1]
        electrodes_dist_pairs[i] = electrode_dist(plxarray, electrode1, electrode2)
    return 0.4 * electrodes_dist_pairs


def Probes_noise_correlation_plotter(probes_nc, v_min, v_max, filename):
    """
    Plots noise correlation of probes
    """
    probes_nc = np.reshape(probes_nc, (10, 10)).T
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(probes_nc, vmin=v_min, vmax=v_max)
    plt.xticks(np.array([0, 9]), [-36, 0])
    plt.yticks(np.array([0, 9]), [2, -34])
    plt.xlabel('Position(deg)', fontweight='bold')
    plt.ylabel('Position(deg)', fontweight='bold')
    plt.subplots_adjust()
    cax = plt.axes([1, 0.23, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min, v_max])
    plt.show()
    fig.savefig(filename, dpi=150)


def Probes_noise_correlation_plotter_P3(probes_nc, v_min, v_max, filename):
    """
    Plots noise correlation of probes
    """
    probes_nc = np.reshape(probes_nc, (10, 10)).T
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(probes_nc, vmin=v_min, vmax=v_max)
    plt.xticks(np.array([0, 9]), [-36 + 20, 0 + 20])
    plt.yticks(np.array([0, 9]), [2, -34])
    plt.xlabel('Position(deg)', fontweight='bold')
    plt.ylabel('Position(deg)', fontweight='bold')
    plt.subplots_adjust()
    cax = plt.axes([1, 0.23, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min, v_max])
    plt.show()
    fig.savefig(filename, dpi=150)


def noisecorrelation_distance_plotter(noisecorrelation, electrodes_distance, y_lim_min, y_lim_max, figure_title,
                                      file_name):
    """
    Plots noise correlations vs electrodes distances
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    y_x = mean_relationship_twoD(electrodes_distance, noisecorrelation, np.arange(0, 4.5, 0.5))

    ax.plot(np.arange(0, 4, 0.5) + 0.25, y_x[0], marker='o', linewidth=3.0, color=(0.419, 0.039, 0.741),
            label='Ecc<5 deg')
    ax.plot(np.arange(0, 4, 0.5) + 0.25, y_x[1], marker='o', linewidth=3.0, color=(0.419, 0.039, 0.741))
    ax.plot(np.arange(0, 4, 0.5) + 0.25, y_x[2], marker='o', linewidth=3.0, color=(0.741, 0.701, 0.039),
            label='8<Ecc<12')
    ax.plot(np.arange(0, 4, 0.5) + 0.25, y_x[3], marker='o', linewidth=3.0, color=(0.741, 0.701, 0.039))
    ax.plot(np.arange(0, 4, 0.5) + 0.25, y_x[4], marker='o', linewidth=3.0, color=(0.039, 0.741, 0.525),
            label='16<Ecc<20')
    ax.plot(np.arange(0, 4, 0.5) + 0.25, y_x[5], marker='o', linewidth=3.0, color=(0.039, 0.741, 0.525))

    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(y_lim_min, y_lim_max)
    plt.ylabel('Noise correlation', fontweight='bold')
    plt.xlabel('Electrodes distances (mm)', fontweight='bold')
    plt.title(figure_title, fontweight='bold', loc='center')
    ax.legend(loc='lower right', fontsize=28)
    plt.show()
    fig.savefig(file_name, dpi=200)


def noisecorrelation_mean_importance_plotter(noisecorrelation, pair_mean_importance, bins_values, centre_points,
                                             y_lim_min, y_lim_max, figure_title, file_name):
    """
    Plots noise correlations vs electrodes mean importance
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    y_x = mean_relationship_twoD(pair_mean_importance, noisecorrelation, bins_values)

    ax.plot(centre_points, y_x[0], marker='o', linewidth=3.0, color=(0.419, 0.039, 0.741), label='Ecc<5 deg')
    ax.plot(centre_points, y_x[1], marker='o', linewidth=3.0, color=(0.419, 0.039, 0.741))
    ax.plot(centre_points, y_x[2], marker='o', linewidth=3.0, color=(0.741, 0.701, 0.039), label='8<Ecc<12')
    ax.plot(centre_points, y_x[3], marker='o', linewidth=3.0, color=(0.741, 0.701, 0.039))
    ax.plot(centre_points, y_x[4], marker='o', linewidth=3.0, color=(0.039, 0.741, 0.525), label='16<Ecc<20')
    ax.plot(centre_points, y_x[5], marker='o', linewidth=3.0, color=(0.039, 0.741, 0.525))

    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(y_lim_min, y_lim_max)
    plt.ylabel('Noise correlation', fontweight='bold')
    plt.xlabel('Electrodes importance', fontweight='bold')
    plt.title(figure_title, fontweight='bold', loc='center')
    ax.legend(loc='lower right', fontsize=28)
    plt.show()
    fig.savefig(file_name, dpi=200)
