print(__doc__)

# Author: Armin Najarpour Foroushani -- <armin.najarpour@gmail.com>
# Neural analysis

###################### Import Libraries ##########################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import itertools
from activity_visualizer import Signal_visualization
from response_calculator import (MUA_Response_dataset, LFP_Response_dataset,
                                 Aray_response_visualization)
from RF_calc import (RF_set_viewer, tuning_plotter, population_contours,
                     RFCenterPosition, ECConArray, RFCenterEccentricity,
                     RFDiameterEccentricityPlot)
from feature_label_selection import Select_responsive_electrodes, responsive_probes
from mg_factor import (probe_distances_array, Pair_eccentricity,
                       Magnification_factor, cortical_distance)
from discrimination import (Discrimination_performance_array,
                            performance_corticaldistance_plotter,
                            Performance_eccentricity_plotter)
from clustering_RF import (RF_cluster, RF_representative_electrode,
                           performance_corticaldistance_plotter_clustering,
                           Performance_eccentricity_plotter_clustering)
from weight_analysis import (SVM_weight_set, electrodes_sorting_by_weight,
                             electrodes_importance_calculator,
                             electrode_pairs_importance,
                             performance_corticaldistance_plotter_importance,
                             Performance_eccentricity_plotter_importance,
                             Electrode_array_show,
                             Electrodes_importance_histogram,
                             Electrode_array_importance,
                             probes_weight, weight_corticaldist,
                             mean_importance_corticaldist)
from bandpass_lfp import (bandpass_LFP,
                          bandpass_performance_corticaldistance_plotter,
                          Bandpass_Performance_eccentricity_plotter)
from correlation_blind import (Discrimination_performance_array_onblind,
                               performance_corticaldistance_onblind_plotter,
                               Performance_eccentricity_onblind_plotter)
from noise_correlations import (MUA_all_probes_all_pairs_noise_correlation,
                                LFP_all_probes_all_pairs_noise_correlation,
                                all_pairs_electrodes_distances,
                                Probes_noise_correlation_plotter,
                                noisecorrelation_distance_plotter,
                                noisecorrelation_mean_importance_plotter)

###################### Main ##########################

######### Set fonts
font = {'family': 'sans-serif', 'weight': 'bold', 'size': '25'}
rc('font', **font)  # pass in the font dict as kwargs

######### Load data
# Probe numbering on the grid starts from 1 for the probe at the upper left
# and increase to 10 to the bottom and continues to the lower right that is 100.
# stim is an array that gives corresponding probe number (label) to each trial.
# 1769 trials were recorded with their neural activity data in MUA and LFP.
# plxarray represents location of electrodes on the implanted electrode array.
# Edges is the time edges for MUA and LFP_Edges is the time samples for the LFPs

MUA = np.load('MUA.npy')
LFP = np.load('LFP.npy')
stim = np.load('stim.npy')
Edges = np.load('Edges.npy')
LFP_Edges = np.load('LFP_Edges.npy')
plxarray = np.load('plxarray.npy')

######### Visualizing recorded data

# A group of electrodes that respond well to stimuli
electrode_set = np.array([3, 4, 5, 7, 10, 12, 18, 19, 20, 21, 24, 26, 27, 29, 31, 37, 45,
                          48, 49, 50, 54, 58, 60, 72, 75, 89, 90, 91], dtype=int)
# A group of probes 
Probe_set = [23, 29, 55, 82, 88]

# MUA
edge_data = Edges + 0.0125
ylab = 'Spikes'
ylimit_bottom, ylimit_top = -0.01, 3.7
Signal_visualization(MUA, stim, Probe_set, electrode_set, edge_data, ylab, ylimit_bottom, ylimit_top, 'r',
                     'Spike_count_traces.png')

# LFP
edge_data = LFP_Edges
ylab = 'LFP (\u03BC V)'
ylimit_bottom, ylimit_top = -300, 300
Signal_visualization(LFP, stim, Probe_set, electrode_set, edge_data, ylab, ylimit_bottom, ylimit_top, 'b',
                     'LFP_traces.png')

######### Visualizing responses

v_min = -0.27
v_max = 0.27

# MUA response for wide window
MUA_dataset = MUA_Response_dataset(MUA, 'w')
Aray_response_visualization(stim, MUA_dataset, Probe_set, plxarray, v_min, v_max, 'MUA_array_response_wide.png')

# MUA response for medium window
MUA_dataset = MUA_Response_dataset(MUA, 'm')
Aray_response_visualization(stim, MUA_dataset, Probe_set, plxarray, v_min, v_max, 'MUA_array_response_medium.png')

# MUA response for narrow window
MUA_dataset = MUA_Response_dataset(MUA, 'n')
Aray_response_visualization(stim, MUA_dataset, Probe_set, plxarray, v_min, v_max, 'MUA_array_response_narrow.png')

v_min = -0.34
v_max = 0.34

# LFP response for wide window
LFP_dataset = LFP_Response_dataset(LFP, 'w')
Aray_response_visualization(stim, -LFP_dataset, Probe_set, plxarray, v_min, v_max, 'LFP_array_response_wide.png')

# LFP response for medium window
LFP_dataset = LFP_Response_dataset(LFP, 'm')
Aray_response_visualization(stim, -LFP_dataset, Probe_set, plxarray, v_min, v_max, 'LFP_array_response_medium.png')

# LFP response for narrow window
LFP_dataset = LFP_Response_dataset(LFP, 'n')
Aray_response_visualization(stim, -LFP_dataset, Probe_set, plxarray, v_min, v_max, 'LFP_array_response_narrow.png')

######### Receptive fields

MUA_dataset = MUA_Response_dataset(MUA, 'w')
LFP_dataset = LFP_Response_dataset(LFP, 'w')
fig = plt.figure(figsize=(12, 12))
RF_mean_diameter_MUA_wide = population_contours(MUA_dataset, stim, electrode_set, 4, 'r')
RF_mean_diameter_LFP_wide = population_contours(-LFP_dataset, stim, electrode_set, 4, 'b')
plt.show()
fig.savefig('RF_contours_wide.png', dpi=150)
RF_center_position_MUA_wide = RFCenterPosition(MUA_dataset, stim, electrode_set, 4)
RF_center_eccentricity_MUA_wide = RFCenterEccentricity(RF_center_position_MUA_wide)
RF_center_position_LFP_wide = RFCenterPosition(-LFP_dataset, stim, electrode_set, 4)
RF_center_eccentricity_LFP_wide = RFCenterEccentricity(RF_center_position_LFP_wide)

RFDiameterEccentricityPlot(RF_center_eccentricity_MUA_wide, RF_mean_diameter_MUA_wide, 'r', 'MUA Wide',
                           'RF_diameter_ecc_MUA_wide')
RFDiameterEccentricityPlot(RF_center_eccentricity_LFP_wide, RF_mean_diameter_LFP_wide, 'b', 'LFP Wide',
                           'RF_diameter_ecc_LFP_wide')

ECConArray(RF_center_eccentricity_MUA_wide, electrode_set, plxarray, 'RF_center_eccentricity_MUA_wide')
ECConArray(RF_center_eccentricity_LFP_wide, electrode_set, plxarray, 'RF_center_eccentricity_LFP_wide')

MUA_dataset = MUA_Response_dataset(MUA, 'm')
LFP_dataset = LFP_Response_dataset(LFP, 'm')
fig = plt.figure(figsize=(12, 12))
RF_mean_diameter_MUA_medium = population_contours(MUA_dataset, stim, electrode_set, 4, 'r')
RF_mean_diameter_LFP_medium = population_contours(-LFP_dataset, stim, electrode_set, 4, 'b')
plt.show()
fig.savefig('RF_contours_medium.png', dpi=150)
RF_center_position_MUA_medium = RFCenterPosition(MUA_dataset, stim, electrode_set, 4)
RF_center_eccentricity_MUA_medium = RFCenterEccentricity(RF_center_position_MUA_medium)
RF_center_position_LFP_medium = RFCenterPosition(-LFP_dataset, stim, electrode_set, 4)
RF_center_eccentricity_LFP_medium = RFCenterEccentricity(RF_center_position_LFP_medium)

RFDiameterEccentricityPlot(RF_center_eccentricity_MUA_medium, RF_mean_diameter_MUA_medium, 'r', 'MUA Medium',
                           'RF_diameter_ecc_MUA_medium')
RFDiameterEccentricityPlot(RF_center_eccentricity_LFP_medium, RF_mean_diameter_LFP_medium, 'b', 'LFP Medium',
                           'RF_diameter_ecc_LFP_medium')

ECConArray(RF_center_eccentricity_MUA_medium, electrode_set, plxarray, 'RF_center_eccentricity_MUA_medium')
ECConArray(RF_center_eccentricity_LFP_medium, electrode_set, plxarray, 'RF_center_eccentricity_LFP_medium')

MUA_dataset = MUA_Response_dataset(MUA, 'n')
LFP_dataset = LFP_Response_dataset(LFP, 'n')
fig = plt.figure(figsize=(12, 12))
RF_mean_diameter_MUA_narrow = population_contours(MUA_dataset, stim, electrode_set, 4, 'r')
RF_mean_diameter_LFP_narrow = population_contours(-LFP_dataset, stim, electrode_set, 4, 'b')
plt.show()
fig.savefig('RF_contours_narrow.png', dpi=150)
RF_center_position_MUA_narrow = RFCenterPosition(MUA_dataset, stim, electrode_set, 4)
RF_center_eccentricity_MUA_narrow = RFCenterEccentricity(RF_center_position_MUA_narrow)
RF_center_position_LFP_narrow = RFCenterPosition(-LFP_dataset, stim, electrode_set, 4)
RF_center_eccentricity_LFP_narrow = RFCenterEccentricity(RF_center_position_LFP_narrow)

RFDiameterEccentricityPlot(RF_center_eccentricity_MUA_narrow, RF_mean_diameter_MUA_narrow, 'r', 'MUA Narrow',
                           'RF_diameter_ecc_MUA_narrow')
RFDiameterEccentricityPlot(RF_center_eccentricity_LFP_narrow, RF_mean_diameter_LFP_narrow, 'b', 'LFP Narrow',
                           'RF_diameter_ecc_LFP_narrow')

ECConArray(RF_center_eccentricity_MUA_narrow, electrode_set, plxarray, 'RF_center_eccentricity_MUA_narrow')
ECConArray(RF_center_eccentricity_LFP_narrow, electrode_set, plxarray, 'RF_center_eccentricity_LFP_narrow')

######### Responsive electrodes

MUA_dataset = MUA_Response_dataset(MUA, 'w')
MUA_responsive_electrodes_wide = Select_responsive_electrodes(MUA_dataset, stim)
MUA_dataset = MUA_Response_dataset(MUA, 'm')
MUA_responsive_electrodes_medium = Select_responsive_electrodes(MUA_dataset, stim)
MUA_dataset = MUA_Response_dataset(MUA, 'n')
MUA_responsive_electrodes_narrow = Select_responsive_electrodes(MUA_dataset, stim)

LFP_dataset = LFP_Response_dataset(LFP, 'w')
LFP_responsive_electrodes_wide = Select_responsive_electrodes(LFP_dataset, stim)
LFP_dataset = LFP_Response_dataset(LFP, 'm')
LFP_responsive_electrodes_medium = Select_responsive_electrodes(LFP_dataset, stim)
LFP_dataset = LFP_Response_dataset(LFP, 'n')
LFP_responsive_electrodes_narrow = Select_responsive_electrodes(LFP_dataset, stim)

######### Responsive probes

MUA_dataset = MUA_Response_dataset(MUA, 'w')
MUA_responsive_probes_wide = responsive_probes(MUA_dataset[:, MUA_responsive_electrodes_wide], stim)
MUA_dataset = MUA_Response_dataset(MUA, 'm')
MUA_responsive_probes_medium = responsive_probes(MUA_dataset[:, MUA_responsive_electrodes_medium], stim)
MUA_dataset = MUA_Response_dataset(MUA, 'n')
MUA_responsive_probes_narrow = responsive_probes(MUA_dataset[:, MUA_responsive_electrodes_narrow], stim)

LFP_dataset = LFP_Response_dataset(LFP, 'w')
LFP_responsive_probes_wide = responsive_probes(-LFP_dataset[:, LFP_responsive_electrodes_wide], stim)
LFP_dataset = LFP_Response_dataset(LFP, 'm')
LFP_responsive_probes_medium = responsive_probes(-LFP_dataset[:, LFP_responsive_electrodes_medium], stim)
LFP_dataset = LFP_Response_dataset(LFP, 'n')
LFP_responsive_probes_narrow = responsive_probes(-LFP_dataset[:, LFP_responsive_electrodes_narrow], stim)

######### Pairs of responsive probes

MUA_probe_pairs_wide = list(itertools.permutations(list(MUA_responsive_probes_wide), 2))
MUA_probe_pairs_wide = np.array(MUA_probe_pairs_wide, dtype=int)
MUA_probe_pairs_medium = list(itertools.permutations(list(MUA_responsive_probes_medium), 2))
MUA_probe_pairs_medium = np.array(MUA_probe_pairs_medium, dtype=int)
MUA_probe_pairs_narrow = list(itertools.permutations(list(MUA_responsive_probes_narrow), 2))
MUA_probe_pairs_narrow = np.array(MUA_probe_pairs_narrow, dtype=int)

LFP_probe_pairs_wide = list(itertools.permutations(list(LFP_responsive_probes_wide), 2))
LFP_probe_pairs_wide = np.array(LFP_probe_pairs_wide, dtype=int)
LFP_probe_pairs_medium = list(itertools.permutations(list(LFP_responsive_probes_medium), 2))
LFP_probe_pairs_medium = np.array(LFP_probe_pairs_medium, dtype=int)
LFP_probe_pairs_narrow = list(itertools.permutations(list(LFP_responsive_probes_narrow), 2))
LFP_probe_pairs_narrow = np.array(LFP_probe_pairs_narrow, dtype=int)

######### Distances of responsive probes pairs

MUA_distance_array_wide = probe_distances_array(MUA_probe_pairs_wide)
MUA_distance_array_medium = probe_distances_array(MUA_probe_pairs_medium)
MUA_distance_array_narrow = probe_distances_array(MUA_probe_pairs_narrow)

LFP_distance_array_wide = probe_distances_array(LFP_probe_pairs_wide)
LFP_distance_array_medium = probe_distances_array(LFP_probe_pairs_medium)
LFP_distance_array_narrow = probe_distances_array(LFP_probe_pairs_narrow)

######### Pairs of responsive probes with <15 distances

MUA_probe_pairs_wide = MUA_probe_pairs_wide[MUA_distance_array_wide < 15]
MUA_probe_pairs_medium = MUA_probe_pairs_medium[MUA_distance_array_medium < 15]
MUA_probe_pairs_narrow = MUA_probe_pairs_narrow[MUA_distance_array_narrow < 15]

LFP_probe_pairs_wide = LFP_probe_pairs_wide[LFP_distance_array_wide < 15]
LFP_probe_pairs_medium = LFP_probe_pairs_medium[LFP_distance_array_medium < 15]
LFP_probe_pairs_narrow = LFP_probe_pairs_narrow[LFP_distance_array_narrow < 15]

######### <15 distances

MUA_distance_array_wide = MUA_distance_array_wide[MUA_distance_array_wide < 15]
MUA_distance_array_medium = MUA_distance_array_medium[MUA_distance_array_medium < 15]
MUA_distance_array_narrow = MUA_distance_array_narrow[MUA_distance_array_narrow < 15]

LFP_distance_array_wide = LFP_distance_array_wide[LFP_distance_array_wide < 15]
LFP_distance_array_medium = LFP_distance_array_medium[LFP_distance_array_medium < 15]
LFP_distance_array_narrow = LFP_distance_array_narrow[LFP_distance_array_narrow < 15]

######### Mean eccentricity, magnification factor, and cortical distances

MUA_ecc_pairs_wide = Pair_eccentricity(MUA_probe_pairs_wide)
MUA_mgf_pairs_wide = Magnification_factor(MUA_ecc_pairs_wide)
MUA_cortical_distances_wide = cortical_distance(MUA_distance_array_wide, MUA_mgf_pairs_wide)

MUA_ecc_pairs_medium = Pair_eccentricity(MUA_probe_pairs_medium)
MUA_mgf_pairs_medium = Magnification_factor(MUA_ecc_pairs_medium)
MUA_cortical_distances_medium = cortical_distance(MUA_distance_array_medium, MUA_mgf_pairs_medium)

MUA_ecc_pairs_narrow = Pair_eccentricity(MUA_probe_pairs_narrow)
MUA_mgf_pairs_narrow = Magnification_factor(MUA_ecc_pairs_narrow)
MUA_cortical_distances_narrow = cortical_distance(MUA_distance_array_narrow, MUA_mgf_pairs_narrow)

LFP_ecc_pairs_wide = Pair_eccentricity(LFP_probe_pairs_wide)
LFP_mgf_pairs_wide = Magnification_factor(LFP_ecc_pairs_wide)
LFP_cortical_distances_wide = cortical_distance(LFP_distance_array_wide, LFP_mgf_pairs_wide)

LFP_ecc_pairs_medium = Pair_eccentricity(LFP_probe_pairs_medium)
LFP_mgf_pairs_medium = Magnification_factor(LFP_ecc_pairs_medium)
LFP_cortical_distances_medium = cortical_distance(LFP_distance_array_medium, LFP_mgf_pairs_medium)

LFP_ecc_pairs_narrow = Pair_eccentricity(LFP_probe_pairs_narrow)
LFP_mgf_pairs_narrow = Magnification_factor(LFP_ecc_pairs_narrow)
LFP_cortical_distances_narrow = cortical_distance(LFP_distance_array_narrow, LFP_mgf_pairs_narrow)

######### Discrimination of responsive probe pairs with <15 distance

MUA_dataset = MUA_Response_dataset(MUA, 'w')
MUA_performance_array_wide = Discrimination_performance_array(MUA_probe_pairs_wide, stim,
                                                              MUA_dataset[:, MUA_responsive_electrodes_wide],
                                                              'Linear SVM')
MUA_dataset = MUA_Response_dataset(MUA, 'm')
MUA_performance_array_medium = Discrimination_performance_array(MUA_probe_pairs_medium, stim,
                                                                MUA_dataset[:, MUA_responsive_electrodes_medium],
                                                                'Linear SVM')
MUA_dataset = MUA_Response_dataset(MUA, 'n')
MUA_performance_array_narrow = Discrimination_performance_array(MUA_probe_pairs_narrow, stim,
                                                                MUA_dataset[:, MUA_responsive_electrodes_narrow],
                                                                'Linear SVM')

LFP_dataset = LFP_Response_dataset(LFP, 'w')
LFP_performance_array_wide = Discrimination_performance_array(LFP_probe_pairs_wide, stim,
                                                              LFP_dataset[:, LFP_responsive_electrodes_wide],
                                                              'Linear SVM')
LFP_dataset = LFP_Response_dataset(LFP, 'm')
LFP_performance_array_medium = Discrimination_performance_array(LFP_probe_pairs_medium, stim,
                                                                LFP_dataset[:, LFP_responsive_electrodes_medium],
                                                                'Linear SVM')
LFP_dataset = LFP_Response_dataset(LFP, 'n')
LFP_performance_array_narrow = Discrimination_performance_array(LFP_probe_pairs_narrow, stim,
                                                                LFP_dataset[:, LFP_responsive_electrodes_narrow],
                                                                'Linear SVM')

######### Plot discrimination performance vs cortical distance

performance_corticaldistance_plotter(MUA_cortical_distances_wide, MUA_cortical_distances_medium,
                                     MUA_cortical_distances_narrow, MUA_performance_array_wide,
                                     MUA_performance_array_medium, MUA_performance_array_narrow, (0.435, 0.043, 0.043),
                                     (0.913, 0.145, 0.145), (0.960, 0.639, 0.639), 'MUA',
                                     'MUA_performance_corticaldist.png')
performance_corticaldistance_plotter(LFP_cortical_distances_wide, LFP_cortical_distances_medium,
                                     LFP_cortical_distances_narrow, LFP_performance_array_wide,
                                     LFP_performance_array_medium, LFP_performance_array_narrow, (0.047, 0.062, 0.372),
                                     (0.145, 0.188, 0.894), (0.611, 0.631, 0.949), 'LFP',
                                     'LFP_performance_corticaldist.png')

######### Plot discrimination performance vs probes mean eccentricity

separation_dist = 4
Performance_eccentricity_plotter(MUA_distance_array_wide, MUA_distance_array_medium, MUA_distance_array_narrow,
                                 MUA_ecc_pairs_wide, MUA_ecc_pairs_medium, MUA_ecc_pairs_narrow,
                                 MUA_performance_array_wide, MUA_performance_array_medium, MUA_performance_array_narrow,
                                 separation_dist, (0.435, 0.043, 0.043), (0.913, 0.145, 0.145), (0.960, 0.639, 0.639),
                                 'MUA 4 deg separation', 'Performance_Eccentricity_MUA_four.png')
Performance_eccentricity_plotter(LFP_distance_array_wide, LFP_distance_array_medium, LFP_distance_array_narrow,
                                 LFP_ecc_pairs_wide, LFP_ecc_pairs_medium, LFP_ecc_pairs_narrow,
                                 LFP_performance_array_wide, LFP_performance_array_medium, LFP_performance_array_narrow,
                                 separation_dist, (0.047, 0.062, 0.372), (0.145, 0.188, 0.894), (0.611, 0.631, 0.949),
                                 'LFP 4 deg separation', 'Performance_Eccentricity_LFP_four.png')

separation_dist = 5.656854249492381
Performance_eccentricity_plotter(MUA_distance_array_wide, MUA_distance_array_medium, MUA_distance_array_narrow,
                                 MUA_ecc_pairs_wide, MUA_ecc_pairs_medium, MUA_ecc_pairs_narrow,
                                 MUA_performance_array_wide, MUA_performance_array_medium, MUA_performance_array_narrow,
                                 separation_dist, (0.435, 0.043, 0.043), (0.913, 0.145, 0.145), (0.960, 0.639, 0.639),
                                 'MUA 5.6 deg separation', 'Performance_Eccentricity_MUA_five.png')
Performance_eccentricity_plotter(LFP_distance_array_wide, LFP_distance_array_medium, LFP_distance_array_narrow,
                                 LFP_ecc_pairs_wide, LFP_ecc_pairs_medium, LFP_ecc_pairs_narrow,
                                 LFP_performance_array_wide, LFP_performance_array_medium, LFP_performance_array_narrow,
                                 separation_dist, (0.047, 0.062, 0.372), (0.145, 0.188, 0.894), (0.611, 0.631, 0.949),
                                 'LFP 5.6 deg separation', 'Performance_Eccentricity_LFP_five.png')

separation_dist = 8
Performance_eccentricity_plotter(MUA_distance_array_wide, MUA_distance_array_medium, MUA_distance_array_narrow,
                                 MUA_ecc_pairs_wide, MUA_ecc_pairs_medium, MUA_ecc_pairs_narrow,
                                 MUA_performance_array_wide, MUA_performance_array_medium, MUA_performance_array_narrow,
                                 separation_dist, (0.435, 0.043, 0.043), (0.913, 0.145, 0.145), (0.960, 0.639, 0.639),
                                 'MUA 8 deg separation', 'Performance_Eccentricity_MUA_eight.png')
Performance_eccentricity_plotter(LFP_distance_array_wide, LFP_distance_array_medium, LFP_distance_array_narrow,
                                 LFP_ecc_pairs_wide, LFP_ecc_pairs_medium, LFP_ecc_pairs_narrow,
                                 LFP_performance_array_wide, LFP_performance_array_medium, LFP_performance_array_narrow,
                                 separation_dist, (0.047, 0.062, 0.372), (0.145, 0.188, 0.894), (0.611, 0.631, 0.949),
                                 'LFP 8 deg separation', 'Performance_Eccentricity_LFP_eight.png')

######### Clustering similar RFs to select the best electrodes for discrimination and to minimize the number of electrodes

MUA_dataset = MUA_Response_dataset(MUA, 'w')  # Wide window for MUA gave the best results
LFP_dataset = LFP_Response_dataset(LFP, 'm')  # Medium window for LFP gave the best results

no_cluster = 3
cat_no = 2
MUA_cat = RF_cluster(stim, MUA_dataset, MUA_responsive_electrodes_wide, no_cluster, cat_no)

fig = plt.figure(figsize=(12, 12))
population_contours(MUA_dataset, stim, MUA_cat, 4, 'r')
plt.show()
fig.savefig('MUA_Clustered_RF_contours2.png', dpi=150)

MUA_three_highest_peak = RF_representative_electrode(MUA_cat, stim, MUA_dataset)
MUA_clustering_selected_electrodes = np.array([49, 19, 18, 58, 10, 45, 54, 72])
RF_set_viewer(stim, MUA_dataset, MUA_clustering_selected_electrodes, -0.27, 0.27, 'sure')
MUA_performance_array_clustering_selected = Discrimination_performance_array(MUA_probe_pairs_wide, stim, MUA_dataset[:,
                                                                                                         MUA_clustering_selected_electrodes],
                                                                             'Linear SVM')

performance_corticaldistance_plotter_clustering(MUA_cortical_distances_wide, MUA_performance_array_wide,
                                                MUA_performance_array_clustering_selected, (0.435, 0.043, 0.043),
                                                (0.952, 0.568, 0.349), 'MUA',
                                                'MUA_performance_corticaldist_clustering.png')
separation_dist = 4
Performance_eccentricity_plotter_clustering(MUA_distance_array_wide, MUA_ecc_pairs_wide, MUA_performance_array_wide,
                                            MUA_performance_array_clustering_selected, separation_dist,
                                            (0.435, 0.043, 0.043), (0.952, 0.568, 0.349), 'MUA 4 deg separation',
                                            'Performance_Eccentricity_MUA_four_clustering.png')
separation_dist = 5.656854249492381
Performance_eccentricity_plotter_clustering(MUA_distance_array_wide, MUA_ecc_pairs_wide, MUA_performance_array_wide,
                                            MUA_performance_array_clustering_selected, separation_dist,
                                            (0.435, 0.043, 0.043), (0.952, 0.568, 0.349), 'MUA 5.6 deg separation',
                                            'Performance_Eccentricity_MUA_five_clustering.png')
separation_dist = 8
Performance_eccentricity_plotter_clustering(MUA_distance_array_wide, MUA_ecc_pairs_wide, MUA_performance_array_wide,
                                            MUA_performance_array_clustering_selected, separation_dist,
                                            (0.435, 0.043, 0.043), (0.952, 0.568, 0.349), 'MUA 8 deg separation',
                                            'Performance_Eccentricity_MUA_eight_clustering.png')

no_cluster = 3
cat_no = 2
LFP_cat = RF_cluster(stim, -LFP_dataset, LFP_responsive_electrodes_medium, no_cluster, cat_no)

fig = plt.figure(figsize=(12, 12))
population_contours(-LFP_dataset, stim, LFP_cat, 4, 'b')
plt.show()
fig.savefig('LFP_Clustered_RF_contours2.png', dpi=150)

LFP_three_highest_peak = RF_representative_electrode(LFP_cat, stim, -LFP_dataset)
LFP_clustering_selected_electrodes = np.array([95, 79, 50, 19, 20, 18, 69, 68, 85])
RF_set_viewer(stim, -LFP_dataset, LFP_clustering_selected_electrodes, -0.34, 0.34, 'sure')
LFP_performance_array_clustering_selected = Discrimination_performance_array(LFP_probe_pairs_medium, stim,
                                                                             LFP_dataset[:,
                                                                             LFP_clustering_selected_electrodes],
                                                                             'Linear SVM')

performance_corticaldistance_plotter_clustering(LFP_cortical_distances_medium, LFP_performance_array_medium,
                                                LFP_performance_array_clustering_selected, (0.145, 0.188, 0.894),
                                                (0.635, 0.349, 0.952), 'LFP',
                                                'LFP_performance_corticaldist_clustering.png')
separation_dist = 4
Performance_eccentricity_plotter_clustering(LFP_distance_array_medium, LFP_ecc_pairs_medium,
                                            LFP_performance_array_medium, LFP_performance_array_clustering_selected,
                                            separation_dist, (0.145, 0.188, 0.894), (0.635, 0.349, 0.952),
                                            'LFP 4 deg separation', 'Performance_Eccentricity_LFP_four_clustering.png')
separation_dist = 5.656854249492381
Performance_eccentricity_plotter_clustering(LFP_distance_array_medium, LFP_ecc_pairs_medium,
                                            LFP_performance_array_medium, LFP_performance_array_clustering_selected,
                                            separation_dist, (0.145, 0.188, 0.894), (0.635, 0.349, 0.952),
                                            'LFP 5.6 deg separation',
                                            'Performance_Eccentricity_LFP_five_clustering.png')
separation_dist = 8
Performance_eccentricity_plotter_clustering(LFP_distance_array_medium, LFP_ecc_pairs_medium,
                                            LFP_performance_array_medium, LFP_performance_array_clustering_selected,
                                            separation_dist, (0.145, 0.188, 0.894), (0.635, 0.349, 0.952),
                                            'LFP 8 deg separation', 'Performance_Eccentricity_LFP_eight_clustering.png')

######### Electrodes importance

MUA_weight_set = SVM_weight_set(MUA_probe_pairs_wide, stim, MUA_dataset[:, MUA_responsive_electrodes_wide])
MUA_weightsorted_electrodes = electrodes_sorting_by_weight(MUA_weight_set, MUA_responsive_electrodes_wide)
MUA_performance_array_importance_four_selected = Discrimination_performance_array(MUA_probe_pairs_wide, stim,
                                                                                  MUA_dataset[:,
                                                                                  MUA_weightsorted_electrodes[0:4]],
                                                                                  'Linear SVM')
MUA_performance_array_importance_six_selected = Discrimination_performance_array(MUA_probe_pairs_wide, stim,
                                                                                 MUA_dataset[:,
                                                                                 MUA_weightsorted_electrodes[0:6]],
                                                                                 'Linear SVM')
MUA_performance_array_importance_eight_selected = Discrimination_performance_array(MUA_probe_pairs_wide, stim,
                                                                                   MUA_dataset[:,
                                                                                   MUA_weightsorted_electrodes[0:8]],
                                                                                   'Linear SVM')
MUA_performance_array_importance_ten_selected = Discrimination_performance_array(MUA_probe_pairs_wide, stim,
                                                                                 MUA_dataset[:,
                                                                                 MUA_weightsorted_electrodes[0:10]],
                                                                                 'Linear SVM')
MUA_performance_array_importance_twelve_selected = Discrimination_performance_array(MUA_probe_pairs_wide, stim,
                                                                                    MUA_dataset[:,
                                                                                    MUA_weightsorted_electrodes[0:12]],
                                                                                    'Linear SVM')

performance_corticaldistance_plotter_importance(MUA_cortical_distances_wide, MUA_performance_array_wide,
                                                MUA_performance_array_importance_four_selected,
                                                MUA_performance_array_importance_six_selected,
                                                MUA_performance_array_importance_eight_selected,
                                                MUA_performance_array_importance_ten_selected,
                                                MUA_performance_array_importance_twelve_selected,
                                                'MUA_performance_corticaldist_importance_selected.png')

separation_dist = 4
Performance_eccentricity_plotter_importance(MUA_distance_array_wide, MUA_ecc_pairs_wide, MUA_performance_array_wide,
                                            MUA_performance_array_importance_four_selected,
                                            MUA_performance_array_importance_six_selected,
                                            MUA_performance_array_importance_eight_selected,
                                            MUA_performance_array_importance_ten_selected,
                                            MUA_performance_array_importance_twelve_selected, separation_dist,
                                            '4 degree separation',
                                            'Performance_Eccentricity_MUA_importance_four_deg.png')
separation_dist = 8
Performance_eccentricity_plotter_importance(MUA_distance_array_wide, MUA_ecc_pairs_wide, MUA_performance_array_wide,
                                            MUA_performance_array_importance_four_selected,
                                            MUA_performance_array_importance_six_selected,
                                            MUA_performance_array_importance_eight_selected,
                                            MUA_performance_array_importance_ten_selected,
                                            MUA_performance_array_importance_twelve_selected, separation_dist,
                                            '8 degree separation',
                                            'Performance_Eccentricity_MUA_importance_eight_deg.png')

LFP_weight_set = SVM_weight_set(LFP_probe_pairs_medium, stim, LFP_dataset[:, LFP_responsive_electrodes_medium])
LFP_weightsorted_electrodes = electrodes_sorting_by_weight(LFP_weight_set, LFP_responsive_electrodes_medium)
LFP_performance_array_importance_four_selected = Discrimination_performance_array(LFP_probe_pairs_medium, stim,
                                                                                  LFP_dataset[:,
                                                                                  LFP_weightsorted_electrodes[0:4]],
                                                                                  'Linear SVM')
LFP_performance_array_importance_six_selected = Discrimination_performance_array(LFP_probe_pairs_medium, stim,
                                                                                 LFP_dataset[:,
                                                                                 LFP_weightsorted_electrodes[0:6]],
                                                                                 'Linear SVM')
LFP_performance_array_importance_eight_selected = Discrimination_performance_array(LFP_probe_pairs_medium, stim,
                                                                                   LFP_dataset[:,
                                                                                   LFP_weightsorted_electrodes[0:8]],
                                                                                   'Linear SVM')
LFP_performance_array_importance_ten_selected = Discrimination_performance_array(LFP_probe_pairs_medium, stim,
                                                                                 LFP_dataset[:,
                                                                                 LFP_weightsorted_electrodes[0:10]],
                                                                                 'Linear SVM')
LFP_performance_array_importance_twelve_selected = Discrimination_performance_array(LFP_probe_pairs_medium, stim,
                                                                                    LFP_dataset[:,
                                                                                    LFP_weightsorted_electrodes[0:12]],
                                                                                    'Linear SVM')

performance_corticaldistance_plotter_importance(LFP_cortical_distances_medium, LFP_performance_array_medium,
                                                LFP_performance_array_importance_four_selected,
                                                LFP_performance_array_importance_six_selected,
                                                LFP_performance_array_importance_eight_selected,
                                                LFP_performance_array_importance_ten_selected,
                                                LFP_performance_array_importance_twelve_selected,
                                                'LFP_performance_corticaldist_importance_selected.png')

separation_dist = 4
Performance_eccentricity_plotter_importance(LFP_distance_array_medium, LFP_ecc_pairs_medium,
                                            LFP_performance_array_medium,
                                            LFP_performance_array_importance_four_selected,
                                            LFP_performance_array_importance_six_selected,
                                            LFP_performance_array_importance_eight_selected,
                                            LFP_performance_array_importance_ten_selected,
                                            LFP_performance_array_importance_twelve_selected, separation_dist,
                                            '4 degree separation',
                                            'Performance_Eccentricity_LFP_importance_four_deg.png')
separation_dist = 8
Performance_eccentricity_plotter_importance(LFP_distance_array_medium, LFP_ecc_pairs_medium,
                                            LFP_performance_array_medium,
                                            LFP_performance_array_importance_four_selected,
                                            LFP_performance_array_importance_six_selected,
                                            LFP_performance_array_importance_eight_selected,
                                            LFP_performance_array_importance_ten_selected,
                                            LFP_performance_array_importance_twelve_selected, separation_dist,
                                            '8 degree separation',
                                            'Performance_Eccentricity_LFP_importance_eight_deg.png')

# When weights of all the discriminations of ==4 deg separation and eccentricity<=8 deg are used

MUA_probe_pairs_separation_four_ecc_eight = MUA_probe_pairs_wide[
                                            (MUA_distance_array_wide == 4) & (MUA_ecc_pairs_wide < 8.1), :]
MUA_weight_set_separation_four_ecc_eight = SVM_weight_set(MUA_probe_pairs_separation_four_ecc_eight, stim,
                                                          MUA_dataset[:, MUA_responsive_electrodes_wide])
MUA_weightsorted_electrodes_separation_four_ecc_eight = electrodes_sorting_by_weight(
    MUA_weight_set_separation_four_ecc_eight, MUA_responsive_electrodes_wide)

MUA_performance_array_importance_four_selected_separation_four_ecc_eight = Discrimination_performance_array(
    MUA_probe_pairs_separation_four_ecc_eight, stim,
    MUA_dataset[:, MUA_weightsorted_electrodes_separation_four_ecc_eight[0:4]], 'Linear SVM')
MUA_performance_array_importance_six_selected_separation_four_ecc_eight = Discrimination_performance_array(
    MUA_probe_pairs_separation_four_ecc_eight, stim,
    MUA_dataset[:, MUA_weightsorted_electrodes_separation_four_ecc_eight[0:6]], 'Linear SVM')
MUA_performance_array_importance_eight_selected_separation_four_ecc_eight = Discrimination_performance_array(
    MUA_probe_pairs_separation_four_ecc_eight, stim,
    MUA_dataset[:, MUA_weightsorted_electrodes_separation_four_ecc_eight[0:8]], 'Linear SVM')
MUA_performance_array_importance_ten_selected_separation_four_ecc_eight = Discrimination_performance_array(
    MUA_probe_pairs_separation_four_ecc_eight, stim,
    MUA_dataset[:, MUA_weightsorted_electrodes_separation_four_ecc_eight[0:10]], 'Linear SVM')
MUA_performance_array_importance_twelve_selected_separation_four_ecc_eight = Discrimination_performance_array(
    MUA_probe_pairs_separation_four_ecc_eight, stim,
    MUA_dataset[:, MUA_weightsorted_electrodes_separation_four_ecc_eight[0:12]], 'Linear SVM')

separation_dist = 4
Performance_eccentricity_plotter_importance(
    MUA_distance_array_wide[(MUA_distance_array_wide == 4) & (MUA_ecc_pairs_wide < 8.1)],
    MUA_ecc_pairs_wide[(MUA_distance_array_wide == 4) & (MUA_ecc_pairs_wide < 8.1)],
    MUA_performance_array_wide[(MUA_distance_array_wide == 4) & (MUA_ecc_pairs_wide < 8.1)],
    MUA_performance_array_importance_four_selected_separation_four_ecc_eight,
    MUA_performance_array_importance_six_selected_separation_four_ecc_eight,
    MUA_performance_array_importance_eight_selected_separation_four_ecc_eight,
    MUA_performance_array_importance_ten_selected_separation_four_ecc_eight,
    MUA_performance_array_importance_twelve_selected_separation_four_ecc_eight, separation_dist, '4 degree separation',
    'Performance_Eccentricity_MUA_importance_four_deg_separation_four_ecc_eight.png')

Electrode_array_show(MUA_weightsorted_electrodes_separation_four_ecc_eight[0:6], plxarray, 'MUA_enough_electrodes')

LFP_probe_pairs_separation_four_ecc_eight = LFP_probe_pairs_medium[
                                            (LFP_distance_array_medium == 4) & (LFP_ecc_pairs_medium < 8.1), :]
LFP_weight_set_separation_four_ecc_eight = SVM_weight_set(LFP_probe_pairs_separation_four_ecc_eight, stim,
                                                          LFP_dataset[:, LFP_responsive_electrodes_medium])
LFP_weightsorted_electrodes_separation_four_ecc_eight = electrodes_sorting_by_weight(
    LFP_weight_set_separation_four_ecc_eight, LFP_responsive_electrodes_medium)

LFP_performance_array_importance_four_selected_separation_four_ecc_eight = Discrimination_performance_array(
    LFP_probe_pairs_separation_four_ecc_eight, stim,
    LFP_dataset[:, LFP_weightsorted_electrodes_separation_four_ecc_eight[0:4]], 'Linear SVM')
LFP_performance_array_importance_six_selected_separation_four_ecc_eight = Discrimination_performance_array(
    LFP_probe_pairs_separation_four_ecc_eight, stim,
    LFP_dataset[:, LFP_weightsorted_electrodes_separation_four_ecc_eight[0:6]], 'Linear SVM')
LFP_performance_array_importance_eight_selected_separation_four_ecc_eight = Discrimination_performance_array(
    LFP_probe_pairs_separation_four_ecc_eight, stim,
    LFP_dataset[:, LFP_weightsorted_electrodes_separation_four_ecc_eight[0:8]], 'Linear SVM')
LFP_performance_array_importance_ten_selected_separation_four_ecc_eight = Discrimination_performance_array(
    LFP_probe_pairs_separation_four_ecc_eight, stim,
    LFP_dataset[:, LFP_weightsorted_electrodes_separation_four_ecc_eight[0:10]], 'Linear SVM')
LFP_performance_array_importance_twelve_selected_separation_four_ecc_eight = Discrimination_performance_array(
    LFP_probe_pairs_separation_four_ecc_eight, stim,
    LFP_dataset[:, LFP_weightsorted_electrodes_separation_four_ecc_eight[0:12]], 'Linear SVM')

separation_dist = 4
Performance_eccentricity_plotter_importance(
    LFP_distance_array_medium[(LFP_distance_array_medium == 4) & (LFP_ecc_pairs_medium < 8.1)],
    LFP_ecc_pairs_medium[(LFP_distance_array_medium == 4) & (LFP_ecc_pairs_medium < 8.1)],
    LFP_performance_array_medium[(LFP_distance_array_medium == 4) & (LFP_ecc_pairs_medium < 8.1)],
    LFP_performance_array_importance_four_selected_separation_four_ecc_eight,
    LFP_performance_array_importance_six_selected_separation_four_ecc_eight,
    LFP_performance_array_importance_eight_selected_separation_four_ecc_eight,
    LFP_performance_array_importance_ten_selected_separation_four_ecc_eight,
    LFP_performance_array_importance_twelve_selected_separation_four_ecc_eight, separation_dist, '4 degree separation',
    'Performance_Eccentricity_LFP_importance_four_deg_separation_four_ecc_eight.png')

Electrode_array_show(LFP_weightsorted_electrodes_separation_four_ecc_eight[0:16], plxarray, 'LFP_enough_electrodes')

######### Distribution of weights over the electrode array

# Histogram of importance values for MUA and LFP

Electrodes_importance_histogram(MUA_weight_set, 'r', 'MUA_importance_histogram.png')
Electrodes_importance_histogram(LFP_weight_set, 'b', 'LFP_importance_histogram.png')

# Distribution of importance values over electrode array

v_min, v_max = 0.0189, 0.0525
Electrode_array_importance(MUA_weight_set, MUA_responsive_electrodes_wide, plxarray, v_min, v_max,
                           'MUA_importance_on_array.png')
v_min, v_max = 0.0074, 0.0156
Electrode_array_importance(LFP_weight_set, LFP_responsive_electrodes_medium, plxarray, v_min, v_max,
                           'LFP_importance_on_array.png')

######### Comparison of tuning curve and weight field of an electrode

electrode = 10

separation_dist = 4
probes_weight(MUA_weight_set, electrode, MUA_responsive_electrodes_wide, MUA_probe_pairs_wide, separation_dist,
              MUA_distance_array_wide, 'MUA_weight_field_elec10_separation4.png')
separation_dist = 8
probes_weight(MUA_weight_set, electrode, MUA_responsive_electrodes_wide, MUA_probe_pairs_wide, separation_dist,
              MUA_distance_array_wide, 'MUA_weight_field_elec10_separation8.png')
separation_dist = 12
probes_weight(MUA_weight_set, electrode, MUA_responsive_electrodes_wide, MUA_probe_pairs_wide, separation_dist,
              MUA_distance_array_wide, 'MUA_weight_field_elec10_separation12.png')

tuning_plotter(stim, MUA_dataset, electrode, -0.27, 0.27, 'MUA_tuning_curve_elec10.png')

electrode = 58
separation_dist = 4
probes_weight(MUA_weight_set, electrode, MUA_responsive_electrodes_wide, MUA_probe_pairs_wide, separation_dist,
              MUA_distance_array_wide, 'MUA_weight_field_elec58_separation4.png')
separation_dist = 8
probes_weight(MUA_weight_set, electrode, MUA_responsive_electrodes_wide, MUA_probe_pairs_wide, separation_dist,
              MUA_distance_array_wide, 'MUA_weight_field_elec58_separation8.png')
separation_dist = 12
probes_weight(MUA_weight_set, electrode, MUA_responsive_electrodes_wide, MUA_probe_pairs_wide, separation_dist,
              MUA_distance_array_wide, 'MUA_weight_field_elec58_separation12.png')

tuning_plotter(stim, MUA_dataset, electrode, -0.27, 0.27, 'MUA_tuning_curve_elec58.png')

electrode = 19
separation_dist = 4
probes_weight(MUA_weight_set, electrode, MUA_responsive_electrodes_wide, MUA_probe_pairs_wide, separation_dist,
              MUA_distance_array_wide, 'MUA_weight_field_elec19_separation4.png')
separation_dist = 8
probes_weight(MUA_weight_set, electrode, MUA_responsive_electrodes_wide, MUA_probe_pairs_wide, separation_dist,
              MUA_distance_array_wide, 'MUA_weight_field_elec19_separation8.png')
separation_dist = 12
probes_weight(MUA_weight_set, electrode, MUA_responsive_electrodes_wide, MUA_probe_pairs_wide, separation_dist,
              MUA_distance_array_wide, 'MUA_weight_field_elec19_separation12.png')

tuning_plotter(stim, MUA_dataset, electrode, -0.27, 0.27, 'MUA_tuning_curve_elec19.png')

electrode = 2
separation_dist = 4
probes_weight(LFP_weight_set, electrode, LFP_responsive_electrodes_medium, LFP_probe_pairs_medium, separation_dist,
              LFP_distance_array_medium, 'LFP_weight_field_elec2_separation4.png')
separation_dist = 8
probes_weight(LFP_weight_set, electrode, LFP_responsive_electrodes_medium, LFP_probe_pairs_medium, separation_dist,
              LFP_distance_array_medium, 'LFP_weight_field_elec2_separation8.png')
separation_dist = 12
probes_weight(LFP_weight_set, electrode, LFP_responsive_electrodes_medium, LFP_probe_pairs_medium, separation_dist,
              LFP_distance_array_medium, 'LFP_weight_field_elec2_separation12.png')

tuning_plotter(stim, -LFP_dataset, electrode, -0.34, 0.34, 'LFP_tuning_curve_elec2.png')

electrode = 30
separation_dist = 4
probes_weight(LFP_weight_set, electrode, LFP_responsive_electrodes_medium, LFP_probe_pairs_medium, separation_dist,
              LFP_distance_array_medium, 'LFP_weight_field_elec30_separation4.png')
separation_dist = 8
probes_weight(LFP_weight_set, electrode, LFP_responsive_electrodes_medium, LFP_probe_pairs_medium, separation_dist,
              LFP_distance_array_medium, 'LFP_weight_field_elec30_separation8.png')
separation_dist = 12
probes_weight(LFP_weight_set, electrode, LFP_responsive_electrodes_medium, LFP_probe_pairs_medium, separation_dist,
              LFP_distance_array_medium, 'LFP_weight_field_elec30_separation12.png')

tuning_plotter(stim, -LFP_dataset, electrode, -0.34, 0.34, 'LFP_tuning_curve_elec30.png')

electrode = 55
separation_dist = 4
probes_weight(LFP_weight_set, electrode, LFP_responsive_electrodes_medium, LFP_probe_pairs_medium, separation_dist,
              LFP_distance_array_medium, 'LFP_weight_field_elec55_separation4.png')
separation_dist = 8
probes_weight(LFP_weight_set, electrode, LFP_responsive_electrodes_medium, LFP_probe_pairs_medium, separation_dist,
              LFP_distance_array_medium, 'LFP_weight_field_elec55_separation8.png')
separation_dist = 12
probes_weight(LFP_weight_set, electrode, LFP_responsive_electrodes_medium, LFP_probe_pairs_medium, separation_dist,
              LFP_distance_array_medium, 'LFP_weight_field_elec55_separation12.png')

tuning_plotter(stim, -LFP_dataset, electrode, -0.34, 0.34, 'LFP_tuning_curve_elec55.png')

electrode = 48
separation_dist = 4
probes_weight(LFP_weight_set, electrode, LFP_responsive_electrodes_medium, LFP_probe_pairs_medium, separation_dist,
              LFP_distance_array_medium, 'LFP_weight_field_elec48_separation4.png')
separation_dist = 8
probes_weight(LFP_weight_set, electrode, LFP_responsive_electrodes_medium, LFP_probe_pairs_medium, separation_dist,
              LFP_distance_array_medium, 'LFP_weight_field_elec48_separation8.png')
separation_dist = 12
probes_weight(LFP_weight_set, electrode, LFP_responsive_electrodes_medium, LFP_probe_pairs_medium, separation_dist,
              LFP_distance_array_medium, 'LFP_weight_field_elec48_separation12.png')

tuning_plotter(stim, -LFP_dataset, electrode, -0.34, 0.34, 'LFP_tuning_curve_elec48.png')

# Relationship between weights of an electrode and cortical distances

MUA_p_values_weight_corticaldist = weight_corticaldist(MUA_weight_set, MUA_cortical_distances_wide,
                                                       MUA_responsive_electrodes_wide, 19)
LFP_p_values_weight_corticaldist = weight_corticaldist(LFP_weight_set, LFP_cortical_distances_medium,
                                                       LFP_responsive_electrodes_medium, 55)

MUA_p_values_importance_corticaldist = mean_importance_corticaldist(MUA_weight_set, MUA_cortical_distances_wide,
                                                                    MUA_responsive_electrodes_wide, 10, 58, 19, 18)
LFP_p_values_importance_corticaldist = mean_importance_corticaldist(LFP_weight_set, LFP_cortical_distances_medium,
                                                                    LFP_responsive_electrodes_medium, 2, 30, 55, 48)

######### Decoding bandpass LFP

# theta

LFP_theta = bandpass_LFP(LFP, 4, 8)
LFP_dataset_theta = LFP_Response_dataset(LFP_theta, 'm')
LFP_performance_array_theta = Discrimination_performance_array(LFP_probe_pairs_medium, stim,
                                                               LFP_dataset_theta[:, LFP_responsive_electrodes_medium],
                                                               'Linear SVM')

# alpha

LFP_alpha = bandpass_LFP(LFP, 8, 12)
LFP_dataset_alpha = LFP_Response_dataset(LFP_alpha, 'm')
LFP_performance_array_alpha = Discrimination_performance_array(LFP_probe_pairs_medium, stim,
                                                               LFP_dataset_alpha[:, LFP_responsive_electrodes_medium],
                                                               'Linear SVM')

# beta

LFP_beta = bandpass_LFP(LFP, 12, 30)
LFP_dataset_beta = LFP_Response_dataset(LFP_beta, 'm')
LFP_performance_array_beta = Discrimination_performance_array(LFP_probe_pairs_medium, stim,
                                                              LFP_dataset_beta[:, LFP_responsive_electrodes_medium],
                                                              'Linear SVM')

# gamma

LFP_gamma = bandpass_LFP(LFP, 30, 50)
LFP_dataset_gamma = LFP_Response_dataset(LFP_gamma, 'm')
LFP_performance_array_gamma = Discrimination_performance_array(LFP_probe_pairs_medium, stim,
                                                               LFP_dataset_gamma[:, LFP_responsive_electrodes_medium],
                                                               'Linear SVM')

# high gamma

LFP_highgamma = bandpass_LFP(LFP, 50, 80)
LFP_dataset_highgamma = LFP_Response_dataset(LFP_highgamma, 'm')
LFP_performance_array_highgamma = Discrimination_performance_array(LFP_probe_pairs_medium, stim,
                                                                   LFP_dataset_highgamma[:,
                                                                   LFP_responsive_electrodes_medium], 'Linear SVM')

# plot performance vs cortical distance

bandpass_performance_corticaldistance_plotter(LFP_cortical_distances_medium, LFP_performance_array_theta,
                                              LFP_performance_array_alpha, LFP_performance_array_beta,
                                              LFP_performance_array_gamma, LFP_performance_array_highgamma,
                                              'Bandpass Performance', 'Performance_bandpass_LFP.png')

separation_dist = 4
Bandpass_Performance_eccentricity_plotter(LFP_distance_array_medium, LFP_ecc_pairs_medium, LFP_performance_array_theta,
                                          LFP_performance_array_alpha, LFP_performance_array_beta,
                                          LFP_performance_array_gamma, LFP_performance_array_highgamma, separation_dist,
                                          '4 deg separation', 'LFP_Bandpass_Performance_eccentricity_four.png')

separation_dist = 8
Bandpass_Performance_eccentricity_plotter(LFP_distance_array_medium, LFP_ecc_pairs_medium, LFP_performance_array_theta,
                                          LFP_performance_array_alpha, LFP_performance_array_beta,
                                          LFP_performance_array_gamma, LFP_performance_array_highgamma, separation_dist,
                                          '8 deg separation', 'LFP_Bandpass_Performance_eccentricity_eight.png')

######### Correlation-blind performance

# All responsive electrodes

MUA_performance_array_blind = Discrimination_performance_array_onblind(MUA_probe_pairs_wide, stim,
                                                                       MUA_dataset[:, MUA_responsive_electrodes_wide])
LFP_performance_array_blind = Discrimination_performance_array_onblind(LFP_probe_pairs_medium, stim,
                                                                       LFP_dataset[:, LFP_responsive_electrodes_medium])

# Six most important electrodes

MUA_performance_array_importance_six_blind = Discrimination_performance_array_onblind(MUA_probe_pairs_wide, stim,
                                                                                      MUA_dataset[:,
                                                                                      MUA_weightsorted_electrodes[0:6]])
LFP_performance_array_importance_six_blind = Discrimination_performance_array_onblind(LFP_probe_pairs_medium, stim,
                                                                                      LFP_dataset[:,
                                                                                      LFP_weightsorted_electrodes[0:6]])

# Plot discrimination performance vs cortical distance

performance_corticaldistance_onblind_plotter(MUA_cortical_distances_wide, MUA_performance_array_wide,
                                             MUA_performance_array_blind, (0.435, 0.043, 0.043), (0.992, 0.576, 0.070),
                                             'MUA', 'MUA_performance_corticaldist_correlationblind.png')
performance_corticaldistance_onblind_plotter(LFP_cortical_distances_medium, LFP_performance_array_medium,
                                             LFP_performance_array_blind, (0.145, 0.188, 0.894), (0.964, 0.070, 0.992),
                                             'LFP', 'LFP_performance_corticaldist_correlationblind.png')

# Plot discrimination performance vs Eccentricity

separation_dist = 4
Performance_eccentricity_onblind_plotter(MUA_distance_array_wide, MUA_ecc_pairs_wide, MUA_performance_array_wide,
                                         MUA_performance_array_blind, separation_dist, (0.435, 0.043, 0.043),
                                         (0.992, 0.576, 0.070), 'MUA 4 deg separation',
                                         'MUA_performance_eccentricity_correlationblind_fourdeg.png')

separation_dist = 8
Performance_eccentricity_onblind_plotter(MUA_distance_array_wide, MUA_ecc_pairs_wide, MUA_performance_array_wide,
                                         MUA_performance_array_blind, separation_dist, (0.435, 0.043, 0.043),
                                         (0.992, 0.576, 0.070), 'MUA 8 deg separation',
                                         'MUA_performance_eccentricity_correlationblind_eightdeg.png')

separation_dist = 4
Performance_eccentricity_onblind_plotter(LFP_distance_array_medium, LFP_ecc_pairs_medium, LFP_performance_array_medium,
                                         LFP_performance_array_blind, separation_dist, (0.145, 0.188, 0.894),
                                         (0.964, 0.070, 0.992), 'LFP 4 deg separation',
                                         'LFP_performance_eccentricity_correlationblind_fourdeg.png')

separation_dist = 8
Performance_eccentricity_onblind_plotter(LFP_distance_array_medium, LFP_ecc_pairs_medium, LFP_performance_array_medium,
                                         LFP_performance_array_blind, separation_dist, (0.145, 0.188, 0.894),
                                         (0.964, 0.070, 0.992), 'LFP 8 deg separation',
                                         'LFP_performance_eccentricity_correlationblind_eightdeg.png')

# Plot discrimination performance vs cortical distance for six most important electrodes

performance_corticaldistance_onblind_plotter(MUA_cortical_distances_wide, MUA_performance_array_importance_six_selected,
                                             MUA_performance_array_importance_six_blind, (0.435, 0.043, 0.043),
                                             (0.992, 0.576, 0.070), 'MUA six best electrodes',
                                             'MUA_performance_importance_six_corticaldist_correlationblind.png')
performance_corticaldistance_onblind_plotter(LFP_cortical_distances_medium,
                                             LFP_performance_array_importance_six_selected,
                                             LFP_performance_array_importance_six_blind, (0.145, 0.188, 0.894),
                                             (0.964, 0.070, 0.992), 'LFP six best electrodes',
                                             'LFP_performance_importance_six_corticaldist_correlationblind.png')

######### Noise correlation analysis

# Set of responsive electrode pairs

MUA_electrode_pairs = list(itertools.combinations(list(MUA_responsive_electrodes_wide), 2))
MUA_electrode_pairs = np.array(MUA_electrode_pairs, dtype=int)

LFP_electrode_pairs = list(itertools.combinations(list(LFP_responsive_electrodes_medium), 2))
LFP_electrode_pairs = np.array(LFP_electrode_pairs, dtype=int)

# Noise correlation for all the pairs of responsive electrode (row) and all probes (column)

MUA_nc_array = MUA_all_probes_all_pairs_noise_correlation(MUA, stim, MUA_responsive_electrodes_wide,
                                                          MUA_responsive_probes_wide)
MUA_nc_array = np.nan_to_num(MUA_nc_array)
LFP_nc_array = LFP_all_probes_all_pairs_noise_correlation(LFP, stim, LFP_responsive_electrodes_medium,
                                                          LFP_responsive_probes_medium)

# Cortical distances of all pairs of responsive electrodes

MUA_electrodes_distance = all_pairs_electrodes_distances(plxarray, MUA_responsive_electrodes_wide)
LFP_electrodes_distance = all_pairs_electrodes_distances(plxarray, LFP_responsive_electrodes_medium)

# Noise correlation of each probe for the best two electrodes

# MUA: 58, 10 which is pair 118

electrode = 58
tuning_plotter(stim, MUA_dataset, electrode, -0.27, 0.27, 'MUA_tuning_curve_electrode_58.png')
electrode = 10
tuning_plotter(stim, MUA_dataset, electrode, -0.27, 0.27, 'MUA_tuning_curve_electrode_10.png')

MUA_probes_nc = MUA_nc_array[:, 118]
MUA_probes_nc[MUA_probes_nc == 0] = np.nan
Probes_noise_correlation_plotter(MUA_probes_nc, np.nanmean(MUA_probes_nc) - 1.5 * np.nanstd(MUA_probes_nc),
                                 np.nanmean(MUA_probes_nc) + 1.5 * np.nanstd(MUA_probes_nc), 'MUA_probes_nc.png')

# LFP: 10, 48 which is pair 772

electrode = 10
tuning_plotter(stim, -LFP_dataset, electrode, -0.34, 0.34, 'LFP_tuning_curve_electrode_10.png')
electrode = 48
tuning_plotter(stim, -LFP_dataset, electrode, -0.34, 0.34, 'LFP_tuning_curve_electrode_48.png')

LFP_probes_nc = LFP_nc_array[:, 772]
Probes_noise_correlation_plotter(LFP_probes_nc, np.nanmean(LFP_probes_nc) - 1.5 * np.nanstd(LFP_probes_nc),
                                 np.nanmean(LFP_probes_nc) + 1.5 * np.nanstd(LFP_probes_nc), 'LFP_probes_nc.png')

# Noise correlation vs electrodes distances in response to probes at different eccentricities

# Under 5 deg ecc: 91, 81
# between 8 and 12 deg ecc: 72, 83
# between 16 and 20 deg ecc: 53, 75

noisecorrelation_distance_plotter(MUA_nc_array[np.array([91, 81, 72, 83, 53, 75]), :], MUA_electrodes_distance, -0.3,
                                  0.3, 'MUA', 'MUA_Noisecorrelation_electrodes_distance.png')
noisecorrelation_distance_plotter(LFP_nc_array[np.array([91, 81, 72, 83, 53, 75]), :], LFP_electrodes_distance, 0.2,
                                  0.8, 'LFP', 'LFP_Noisecorrelation_electrodes_distance.png')

# Plot noise correlation vs electrodes mean importance

MUA_electrodes_importance = electrodes_importance_calculator(MUA_weight_set)
MUA_pair_mean_importance = electrode_pairs_importance(MUA_electrode_pairs, MUA_electrodes_importance,
                                                      MUA_responsive_electrodes_wide)
noisecorrelation_mean_importance_plotter(MUA_nc_array[np.array([91, 81, 72, 83, 53, 75]), :], MUA_pair_mean_importance,
                                         np.arange(0.02, 0.095, 0.015), np.arange(0.02, 0.08, 0.015) + 0.0075, -0.3,
                                         0.3, 'MUA', 'MUA_Noisecorrelation_electrodes_mean_importance.png')

LFP_electrodes_importance = electrodes_importance_calculator(LFP_weight_set)
LFP_pair_mean_importance = electrode_pairs_importance(LFP_electrode_pairs, LFP_electrodes_importance,
                                                      LFP_responsive_electrodes_medium)
noisecorrelation_mean_importance_plotter(LFP_nc_array[np.array([91, 81, 72, 83, 53, 75]), :], LFP_pair_mean_importance,
                                         np.arange(0.004, 0.024, 0.004), np.arange(0.004, 0.02, 0.004) + 0.002, 0.2,
                                         0.8, 'LFP', 'LFP_Noisecorrelation_electrodes_mean_importance.png')
