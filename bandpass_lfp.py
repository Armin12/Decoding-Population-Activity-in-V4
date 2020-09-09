import numpy as np
import matplotlib.pyplot as plt
from mean_bin import mean_relationship
from scipy.signal import butter, lfilter
from scipy.stats import pearsonr


def bandpass_LFP(arr, lowcut, highcut):
    """Bandpass LFP
    Bandpass LFP between lowcut and highcut frequencies
    """
    fs = 500
    nyq = 0.5 * fs
    order = 4
    low = lowcut / nyq
    high = highcut / nyq
    w_low, w_high = butter(order, [low, high], btype='band')
    arr_f = lfilter(w_low, w_high, arr)
    return arr_f


def bandpass_performance_corticaldistance_plotter(cortical_distances, performance_array_theta, performance_array_alpha,
                                                  performance_array_beta, performance_array_gamma,
                                                  performance_array_highgamma, figure_title, file_name):
    """
    Plot discrimination performance of LFP versus cortical distance at 
    different frequency bands
    """
    fig = plt.figure(figsize=(12, 10))

    y_x_theta, y_x_std_theta = mean_relationship(cortical_distances, performance_array_theta, np.arange(0.5, 5.5, 0.5))
    y_x_alpha, y_x_std_alpha = mean_relationship(cortical_distances, performance_array_alpha, np.arange(0.5, 5.5, 0.5))
    y_x_beta, y_x_std_beta = mean_relationship(cortical_distances, performance_array_beta, np.arange(0.5, 5.5, 0.5))
    y_x_gamma, y_x_std_gamma = mean_relationship(cortical_distances, performance_array_gamma, np.arange(0.5, 5.5, 0.5))
    y_x_highgamma, y_x_std_highgamma = mean_relationship(cortical_distances, performance_array_highgamma,
                                                         np.arange(0.5, 5.5, 0.5))

    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_theta, marker='o', linewidth=3.0, color=(0.035, 0.062, 0.682))
    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_alpha, marker='o', linewidth=3.0, color=(0.298, 0.662, 0.941))
    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_beta, marker='o', linewidth=3.0, color=(0.031, 0.568, 0.098))
    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_gamma, marker='o', linewidth=3.0, color=(0.960, 0.050, 0.019))
    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_highgamma, marker='o', linewidth=3.0, color=(0.960, 0.454, 0.019))

    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3, 1)
    plt.ylabel('Performance', fontweight='bold')
    plt.xlabel('Cortical distance (mm)', fontweight='bold')
    plt.title(figure_title, fontweight='bold', loc='center')
    plt.legend(('Theta', 'Alpha', 'Beta', 'Gamma', 'High-gamma'), loc='upper right', fontsize=28)
    plt.show()
    fig.savefig(file_name, dpi=200)


def Bandpass_Performance_eccentricity_plotter(distance_array, ecc_pairs, performance_array_theta,
                                              performance_array_alpha, performance_array_beta, performance_array_gamma,
                                              performance_array_highgamma, dist_value, figure_title, file_name):
    """
    Plot discrimination performance of LFP versus eccentricity at different 
    frequency bands
    """
    ecc_values = ecc_pairs[distance_array == dist_value]
    performance_array_values_theta = performance_array_theta[distance_array == dist_value]
    performance_array_values_alpha = performance_array_alpha[distance_array == dist_value]
    performance_array_values_beta = performance_array_beta[distance_array == dist_value]
    performance_array_values_gamma = performance_array_gamma[distance_array == dist_value]
    performance_array_values_highgamma = performance_array_highgamma[distance_array == dist_value]

    y_x_theta, y_x_std_theta = mean_relationship(ecc_values, performance_array_values_theta, np.arange(3, 27, 3))
    y_x_alpha, y_x_std_alpha = mean_relationship(ecc_values, performance_array_values_alpha, np.arange(3, 27, 3))
    y_x_beta, y_x_std_beta = mean_relationship(ecc_values, performance_array_values_beta, np.arange(3, 27, 3))
    y_x_gamma, y_x_std_gamma = mean_relationship(ecc_values, performance_array_values_gamma, np.arange(3, 27, 3))
    y_x_highgamma, y_x_std_highgamma = mean_relationship(ecc_values, performance_array_values_highgamma,
                                                         np.arange(3, 27, 3))

    coeff, p_value_theta = pearsonr(ecc_values, performance_array_values_theta)
    coeff, p_value_alpha = pearsonr(ecc_values, performance_array_values_alpha)
    coeff, p_value_beta = pearsonr(ecc_values, performance_array_values_beta)
    coeff, p_value_gamma = pearsonr(ecc_values, performance_array_values_gamma)
    coeff, p_value_highgamma = pearsonr(ecc_values, performance_array_values_highgamma)

    fig = plt.figure(figsize=(12, 10))

    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_theta, linewidth=3.0, marker='o', color=(0.035, 0.062, 0.682))
    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_alpha, linewidth=3.0, marker='o', color=(0.298, 0.662, 0.941))
    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_beta, linewidth=3.0, marker='o', color=(0.031, 0.568, 0.098))
    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_gamma, linewidth=3.0, marker='o', color=(0.960, 0.050, 0.019))
    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_highgamma, linewidth=3.0, marker='o', color=(0.960, 0.454, 0.019))

    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3, 1)
    plt.ylabel('Performance', fontweight='bold')
    plt.xlabel('Eccentricity (deg)', fontweight='bold')
    plt.legend(('Theta p-value={0}'.format(np.round_(p_value_theta, 4)),
                'Alpha p-value={0}'.format(np.round_(p_value_alpha, 4)),
                'Beta p-value={0}'.format(np.round_(p_value_beta, 4)),
                'Gamma p-value={0}'.format(np.round_(p_value_gamma, 4)),
                'High-gamma p-value={0}'.format(np.round_(p_value_highgamma, 4))), loc='upper right', fontsize=26)
    plt.title(figure_title, fontweight='bold', loc='center')
    plt.show()
    fig.savefig(file_name, dpi=200)
