import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from discrimination import train_test_generator
from sklearn import preprocessing
from scipy.stats import pearsonr
from mean_bin import mean_relationship


def linear_svc_weights(X_train, X_test, y_train, y_test):
    """SVM linear weights
    Determines linear SVM weights for a binary classification
    """
    param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12, 50, 100, 500, 1000]}
    clf = GridSearchCV(LinearSVC(penalty='l2', loss='squared_hinge', multi_class='ovr', class_weight='balanced'),
                       param_grid, cv=5, scoring='roc_auc')
    clf_train = clf.fit(X_train, y_train)
    SVMweights = clf_train.best_estimator_.coef_
    return SVMweights


def SVM_weight_set(probes_pairs, stim, dataset):
    """Array of weights for all discriminations
    Weights of all the discriminations: each row is a discrimination which
    includes all the weights assigned to the electrodes for that discrimination
    """
    weight_set = np.zeros((len(probes_pairs), dataset.shape[1]))
    for i in np.arange(len(probes_pairs)):
        probe1, probe2 = probes_pairs[i, 0], probes_pairs[i, 1]
        X_train, X_test, y_train, y_test = train_test_generator(probe1, probe2, stim, dataset)
        SVMweights = linear_svc_weights(X_train, X_test, y_train, y_test)
        weight_set[i] = SVMweights
    return weight_set


def electrodes_sorting_by_weight(weight_set, responsive_electrodes):
    """Electrode sorting
    This function sorts electrodes from the best to the worst based on their 
    trial-averaged squared weights.
    """
    weight_set = preprocessing.normalize(weight_set)
    electrodes_Importance = np.mean(weight_set ** 2, axis=0)
    electrode_rank = np.argsort(electrodes_Importance)
    electrode_rank = electrode_rank[::-1]
    weightsorted_electrodes = responsive_electrodes[electrode_rank]
    return weightsorted_electrodes


def electrodes_importance_calculator(weight_set):
    """Electrodes importance values
    Importance values of electrodes.
    """
    weight_set = preprocessing.normalize(weight_set)
    electrodes_Importance = np.mean(weight_set ** 2, axis=0)
    return electrodes_Importance


def electrode_pairs_importance(electrode_pairs, electrodes_importance, responsive_electrodes):
    """Importance of electrode pair
    Importance of a pair of electrodes.
    """
    pair_mean_importance = np.zeros(len(electrode_pairs))
    for i in np.arange(len(electrode_pairs)):
        pair_mean_importance[i] = (electrodes_importance[responsive_electrodes == electrode_pairs[i, 0]] +
                                   electrodes_importance[responsive_electrodes == electrode_pairs[i, 1]]) / 2
    return pair_mean_importance


def performance_corticaldistance_plotter_importance(cortical_distances, performance_array, performance_array_four,
                                                    performance_array_six, performance_array_eight,
                                                    performance_array_ten, performance_array_twelve, file_name):
    """Performance-cortical distance for the best electrodes
    Perfromance as a function of cortical distance using the best 4, 6, 8, 10, 
    and 12 electrodes
    """
    y_x, y_x_std = mean_relationship(cortical_distances, performance_array, np.arange(0.5, 5.5, 0.5))
    y_x_four, y_x_std_four = mean_relationship(cortical_distances, performance_array_four, np.arange(0.5, 5.5, 0.5))
    y_x_six, y_x_std_six = mean_relationship(cortical_distances, performance_array_six, np.arange(0.5, 5.5, 0.5))
    y_x_eight, y_x_std_eight = mean_relationship(cortical_distances, performance_array_eight, np.arange(0.5, 5.5, 0.5))
    y_x_ten, y_x_std_ten = mean_relationship(cortical_distances, performance_array_ten, np.arange(0.5, 5.5, 0.5))
    y_x_twelve, y_x_std_twelve = mean_relationship(cortical_distances, performance_array_twelve,
                                                   np.arange(0.5, 5.5, 0.5))

    fig = plt.figure(figsize=(12, 10))

    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x, marker='o', linewidth=3.0, color=(0.031, 0.031, 0.027))
    # plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x-y_x_std,y_x+y_x_std,color=(0.031, 0.031, 0.027))
    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_four, marker='o', linewidth=3.0, color=(0.054, 0.925, 0.964))
    # plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_four-y_x_std_four,y_x_four+y_x_std_four,color=(0.054, 0.925, 0.964))
    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_six, marker='o', linewidth=3.0, color=(0.427, 0.964, 0.054))
    # plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_six-y_x_std_six,y_x_six+y_x_std_six,color=(0.427, 0.964, 0.054))
    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_eight, marker='o', linewidth=3.0, color=(0.964, 0.937, 0.054))
    # plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_eight-y_x_std_eight,y_x_eight+y_x_std_eight,color=(0.964, 0.937, 0.054))
    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_ten, marker='o', linewidth=3.0, color=(0.964, 0.054, 0.780))
    # plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_ten-y_x_std_ten,y_x_ten+y_x_std_ten,color=(0.964, 0.054, 0.780))
    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_twelve, marker='o', linewidth=3.0, color=(0.964, 0.098, 0.054))
    # plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_twelve-y_x_std_twelve,y_x_twelve+y_x_std_twelve,color=(0.964, 0.098, 0.054))

    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3, 1)
    plt.ylabel('Performance', fontweight='bold')
    plt.xlabel('Cortical distance (mm)', fontweight='bold')
    plt.legend(('All', '4', '6', '8', '10', '12'), loc='lower right', fontsize=32)
    plt.show()
    fig.savefig(file_name, dpi=200)


def Performance_eccentricity_plotter_importance(distance_array, ecc_pairs, performance_array, performance_array_four,
                                                performance_array_six, performance_array_eight, performance_array_ten,
                                                performance_array_twelve, dist_value, figure_title, file_name):
    """Performance-eccentricity for the best electrodes
    Perfromance as a function of eccentricity using the best 4, 6, 8, 10, and 
    12 electrodes for specific value of probes separation
    """
    ecc_values = ecc_pairs[distance_array == dist_value]

    performance_array_values = performance_array[distance_array == dist_value]
    performance_array_values_four = performance_array_four[distance_array == dist_value]
    performance_array_values_six = performance_array_six[distance_array == dist_value]
    performance_array_values_eight = performance_array_eight[distance_array == dist_value]
    performance_array_values_ten = performance_array_ten[distance_array == dist_value]
    performance_array_values_twelve = performance_array_twelve[distance_array == dist_value]

    y_x, y_x_std = mean_relationship(ecc_values, performance_array_values, np.arange(3, 27, 3))
    y_x_four, y_x_std_four = mean_relationship(ecc_values, performance_array_values_four, np.arange(3, 27, 3))
    y_x_six, y_x_std_six = mean_relationship(ecc_values, performance_array_values_six, np.arange(3, 27, 3))
    y_x_eight, y_x_std_eight = mean_relationship(ecc_values, performance_array_values_eight, np.arange(3, 27, 3))
    y_x_ten, y_x_std_ten = mean_relationship(ecc_values, performance_array_values_ten, np.arange(3, 27, 3))
    y_x_twelve, y_x_std_twelve = mean_relationship(ecc_values, performance_array_values_twelve, np.arange(3, 27, 3))

    coeff, p_value = pearsonr(ecc_values, performance_array_values)
    coeff, p_value_four = pearsonr(ecc_values, performance_array_values_four)
    coeff, p_value_six = pearsonr(ecc_values, performance_array_values_six)
    coeff, p_value_eight = pearsonr(ecc_values, performance_array_values_eight)
    coeff, p_value_ten = pearsonr(ecc_values, performance_array_values_ten)
    coeff, p_value_twelve = pearsonr(ecc_values, performance_array_values_twelve)

    fig = plt.figure(figsize=(12, 10))

    plt.plot(np.arange(3, 24, 3) + 1.5, y_x, linewidth=3.0, marker='o', color=(0.031, 0.031, 0.027))
    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_four, linewidth=3.0, marker='o', color=(0.054, 0.925, 0.964))
    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_six, linewidth=3.0, marker='o', color=(0.427, 0.964, 0.054))
    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_eight, linewidth=3.0, marker='o', color=(0.964, 0.937, 0.054))
    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_ten, linewidth=3.0, marker='o', color=(0.964, 0.054, 0.780))
    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_twelve, linewidth=3.0, marker='o', color=(0.964, 0.098, 0.054))

    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3, 1)
    plt.ylabel('Performance', fontweight='bold')
    plt.xlabel('Eccentricity (deg)', fontweight='bold')
    plt.legend(('All p-value={0}'.format(np.round_(p_value, 4)), '4 p-value={0}'.format(np.round_(p_value_four, 4)),
                '6 p-value={0}'.format(np.round_(p_value_six, 4)), '8 p-value={0}'.format(np.round_(p_value_eight, 4)),
                '10 p-value={0}'.format(np.round_(p_value_ten, 4)),
                '12 p-value={0}'.format(np.round_(p_value_twelve, 4))), loc='lower right', fontsize=26)
    plt.title(figure_title, fontweight='bold', loc='center')
    plt.show()
    fig.savefig(file_name, dpi=200)


def Electrode_array_show(electrode_set, plxarray, filename):
    """Show location of selected electrodes
    Visualize location of electrodes on the electrode array
    """
    Array_ = np.zeros(100)
    Array_[:] = np.nan
    Array_[electrode_set] = 1
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(Array_[plxarray - 1])
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlabel('Electrodes', fontsize=26, fontweight='bold')
    plt.ylabel('Electrodes', fontsize=26, fontweight='bold')
    plt.show()
    fig.savefig(filename, dpi=350)


def Electrodes_importance_histogram(weight_set, bar_color, filename):
    """Histogram of importance values
    Histogram of importance values
    """
    weight_set_normalized = preprocessing.normalize(weight_set)
    Importance_all = np.mean(weight_set_normalized ** 2, axis=0)
    mean_importance = np.mean(Importance_all)
    mean_importance = np.round_(mean_importance, 4)
    std_importance = np.std(Importance_all)
    std_importance = np.round_(std_importance, 4)
    fig = plt.figure(figsize=(9, 9))
    plt.hist(Importance_all, bins=np.arange(0, 0.105, 0.005), color=bar_color, edgecolor='black')
    plt.xlabel('Importance value', fontweight='bold')
    plt.ylabel('Number of electrodes', fontweight='bold')
    plt.ylim(0, 40)
    plt.title('Mean=%.4f,SD=%.4f' % (mean_importance, std_importance), fontweight='bold', loc='center', fontsize=32)
    plt.show()
    fig.savefig(filename, dpi=200)


def Electrode_array_importance(weight_set, responsive_electrodes, plxarray, v_min, v_max, filename):
    """Importance values on the array
    Visualize importance values of electrodes on the electrode array
    """
    weight_set_normalized = preprocessing.normalize(weight_set)
    Importance_all = np.mean(weight_set_normalized ** 2, axis=0)
    Aray_response = np.zeros(100)
    Aray_response[:] = np.nan
    Aray_response[responsive_electrodes] = Importance_all
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(Aray_response[plxarray - 1], vmin=v_min, vmax=v_max)
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlabel('Electrodes', fontweight='bold')
    plt.ylabel('Electrodes', fontweight='bold')
    plt.subplots_adjust()
    cax = plt.axes([0.95, 0.13, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min, v_max])
    plt.show()
    fig.savefig(filename, dpi=200)


def probes_weight(weight_set, electrode, responsive_electrodes, probe_pairs, dist_value, distance_array, filename):
    """
    Weights values related to discrimination of a probe
    """
    weight_set = preprocessing.normalize(weight_set)
    electrode = np.ravel(np.argwhere(responsive_electrodes == electrode))[0]
    electrode_weights = weight_set[:, electrode]
    probes_weights = np.zeros(100)
    probes_weights[:] = np.nan
    for i in np.unique(probe_pairs[:, 0]):
        probes_weights[i - 1] = -np.mean(electrode_weights[(probe_pairs[:, 0] == i) & (distance_array == dist_value)])
    v_min = np.nanmin(probes_weights)
    v_max = np.nanmax(probes_weights)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(np.reshape(probes_weights, (10, 10)).T)
    plt.xticks(np.array([0, 9]), [-36, 0])
    plt.yticks(np.array([0, 9]), [2, -34])
    plt.xlabel('Position(deg)', fontweight='bold')
    plt.ylabel('Position(deg)', fontweight='bold')
    plt.subplots_adjust()
    cax = plt.axes([1, 0.23, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min, v_max])
    plt.show()
    fig.savefig(filename, dpi=150)


def probes_weight_P3(weight_set, electrode, responsive_electrodes, probe_pairs, dist_value, distance_array, filename):
    """
    Weights values related to discrimination of a probe
    """
    weight_set = preprocessing.normalize(weight_set)
    electrode = np.ravel(np.argwhere(responsive_electrodes == electrode))[0]
    electrode_weights = weight_set[:, electrode]
    probes_weights = np.zeros(100)
    probes_weights[:] = np.nan
    for i in np.unique(probe_pairs[:, 0]):
        probes_weights[i - 1] = -np.mean(electrode_weights[(probe_pairs[:, 0] == i) & (distance_array == dist_value)])
    v_min = np.nanmin(probes_weights)
    v_max = np.nanmax(probes_weights)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(np.reshape(probes_weights, (10, 10)).T)
    plt.xticks(np.array([0, 9]), [-36 + 20, 0 + 20])
    plt.yticks(np.array([0, 9]), [2, -34])
    plt.xlabel('Position(deg)', fontweight='bold')
    plt.ylabel('Position(deg)', fontweight='bold')
    plt.subplots_adjust()
    cax = plt.axes([1, 0.23, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min, v_max])
    plt.show()
    fig.savefig(filename, dpi=150)


def weight_corticaldist(weight_set, cortical_distances, responsive_electrodes, electrode):
    """
    Check any relationship between weights and cortical distance
    """
    weight_set = preprocessing.normalize(weight_set)
    electrode = np.ravel(np.argwhere(responsive_electrodes == electrode))[0]
    electrode_weights = weight_set[:, electrode]
    coeff, p_value = pearsonr(cortical_distances, electrode_weights)
    return p_value


def mean_importance_corticaldist(weight_set, cortical_distances, responsive_electrodes, electrode1, electrode2,
                                 electrode3, electrode4):
    """
    Check any relationship between weights and cortical distance for specific 
    electrodes
    """
    weight_set = preprocessing.normalize(weight_set)
    importance_set = weight_set ** 2
    electrode1 = np.ravel(np.argwhere(responsive_electrodes == electrode1))[0]
    electrode2 = np.ravel(np.argwhere(responsive_electrodes == electrode2))[0]
    electrode3 = np.ravel(np.argwhere(responsive_electrodes == electrode3))[0]
    electrode4 = np.ravel(np.argwhere(responsive_electrodes == electrode4))[0]
    electrode_importance1 = importance_set[:, electrode1]
    electrode_importance2 = importance_set[:, electrode2]
    electrode_importance3 = importance_set[:, electrode3]
    electrode_importance4 = importance_set[:, electrode4]
    electrode_importance = 0.25 * (
                electrode_importance1 + electrode_importance2 + electrode_importance3 + electrode_importance4)
    coeff, p_value = pearsonr(cortical_distances, electrode_importance)
    return p_value
