import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from mean_bin import mean_relationship
from discrimination import train_test_generator
from scipy.stats import pearsonr


def Discrimination_performance_array_onblind(probes_pairs, stim, dataset):
    """Effect of noise correlations on coding from decoding perspective
    Discrimination performance of correlation-aware versus correlation-blind
    decoder (classifier)
    """

    performance_array = np.zeros(len(probes_pairs))
    for i in np.arange(len(probes_pairs)):
        print(i)
        probe1, probe2 = probes_pairs[i, 0], probes_pairs[i, 1]

        X_clf, X_test, y_clf, y_test = train_test_generator(probe1, probe2, stim, dataset)

        skf = StratifiedKFold(n_splits=5)
        classification_performance = 0
        for train_index, test_index in skf.split(X_clf, y_clf):
            X_train, X_test = X_clf[train_index], X_clf[test_index]
            y_train, y_test = y_clf[train_index], y_clf[test_index]

            X_train0 = shuffler(X_train[y_train == 0])
            X_train1 = shuffler(X_train[y_train == 1])

            X_train = np.vstack((X_train0, X_train1))
            y_train = np.hstack((np.zeros((len(X_train0)), dtype=int), np.ones((len(X_train1)), dtype=int)))

            order = np.argsort(np.random.random(len(X_train)))
            X_train = X_train[order]
            y_train = y_train[order]

            param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12, 50, 100, 500, 1000]}
            clf = GridSearchCV(
                LinearSVC(penalty='l2', loss='squared_hinge', multi_class='ovr', class_weight='balanced'), param_grid,
                cv=5, scoring='roc_auc')
            clf_train = clf.fit(X_train, y_train)
            classification_performance = classification_performance + clf_train.score(X_test, y_test)

        classification_performance = classification_performance / 5
        performance_array[i] = classification_performance

    return performance_array


def shuffler(dataset):
    """Shuffler
    For each electrode, this function, shuffles the order of trials for 
    responses to a particular stimulus
    """
    shuffled_dataset = np.zeros((dataset.shape))
    for j in np.arange(dataset.shape[1]):
        order = np.argsort(np.random.random(len(dataset)))
        shuffled_dataset[:, j] = dataset[:, j][order]
    return shuffled_dataset


def performance_corticaldistance_onblind_plotter(cortical_distances, performance_array, performance_array_blind,
                                                 trace_color, trace_color_blind, figure_title, file_name):
    """Performance-cortical distance
    Plots of performance-cortical distance comparing correlation-aware and 
    correlation-blind decoders
    """
    fig = plt.figure(figsize=(12, 10))

    y_x, y_x_std = mean_relationship(cortical_distances, performance_array, np.arange(0.5, 5.5, 0.5))
    y_x_blind, y_x_std_blind = mean_relationship(cortical_distances, performance_array_blind, np.arange(0.5, 5.5, 0.5))

    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x, marker='o', linewidth=3.0, color=trace_color)
    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_blind, marker='o', linewidth=3.0, color=trace_color_blind)

    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3, 1)
    plt.ylabel('Performance', fontweight='bold')
    plt.xlabel('Cortical distance (mm)', fontweight='bold')
    plt.title(figure_title, fontweight='bold', loc='center')
    plt.legend(('Correlation-aware', 'Correlation-blind'), loc='lower right', fontsize=32)
    plt.show()
    fig.savefig(file_name, dpi=200)


def Performance_eccentricity_onblind_plotter(distance_array, ecc_pairs, performance_array, performance_array_blind,
                                             dist_value, trace_color, trace_color_blind, figure_title, file_name):
    """Performance-eccentricity
    Plots of performance-eccentricity comparing correlation-aware and 
    correlation-blind decoders
    """
    ecc_values = ecc_pairs[distance_array == dist_value]
    performance_array_values = performance_array[distance_array == dist_value]
    performance_array_values_blind = performance_array_blind[distance_array == dist_value]

    y_x, y_x_std = mean_relationship(ecc_values, performance_array_values, np.arange(3, 27, 3))
    y_x_blind, y_x_std_blind = mean_relationship(ecc_values, performance_array_values_blind, np.arange(3, 27, 3))

    coeff, p_value = pearsonr(ecc_values, performance_array_values)
    coeff, p_value_blind = pearsonr(ecc_values, performance_array_values_blind)

    fig = plt.figure(figsize=(12, 10))

    plt.plot(np.arange(3, 24, 3) + 1.5, y_x, linewidth=3.0, marker='o', color=trace_color)
    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_blind, linewidth=3.0, marker='o', color=trace_color_blind)

    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3, 1)
    plt.ylabel('Performance', fontweight='bold')
    plt.xlabel('Eccentricity (deg)', fontweight='bold')
    plt.legend(('Correlation-aware p-value={0}'.format(np.round_(p_value, 4)),
                'Correlation-blind p-value={0}'.format(np.round_(p_value_blind, 4))), loc='lower right', fontsize=28)
    plt.title(figure_title, fontweight='bold', loc='center')
    plt.show()
    fig.savefig(file_name, dpi=200)


def Performance_eccentricity_onblind_plotter_P3(distance_array, ecc_pairs, performance_array, performance_array_blind,
                                                dist_value, trace_color, trace_color_blind, figure_title, file_name):
    """Performance-eccentricity
    Plots of performance-eccentricity comparing correlation-aware and 
    correlation-blind decoders
    """
    ecc_values = ecc_pairs[distance_array == dist_value]
    performance_array_values = performance_array[distance_array == dist_value]
    performance_array_values_blind = performance_array_blind[distance_array == dist_value]

    y_x, y_x_std = mean_relationship(ecc_values, performance_array_values, np.arange(3, 21, 3))
    y_x_blind, y_x_std_blind = mean_relationship(ecc_values, performance_array_values_blind, np.arange(3, 21, 3))

    coeff, p_value = pearsonr(ecc_values, performance_array_values)
    coeff, p_value_blind = pearsonr(ecc_values, performance_array_values_blind)

    fig = plt.figure(figsize=(12, 10))

    plt.plot(np.arange(3, 18, 3) + 1.5, y_x, linewidth=3.0, marker='o', color=trace_color)
    plt.plot(np.arange(3, 18, 3) + 1.5, y_x_blind, linewidth=3.0, marker='o', color=trace_color_blind)

    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3, 1)
    plt.ylabel('Performance', fontweight='bold')
    plt.xlabel('Eccentricity (deg)', fontweight='bold')
    plt.legend(('Correlation-aware p-value={0}'.format(np.round_(p_value, 4)),
                'Correlation-blind p-value={0}'.format(np.round_(p_value_blind, 4))), loc='lower right', fontsize=28)
    plt.title(figure_title, fontweight='bold', loc='center')
    plt.show()
    fig.savefig(file_name, dpi=200)
