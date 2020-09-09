import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV)
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mean_bin import mean_relationship
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr


def train_test_generator(probe1, probe2, stim, dataset):
    """Train-test splitter
    This function creates training and test sets for binary classification
    """
    X_clf = np.vstack((dataset[stim == probe1], dataset[stim == probe2]))
    y_clf = np.hstack(
        (np.zeros((len(stim[stim == probe1])), dtype=int), np.ones((len(stim[stim == probe2])), dtype=int)))
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0, random_state=0)
    return X_train, X_test, y_train, y_test


def Response_discriminator(X_train, X_test, y_train, y_test, estimator_type):
    """Response discriminator
    Binary classifier with different estimators. For each probe pair, 5-fold 
    cross-validation performance was calculated and grid search was used to 
    find the best hyperparameters to prevent overfitting. Classification 
    performance was measured as area under ROC curve. Outputs are training, 
    validation, and test performances
    """
    if estimator_type == 'Logistic Regression':
        param_grid = {'C': [0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12, 100, 1000]}
        clf = GridSearchCV(
            LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear', multi_class='ovr'),
            param_grid, cv=5, scoring='roc_auc')
        clf_train = clf.fit(X_train, y_train)
        Training_score = clf_train.score(X_train, y_train)
        Validation_score = clf_train.best_score_
        # classification_performance=clf_train.score(X_test, y_test)
    elif estimator_type == 'LDA':
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf_train = clf.fit(X_train, y_train)
        y_train_score = clf_train.decision_function(X_train)
        Training_score = roc_auc_score(y_train, y_train_score, average='weighted')
        Validation_score = cross_val_score(clf_train, X_train, y_train, cv=5, scoring='roc_auc').mean()
        # y_test_score=clf_train.decision_function(X_test)
        # classification_performance=roc_auc_score(y_test,y_test_score,average='weighted')
    elif estimator_type == 'Linear SVM':
        param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12, 50, 100, 500, 1000]}
        clf = GridSearchCV(LinearSVC(penalty='l2', loss='squared_hinge', multi_class='ovr', class_weight='balanced'),
                           param_grid, cv=5, scoring='roc_auc')
        clf_train = clf.fit(X_train, y_train)
        Training_score = clf_train.score(X_train, y_train)
        Validation_score = clf_train.best_score_
        # classification_performance=clf_train.score(X_test, y_test)
    else:
        param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12, 50, 100, 500, 1000],
                      'gamma': [10, 1, 0.1, 0.05, 0.01, 0.005, 0.007, 0.001, 0.0005, 0.0001, 0.00001, 0.000001],
                      'kernel': ['rbf']}
        clf = GridSearchCV(SVC(decision_function_shape='ovr', class_weight='balanced'), param_grid, cv=5,
                           scoring='roc_auc')
        clf_train = clf.fit(X_train, y_train)
        Training_score = clf_train.score(X_train, y_train)
        Validation_score = clf_train.best_score_
        # classification_performance=clf_train.score(X_test, y_test)
    return Training_score, Validation_score


def Discrimination_performance_array(probes_pairs, stim, dataset, estimator_type):
    """
    This function calculates validation and test performance for all the 
    discriminations (all responsive probe pairs)
    """
    performance_array = np.zeros(len(probes_pairs))
    for i in np.arange(len(probes_pairs)):
        print(i)
        probe1, probe2 = probes_pairs[i, 0], probes_pairs[i, 1]
        X_train, X_test, y_train, y_test = train_test_generator(probe1, probe2, stim, dataset)
        Training_score, Validation_score = Response_discriminator(X_train, X_test, y_train, y_test, estimator_type)
        performance_array[i] = Validation_score
    return performance_array


def performance_corticaldistance_plotter(cortical_distances_wide, cortical_distances_medium, cortical_distances_narrow,
                                         performance_array_wide, performance_array_medium, performance_array_narrow,
                                         trace_color_wide, trace_color_medium, trace_color_narrow, figure_title,
                                         file_name):
    """Performance vs cortical distance
    This function plots discrimination performance vs cortical distance and 
    compares resultss for wide, medium, and narrow windows.
    """
    fig = plt.figure(figsize=(12, 10))

    y_x_wide, y_x_std_wide = mean_relationship(cortical_distances_wide, performance_array_wide,
                                               np.arange(0.5, 5.5, 0.5))
    y_x_medium, y_x_std_medium = mean_relationship(cortical_distances_medium, performance_array_medium,
                                                   np.arange(0.5, 5.5, 0.5))
    y_x_narrow, y_x_std_narrow = mean_relationship(cortical_distances_narrow, performance_array_narrow,
                                                   np.arange(0.5, 5.5, 0.5))

    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_wide, marker='o', linewidth=3.0, color=trace_color_wide)
    # plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_wide-y_x_std_wide,y_x_wide+y_x_std_wide,color=trace_color_wide)

    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_medium, marker='o', linewidth=3.0, color=trace_color_medium)
    # plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_medium-y_x_std_medium,y_x_medium+y_x_std_medium,color=trace_color_medium)

    plt.plot(np.arange(0.5, 5, 0.5) + 0.25, y_x_narrow, marker='o', linewidth=3.0, color=trace_color_narrow)
    # plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_narrow-y_x_std_narrow,y_x_narrow+y_x_std_narrow,color=trace_color_narrow)

    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3, 1)
    plt.ylabel('Performance', fontweight='bold')
    plt.xlabel('Cortical distance (mm)', fontweight='bold')
    plt.title(figure_title, fontweight='bold', loc='center')
    plt.legend(('Wide', 'Medium', 'Narrow'), loc='lower right', fontsize=32)
    plt.show()
    fig.savefig(file_name, dpi=200)


def Performance_eccentricity_plotter(distance_array_wide, distance_array_medium, distance_array_narrow, ecc_pairs_wide,
                                     ecc_pairs_medium, ecc_pairs_narrow, performance_array_wide,
                                     performance_array_medium, performance_array_narrow, dist_value, trace_color_wide,
                                     trace_color_medium, trace_color_narrow, figure_title, file_name):
    """
    This function plots discrimination performance as a function of probes mean 
    eccentricity for specific probes separation. Then, it calculates linear 
    correlation p-value between performance and eccentricity.
    """
    ecc_values_wide = ecc_pairs_wide[distance_array_wide == dist_value]
    performance_array_values_wide = performance_array_wide[distance_array_wide == dist_value]
    ecc_values_medium = ecc_pairs_medium[distance_array_medium == dist_value]
    performance_array_values_medium = performance_array_medium[distance_array_medium == dist_value]
    ecc_values_narrow = ecc_pairs_narrow[distance_array_narrow == dist_value]
    performance_array_values_narrow = performance_array_narrow[distance_array_narrow == dist_value]

    y_x_wide, y_x_std_wide = mean_relationship(ecc_values_wide, performance_array_values_wide, np.arange(3, 27, 3))
    y_x_medium, y_x_std_medium = mean_relationship(ecc_values_medium, performance_array_values_medium,
                                                   np.arange(3, 27, 3))
    y_x_narrow, y_x_std_narrow = mean_relationship(ecc_values_narrow, performance_array_values_narrow,
                                                   np.arange(3, 27, 3))

    coeff, p_value_wide = pearsonr(ecc_values_wide, performance_array_values_wide)
    coeff, p_value_medium = pearsonr(ecc_values_medium, performance_array_values_medium)
    coeff, p_value_narrow = pearsonr(ecc_values_narrow, performance_array_values_narrow)

    fig = plt.figure(figsize=(12, 10))

    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_wide, linewidth=3.0, marker='o', color=trace_color_wide)
    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_medium, linewidth=3.0, marker='o', color=trace_color_medium)
    plt.plot(np.arange(3, 24, 3) + 1.5, y_x_narrow, linewidth=3.0, marker='o', color=trace_color_narrow)

    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3, 1)
    plt.ylabel('Performance', fontweight='bold')
    plt.xlabel('Eccentricity (deg)', fontweight='bold')
    plt.legend(('Wide p-value={0}'.format(np.round_(p_value_wide, 4)),
                'Medium p-value={0}'.format(np.round_(p_value_medium, 4)),
                'Narrow p-value={0}'.format(np.round_(p_value_narrow, 4))), loc='lower right', fontsize=32)
    plt.title(figure_title, fontweight='bold', loc='center')
    plt.show()
    fig.savefig(file_name, dpi=200)


def Performance_eccentricity_plotter_P3(distance_array_wide, distance_array_medium, distance_array_narrow,
                                        ecc_pairs_wide, ecc_pairs_medium, ecc_pairs_narrow, performance_array_wide,
                                        performance_array_medium, performance_array_narrow, dist_value,
                                        trace_color_wide, trace_color_medium, trace_color_narrow, figure_title,
                                        file_name):
    """
    This function plots discrimination performance as a function of probes mean 
    eccentricity for specific probes separation. Then, it calculates linear 
    correlation p-value between performance and eccentricity.
    """
    ecc_values_wide = ecc_pairs_wide[distance_array_wide == dist_value]
    performance_array_values_wide = performance_array_wide[distance_array_wide == dist_value]
    ecc_values_medium = ecc_pairs_medium[distance_array_medium == dist_value]
    performance_array_values_medium = performance_array_medium[distance_array_medium == dist_value]
    ecc_values_narrow = ecc_pairs_narrow[distance_array_narrow == dist_value]
    performance_array_values_narrow = performance_array_narrow[distance_array_narrow == dist_value]

    y_x_wide, y_x_std_wide = mean_relationship(ecc_values_wide, performance_array_values_wide, np.arange(3, 21, 3))
    y_x_medium, y_x_std_medium = mean_relationship(ecc_values_medium, performance_array_values_medium,
                                                   np.arange(3, 21, 3))
    y_x_narrow, y_x_std_narrow = mean_relationship(ecc_values_narrow, performance_array_values_narrow,
                                                   np.arange(3, 21, 3))

    coeff, p_value_wide = pearsonr(ecc_values_wide, performance_array_values_wide)
    coeff, p_value_medium = pearsonr(ecc_values_medium, performance_array_values_medium)
    coeff, p_value_narrow = pearsonr(ecc_values_narrow, performance_array_values_narrow)

    fig = plt.figure(figsize=(12, 10))

    plt.plot(np.arange(3, 18, 3) + 1.5, y_x_wide, linewidth=3.0, marker='o', color=trace_color_wide)
    plt.plot(np.arange(3, 18, 3) + 1.5, y_x_medium, linewidth=3.0, marker='o', color=trace_color_medium)
    plt.plot(np.arange(3, 18, 3) + 1.5, y_x_narrow, linewidth=3.0, marker='o', color=trace_color_narrow)

    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3, 1)
    plt.ylabel('Performance', fontweight='bold')
    plt.xlabel('Eccentricity (deg)', fontweight='bold')
    plt.legend(('Wide p-value={0}'.format(np.round_(p_value_wide, 4)),
                'Medium p-value={0}'.format(np.round_(p_value_medium, 4)),
                'Narrow p-value={0}'.format(np.round_(p_value_narrow, 4))), loc='lower right', fontsize=32)
    plt.title(figure_title, fontweight='bold', loc='center')
    plt.show()
    fig.savefig(file_name, dpi=200)
