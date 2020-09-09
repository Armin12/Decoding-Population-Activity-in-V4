import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


def Select_responsive_electrodes(dataset, stim):
    """Responsive electrodes (feature selection)
    This function determines responsive electrode (features) for each response 
    dataset. It gives an score to each of features (96 electrodes) and assigns 
    a p-value to each scores. Features with p-values bigger than 0.05 are 
    excluded from the analysis.
    """
    FS = SelectKBest(f_classif, k=96).fit(dataset, stim)
    responsive_electrodes = np.ravel(np.argwhere(FS.pvalues_ < 0.05))
    return responsive_electrodes


def responsive_probes(dataset, stim):
    """Responsive probes (select relevant labels)
    Some probes do not elicit response in the population of electrodes. To 
    determine if a probe position elicits response in the population, we 
    calculated mean z-scored responses across electrodes and trials for that 
    probe position. Probe positions with positive mean z-score were included in 
    the analysis.
    """
    resp_probes = np.array([], dtype=int)
    for i in np.arange(1, 101):
        if np.mean(dataset[stim == i]) > 0:
            resp_probes = np.append(resp_probes, i)
    return resp_probes
