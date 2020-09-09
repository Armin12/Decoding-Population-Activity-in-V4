import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def MUA_Response_dataset(MUA, time_window):
    """MUA response calculator
    This function calculates MUA response to stimulus from the recorded 
    activity and outputs responses of all the electrodes in all trials. 
    'w','m', and 'n' are abbreviations for winde, medium, and narrow window. 
    At the end, responses are standardized (z-scored).
    """
    if time_window == 'w':
        dataset = np.sum(MUA[:, :, 16:22], axis=2)  # wide 50-200 ms
    elif time_window == 'm':
        dataset = np.sum(MUA[:, :, 16:18], axis=2)  # midium 50-100 ms
    elif time_window == 'n':
        dataset = np.sum(MUA[:, :, 16:17], axis=2)  # narrow 50-75 ms
    dataset = preprocessing.StandardScaler().fit_transform(dataset)
    return dataset


def LFP_Response_dataset(LFP, time_window):
    """LFP response calculator
    This function calculates LFP response to stimulus from the recorded 
    activity and outputs responses of all the electrodes in all trials. 
    'w','m', and 'n' are abbreviations for winde, medium, and narrow window. 
    At the end, responses are standardized (z-scored).
    """
    if time_window == 'w':
        dataset = np.sum(LFP[:, :, 200:275], axis=2)  # wide 50-200 ms
    elif time_window == 'm':
        dataset = np.sum(LFP[:, :, 200:225], axis=2)  # medium 50-100 ms
    elif time_window == 'n':
        dataset = np.sum(LFP[:, :, 200:212], axis=2)  # narrow 50-75 ms
    dataset = preprocessing.StandardScaler().fit_transform(dataset)
    return dataset


def Aray_response_visualization(stim, dataset, Probe_set, plxarray, v_min, v_max, filename):
    """
    This function plots the mean responses to a set of stimuli on the electrode 
    array
    """
    c = 1
    fig = plt.figure(figsize=(9, 12))
    for i in Probe_set:
        Aray_response = np.zeros(100)
        Aray_response[0:96] = np.mean(dataset[stim == i, :], axis=0)
        Aray_response[np.arange(96, 100)] = np.nan
        fig.add_subplot(len(Probe_set), 1, c)
        plt.imshow(Aray_response[plxarray - 1], vmin=v_min, vmax=v_max)
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
        c = c + 1
    plt.xlabel('Response z-score')
    plt.subplots_adjust()
    cax = plt.axes([0.7, 0.13, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min, v_max])
    plt.show()
    fig.savefig(filename, dpi=350)
