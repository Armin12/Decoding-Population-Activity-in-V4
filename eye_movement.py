import numpy as np
from probes_coord import dim_calculator


def eye_movement_analysis(eyex_fr1, eyey_fr1, stim):
    """
    This function corrects the probe positions for the eye movement in each 
    trial. It maps the position of a probe in a trial to the closest probe 
    position on the grid.
    """
    eye_movement_array = np.vstack((eyex_fr1, eyey_fr1)).T

    real_stim = np.zeros(len(stim))
    probe_real_position = np.zeros((len(stim), 2))
    for i in np.arange(len(stim)):
        probe_real_position[i, :] = dim_calculator()[stim[i] - 1] - eye_movement_array[i, :]
        full_probe_dist = probe_real_position[i, :] - dim_calculator()
        dismat = np.sqrt((full_probe_dist)[:, 0] ** 2 + (full_probe_dist)[:, 1] ** 2)
        real_stim[i] = np.argmin(dismat) + 1

    return real_stim
