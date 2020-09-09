
import numpy as np
import matplotlib.pyplot as plt

def Signal_visualization(arr,stim,Probe_set,electrode_set,edge_data,ylab,ylimit_bottom,ylimit_top,trcolor,filename):
    """Visualizer of MUA and LFP
    This function plots time traces of signals recorded from speific group 
    of electrodes and in response to a specific set of probes
    """
    arr_elec=arr[:,electrode_set,:]
    fig=plt.figure(figsize=(9, 12))
    c=1 
    for i in Probe_set:
        resp=np.mean(arr_elec[stim==i],axis=0)
        fig.add_subplot(len(Probe_set),1,c)
        plt.plot(edge_data*1000,resp.T,linewidth=0.3,color=trcolor)
        plt.yticks(np.array([ylimit_bottom,ylimit_top]),[ylimit_bottom,ylimit_top])
        plt.ylabel(ylab,fontsize=25,fontweight='bold')
        plt.axis('tight')
        plt.tight_layout()
        plt.xlim(-100,300)
        plt.ylim(ylimit_bottom,ylimit_top)
        c=c+1
    plt.xlabel('Time (ms)',fontsize=25,fontweight='bold')
    plt.show()
    fig.savefig(filename, dpi=350)