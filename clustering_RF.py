import numpy as np
import matplotlib.pyplot as plt
from RF_calc import stimulus_averaged_responses
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from mean_bin import mean_relationship


def RF_cluster(stim,dataset,responsive_electrodes,no_cluster,cat_no):
    """Clustering RFs
    This function for each given response dataset, clusters the responsive 
    electrodes into no_cluster groups and takes cat_no-th group to produce 
    output. We clustered the receptive fields to find the similar receptive 
    fields.
    """
    averaged_responses=stimulus_averaged_responses(stim,dataset)
    clustering_labels = KMeans(init='k-means++',n_clusters=no_cluster, n_init=15, random_state=0).fit_predict(averaged_responses[:,responsive_electrodes].T)
    cat=responsive_electrodes[np.ravel(np.argwhere(clustering_labels==cat_no))]
    return cat


def RF_representative_electrode(cat,stim,dataset):
    """RF representative electrodes
    Select 1-3 electrodes from each group with the highest RF peak
    """
    averaged_responses=stimulus_averaged_responses(stim,dataset)
    RF_peak=np.amax(averaged_responses,axis=0)
    sorted_cat=cat[np.argsort(RF_peak[cat])]
    sorted_cat=sorted_cat[::-1]
    three_highest_peak=sorted_cat[0:3]
    return three_highest_peak


def performance_corticaldistance_plotter_clustering(cortical_distances,performance_array,performance_array_clustering,trace_color,trace_color_clustering,figure_title,file_name):
    """Performance for clustering selected electrodes
    Plot performance as a function of cortical distance comparing all vs 
    clustering-selected electrodes
    """
    y_x,y_x_std=mean_relationship(cortical_distances,performance_array,np.arange(0.5,5.5,0.5))
    y_x_clustering,y_x_std_clustering=mean_relationship(cortical_distances,performance_array_clustering,np.arange(0.5,5.5,0.5))
    
    fig=plt.figure(figsize=(12, 10))
    
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x,marker='o',linewidth=3.0,color=trace_color)
    #plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x-y_x_std,y_x+y_x_std,color=trace_color)
    
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_clustering,marker='o',linewidth=3.0,color=trace_color_clustering)
    #plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_clustering-y_x_std_clustering,y_x_clustering+y_x_std_clustering,color=trace_color_clustering)
    
    
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3,1)
    plt.ylabel('Performance',fontweight='bold')
    plt.xlabel('Cortical distance (mm)',fontweight='bold')
    plt.title(figure_title,fontweight='bold',loc='center')
    plt.legend(('All','Clustering-selected'),loc='lower right',fontsize=32)
    plt.show()
    fig.savefig(file_name, dpi=200)

def Performance_eccentricity_plotter_clustering(distance_array,ecc_pairs,performance_array,performance_array_clustering,dist_value,trace_color,trace_color_clustering,figure_title,file_name):
    """Performance for clustering selected electrodes
    Plot performance as a function of eccentricity comparing all vs 
    clustering-selected electrodes
    """
    ecc_values=ecc_pairs[distance_array==dist_value]
    performance_array_values=performance_array[distance_array==dist_value]
    performance_array_values_clustering=performance_array_clustering[distance_array==dist_value]
    
    y_x,y_x_std=mean_relationship(ecc_values,performance_array_values,np.arange(3,27,3))
    y_x_clustering,y_x_std_clustering=mean_relationship(ecc_values,performance_array_values_clustering,np.arange(3,27,3))
    
    coeff,p_value=pearsonr(ecc_values,performance_array_values)
    coeff,p_value_clustering=pearsonr(ecc_values,performance_array_values_clustering)
    
    fig=plt.figure(figsize=(12, 10))
    
    plt.plot(np.arange(3,24,3)+1.5,y_x,linewidth=3.0,marker='o',color=trace_color)
    plt.plot(np.arange(3,24,3)+1.5,y_x_clustering,linewidth=3.0,marker='o',color=trace_color_clustering)
    
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3,1)
    plt.ylabel('Performance',fontweight='bold')
    plt.xlabel('Eccentricity (deg)',fontweight='bold')
    plt.legend(('All p-value={0}'.format(np.round_(p_value,4)),'Clustering-selected p-value={0}' .format(np.round_(p_value_clustering,4))),loc='lower right',fontsize=28)
    plt.title(figure_title,fontweight='bold',loc='center')
    plt.show()
    fig.savefig(file_name, dpi=200)
    
