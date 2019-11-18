print(__doc__)

# Author: Armin Najarpour Foroushani -- <armin.najarpour-foroushani@polymtl.ca>
# Neural analysis

###################### Import Libraries ##########################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC,LinearSVC
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from scipy.signal import butter,lfilter
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr


###################### Set fonts ##########################
font = {'family' : 'sans-serif','weight' : 'bold','size'   : '40'}
rc('font', **font)  # pass in the font dict as kwargs


###################### Methods ##########################

######### Visualization of recordings

#This function plots time traces of signals recorded from speific group
#of electrodes and in response to a specific set of probes

def Signal_visualization(arr,stim,Probe_set,electrode_set,edge_data,ylab,ylimit_bottom,ylimit_top,trcolor,filename):
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

######### Response dataset

#These two function calculates the responses to stimuli from the recorded activities
#and outputs the responses of all the electrodes in all the trials
# 'w','m', and 'n' are abbreviations for winde, medium, and narrow that show
#which response window was considered. Responses are z-scored (standardized) for each electrode.

def MUA_Response_dataset(MUA,time_window):
    if time_window=='w':
        dataset=np.sum(MUA[:,:,16:22], axis=2)#wide 50-200 ms
    elif time_window=='m':
        dataset=np.sum(MUA[:,:,16:18], axis=2)#midium 50-100 ms
    elif time_window=='n':
        dataset=np.sum(MUA[:,:,16:17], axis=2)#narrow 50-75 ms
    dataset = preprocessing.StandardScaler().fit_transform(dataset)
    return dataset

def LFP_Response_dataset(LFP,time_window):
    if time_window=='w':
        dataset=np.sum(LFP[:,:,200:275], axis=2)#wide 50-200 ms
    elif time_window=='m':
        dataset=np.sum(LFP[:,:,200:225], axis=2)#medium 50-100 ms
    elif time_window=='n':
        dataset=np.sum(LFP[:,:,200:212], axis=2)#narrow 50-75 ms
    dataset = preprocessing.StandardScaler().fit_transform(dataset)
    return dataset

#This function plots the mean responses to a set of stimuli on the 
#electrode array
def Aray_response_visualization(stim,dataset,Probe_set,plxarray,v_min,v_max,filename):
    c=1
    fig=plt.figure(figsize=(9,12))
    for i in Probe_set:
        Aray_response=np.zeros(100)
        Aray_response[0:96]=np.mean(dataset[stim==i,:],axis=0)
        Aray_response[np.arange(96,100)]=np.nan
        fig.add_subplot(len(Probe_set),1,c)
        plt.imshow(Aray_response[plxarray-1],vmin=v_min, vmax=v_max)
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
        c=c+1
    plt.xlabel('Response z-score')
    plt.subplots_adjust()
    cax = plt.axes([0.7, 0.13, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min,v_max])
    plt.show()
    fig.savefig(filename, dpi=350)


######### Probes coordination

#coordination of probes on the grid in visual degree with fovea location as the reference
#location of the fixed point (fovea) is at the upper right of the grid
#The distance between neighbouring probes is 4 visual degrees. So, the
#distance between the first and the last probe in a row or column is
#36 visual degrees
def dim_calculator():
    probe_set=np.arange(1,101)
    X=-36+((probe_set-1)//10)*4
    Y=2-((probe_set-1)%10)*4
    dim=np.vstack((X,Y)).T
    return dim

######### Receptive fields

#This function calculates the average responses to each stimulus
#For each probe location, 96 response values (each for an electrode) are obtained
def stimulus_averaged_responses(stim,dataset):
    averaged_responses=np.zeros((100,96))
    for i in np.arange(1,101):
        averaged_responses[i-1,:]=np.mean(dataset[stim==i,:],axis=0)
    return averaged_responses

#This function chooses the mean responses of an electrode to all the probe locations
def RF(averaged_responses,electrode):
    RF_electrode=averaged_responses[:,electrode]
    return RF_electrode

#This function plots a receptive field when takes the receptive field vector of an electrode RF_electrode 
def RF_plotter(RF_electrode,v_min,v_max,filename):
    fig=plt.figure(figsize=(9, 12))
    RF_electrode=np.reshape(RF_electrode,(10,10)).T
    RF_electrode=gaussian_filter(RF_electrode,sigma=0.8)
    plt.imshow(RF_electrode,interpolation='bilinear',vmin=v_min, vmax=v_max)
    plt.xticks(np.array([0,9]),[-36,0])
    plt.yticks(np.array([0,9]),[2,-34])
    plt.xlabel('Position(deg)',fontsize=40,fontweight='bold')
    plt.ylabel('Position(deg)',fontsize=40,fontweight='bold')
    plt.subplots_adjust()
    cax = plt.axes([1, 0.23, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min,v_max])
    plt.show()
    fig.savefig(filename, dpi=150)

#This function plots receptive fields of a group of electrodes
def RF_set_viewer(stim,dataset,electrode_set,v_min,v_max,filename):
    fig=plt.figure(figsize=(15, 15))
    averaged_responses=stimulus_averaged_responses(stim,dataset)
    c=1
    for electrode in electrode_set:
        RF_electrode=RF(averaged_responses,electrode)
        RF_electrode=np.reshape(RF_electrode,(10,10)).T
        RF_electrode=gaussian_filter(RF_electrode,sigma=0.8)
        fig.add_subplot(10,10,c)
        plt.imshow(RF_electrode,interpolation='bilinear',vmin=v_min, vmax=v_max)
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
        c=c+1
    plt.show()
    fig.savefig(filename, dpi=350)

def tuning_plotter(stim,dataset,electrode,v_min,v_max,filename):
    averaged_responses=stimulus_averaged_responses(stim,dataset)
    RF_electrode=RF(averaged_responses,electrode)
    RF_electrode=np.reshape(RF_electrode,(10,10)).T
    fig=plt.figure(figsize=(5, 5))
    plt.imshow(RF_electrode,vmin=v_min, vmax=v_max)
    plt.xticks(np.array([0,9]),[-36,0])
    plt.yticks(np.array([0,9]),[2,-34])
    plt.xlabel('Position(deg)',fontweight='bold')
    plt.ylabel('Position(deg)',fontweight='bold')
    plt.subplots_adjust()
    cax = plt.axes([1, 0.23, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min,v_max])
    plt.show()
    fig.savefig(filename, dpi=150)

#2D Gaussian function that takes x as the input and outputs a corresponding Gaussian output value
#This will be called in the gaussian_RF function
def two_d_gaussian(x,A,x1_0,x2_0,sigma_x1,sigma_x2,d):
    return A*np.exp(-((x[:,0]-x1_0)**2/(2*sigma_x1**2))-((x[:,1]-x2_0)**2/(2*sigma_x2**2)))+d

#This function fits a 2D Gaussian to the Receptive field of an electrode
#in terms of its x-y locations on the grid
def gaussian_RF(RF_electrode,x1_0,x2_0,sigma_x1,sigma_x2):
    dim=dim_calculator()
    popt,pcov = curve_fit(two_d_gaussian,dim,RF_electrode,p0=[1,x1_0,x2_0,sigma_x1,sigma_x2,0],bounds=([-2,-36,-34,2,2,-2], [3,0,2,30,30,2]))
    RF_gaussian=two_d_gaussian(dim,*popt)
    return RF_gaussian,popt

#This function calculates full width at half maximum (FWHM) of the fitted Gaussian to
#a receptive field. It takes as input the parameters of the fitted Gaussian
def FWHM_gaussian(param_G):
    X=np.arange(-36,4,4)
    Y=np.arange(2,-38,-4)
    X,Y = np.meshgrid(X,Y)
    Z=param_G[0]*np.exp(-((X-param_G[1])**2/(2*param_G[3]**2))-((Y-param_G[2])**2/(2*param_G[4]**2)))+param_G[5]-(param_G[5]+param_G[0])/2
    return X,Y,Z

#This function takes the parameters of the fitted Gaussian to calculate
#the diameters of the FWHM ellipse
def RF_diameter(param_G):
    Dx=2*np.abs(param_G[3])*np.sqrt(-2*np.log(1/2-(1/2)*(param_G[5]/param_G[0])))
    Dy=2*np.abs(param_G[4])*np.sqrt(-2*np.log(1/2-(1/2)*(param_G[5]/param_G[0])))
    return Dx,Dy

#This function plots the fitted FWHM ellipse receptive fields of a group of electrodes
#In addition, it returns the mean RF diameter of all the electrodes in
#the electrode_set
def population_contours(dataset,stim,electrode_set,sigm,clr):
    dim=dim_calculator()
    averaged_responses=stimulus_averaged_responses(stim,dataset)
    RF_mean_diameter=np.zeros(len(electrode_set))
    c=0
    for electrode in electrode_set:
        RF_electrode=RF(averaged_responses,electrode)
        ind=np.argmax(RF_electrode)
        x1_0,x2_0,sigma_x1,sigma_x2=dim[ind,0],dim[ind,1],sigm,sigm
        RF_G,param_G=gaussian_RF(RF_electrode,x1_0,x2_0,sigma_x1,sigma_x2)
        X,Y,Z=FWHM_gaussian(param_G)
        Dx,Dy=RF_diameter(param_G)
        D=0.5*(Dx+Dy)
        RF_mean_diameter[c]=D
        plt.contour(X,Y,Z,[0],colors=clr,linewidths=2)
        c=c+1
    plt.xlabel('Position(deg)',fontsize=40,fontweight='bold')
    plt.ylabel('Position(deg)',fontsize=40,fontweight='bold') 
    return RF_mean_diameter

######### Responsive electrodes 

#This function determines the responsive electrode for each response dataset
#It gives scores to each of the 96 features (electrodes) and also assigns p-values
#for the assigned scores. The features with p-values higher than 0.05 are excluded
#from the analysis
def Select_responsive_electrodes(dataset,stim):
    FS = SelectKBest(f_classif, k=96).fit(dataset,stim)
    responsive_electrodes=np.ravel(np.argwhere(FS.pvalues_<0.05))
    return responsive_electrodes

######### Responsive probes

#Some probes do not elicit response in the population of electrodes
#To determine if a probe position elicits response in the population, we
#calculated the mean z-scored values of the response dataset across electrodes and trials
#for that probe position. Probe positions with positive mean z-score were included for analysis.
def responsive_probes(dataset,stim):
    resp_probes=np.array([],dtype=int)
    for i in np.arange(1,101):
        if np.mean(dataset[stim==i])>0:
            resp_probes=np.append(resp_probes,i)
    return resp_probes

######### Distances of probe pairs

#This function calculates the Euclidean distance between two probes
#in terms of visual degrees
def probe_distance(probe1,probe2):
    dim=dim_calculator()
    delta_dim=dim[probe1-1]-dim[probe2-1]
    dist=np.sqrt(delta_dim[0]**2+delta_dim[1]**2)
    return dist

#This function calculates distances between each pair of probes given by
#probes_pairs
def probe_distances_array(probes_pairs):
    distance_array=np.zeros(len(probes_pairs),dtype=float)
    for i in np.arange(len(probes_pairs)):
        probe1,probe2 = probes_pairs[i,0], probes_pairs[i,1]
        distance_array[i]=probe_distance(probe1,probe2)
    return distance_array

######### Eccentricity and magnification factor

#This function calculates the eccentricity of each probe you want
def Probe_eccentricity(probe,dim):
    probe_ecc=np.sqrt(dim[probe-1,0]**2+dim[probe-1,1]**2)
    return probe_ecc

#This function calculates mean eccentricity of two probes for an array of
#probe pairs
def Pair_eccentricity(probe_pairs):
    dim=dim_calculator()
    ecc_pairs=np.zeros(len(probe_pairs))
    for i in np.arange(len(probe_pairs)):
        ecc_pairs[i]=0.5*(Probe_eccentricity(probe_pairs[i,0],dim)+Probe_eccentricity(probe_pairs[i,1],dim))
    return ecc_pairs

#This function calculates the cortical magnification factor at specific
#eccentricities for area V4 macaque monkey
#Negative power doesn't work for an array in Python 3
def Magnification_factor(ecc_pairs):
    mgf_pairs=np.zeros(len(ecc_pairs))
    for i in np.arange(len(ecc_pairs)):
        mgf_pairs[i]=3.01*ecc_pairs[i]**(-0.9)
    return mgf_pairs

#This function finds cortical distance values by multiplying the cortical magnification
#to probe distances
def cortical_distance(distance_array,mgf_pairs):
    cortical_distances=np.multiply(distance_array,mgf_pairs)
    return cortical_distances

######### Discrimination

#Create the training and test set for a binary classification
def train_test_generator(probe1,probe2,stim,dataset):
    X_clf=np.vstack((dataset[stim==probe1],dataset[stim==probe2]))
    y_clf=np.hstack((np.zeros((len(stim[stim==probe1])), dtype=int),np.ones((len(stim[stim==probe2])), dtype=int)))
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0, random_state=0)
    return X_train, X_test, y_train, y_test


#Binary classifier using different estimators. The outputs are the
#training, cross-validation, and test performances
def Response_discriminator(X_train, X_test, y_train, y_test,estimator_type):
    if estimator_type=='Logistic Regression':
        param_grid = {'C': [0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12, 100, 1000]}
        clf = GridSearchCV(LogisticRegression(penalty='l2',class_weight='balanced', solver='liblinear', multi_class='ovr'), param_grid, cv=5, scoring='roc_auc')
        clf_train=clf.fit(X_train, y_train)
        Training_score=clf_train.score(X_train, y_train)
        Validation_score=clf_train.best_score_
        #classification_performance=clf_train.score(X_test, y_test)
    elif estimator_type=='LDA':
        clf = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
        clf_train=clf.fit(X_train, y_train)
        y_train_score=clf_train.decision_function(X_train)
        Training_score=roc_auc_score(y_train,y_train_score,average='weighted')
        Validation_score=cross_val_score(clf_train,X_train,y_train,cv=5,scoring='roc_auc').mean()
        #y_test_score=clf_train.decision_function(X_test)
        #classification_performance=roc_auc_score(y_test,y_test_score,average='weighted')
    elif estimator_type=='Linear SVM':
        param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12,50, 100,500, 1000]}
        clf = GridSearchCV(LinearSVC(penalty='l2',loss='squared_hinge',multi_class='ovr',class_weight='balanced'), param_grid, cv=5, scoring='roc_auc') 
        clf_train=clf.fit(X_train, y_train)
        Training_score=clf_train.score(X_train, y_train)
        Validation_score=clf_train.best_score_
        #classification_performance=clf_train.score(X_test, y_test)
    else:
        param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12,50, 100,500, 1000], 'gamma': [10, 1, 0.1, 0.05, 0.01, 0.005, 0.007, 0.001, 0.0005, 0.0001, 0.00001,0.000001], 'kernel': ['rbf']}
        clf = GridSearchCV(SVC(decision_function_shape='ovr',class_weight='balanced'), param_grid, cv=5, scoring='roc_auc') 
        clf_train=clf.fit(X_train, y_train)
        Training_score=clf_train.score(X_train, y_train)
        Validation_score=clf_train.best_score_
        #classification_performance=clf_train.score(X_test, y_test)
    return Training_score,Validation_score

#Calculates the validation and test performance for all the discriminations
#(all responsive probe pairs)
def Discrimination_performance_array(probes_pairs,stim,dataset,estimator_type):
    performance_array=np.zeros(len(probes_pairs))
    for i in np.arange(len(probes_pairs)):
        print(i)
        probe1, probe2 = probes_pairs[i,0], probes_pairs[i,1]
        X_train, X_test, y_train, y_test = train_test_generator(probe1, probe2, stim, dataset)
        Training_score,Validation_score=Response_discriminator(X_train, X_test, y_train, y_test,estimator_type)
        performance_array[i]=Validation_score
    return performance_array


######### Mean relationship

#This function takes two corresponding arrays and calculates mean of y versus
#x averaged over specific range of x
def mean_relationship(x,y,bins_values):
    sort_ind_x=np.argsort(x)
    x=x[sort_ind_x]
    y=y[sort_ind_x]
    hist, bin_edges=np.histogram(x,bins=bins_values)
    array_end=np.cumsum(hist)
    array_start=np.cumsum(hist)-hist
    y_x=np.zeros(len(array_start))
    y_x_std=np.zeros(len(array_start))
    for i in np.arange(len(array_start)):
        y_x[i]=np.mean(y[array_start[i]:array_end[i]])
        y_x_std[i]=np.std(y[array_start[i]:array_end[i]])
    return y_x,y_x_std

def mean_relationship_twoD(x,y,bins_values):
    sort_ind_x=np.argsort(x)
    x=x[sort_ind_x]
    y=y[:,sort_ind_x]
    hist, bin_edges=np.histogram(x,bins=bins_values)
    array_end=np.cumsum(hist)
    array_start=np.cumsum(hist)-hist
    y_x=np.zeros((len(y),len(array_start)))
    for i in np.arange(len(array_start)):
        y_x[:,i]=np.mean(y[:,array_start[i]:array_end[i]],axis=1)
    return y_x

######### Plots discrimination performance vs cortical distance

#Asymptotic exponential function
def asymptotic_function(x,a):
    return 1-np.exp(-a*x)

#This function fits a curve for discrimination performance in terms of cortical distance
def Performance_corticaldistance_fit(cortical_distances,performance_array):
    popt,pcov = curve_fit(asymptotic_function,cortical_distances,performance_array)
    fitted_performance=asymptotic_function(cortical_distances,*popt)
    return fitted_performance,popt

#Plot the discrimination performance at different cortical distances
#and a function fitted to estimate the performance
#function y=1-exp(-ax) was fitted to the performance data as a function of cortical distance and least squares was used
#to find the optimized parameters.
def performance_corticaldistance_relationship(cortical_distances,performance_array,ylim_min,ylim_max,trace_color,file_name):
    sort_ind=np.argsort(cortical_distances)
    cortical_distances=cortical_distances[sort_ind]
    performance_array=performance_array[sort_ind]
    fitted_performance,popt=Performance_corticaldistance_fit(cortical_distances,performance_array)
    fig=plt.figure(figsize=(12, 8))
    plt.plot(cortical_distances, fitted_performance, linewidth=3.0, color=trace_color)
    plt.scatter(cortical_distances, performance_array, marker='o', c=trace_color)
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(ylim_min,ylim_max)
    plt.ylabel('Performance',fontweight='bold', fontsize=30)
    plt.xlabel('Cortical distance (mm)',fontweight='bold')
    plt.show()
    fig.savefig(file_name, dpi=350)
    return fitted_performance,popt


#Plot the discrimination performance at different cortical distances
#and give plot comparing wide,medium, and narrow response windows.
def performance_corticaldistance_plotter(cortical_distances_wide,cortical_distances_medium,cortical_distances_narrow,performance_array_wide,performance_array_medium,performance_array_narrow,trace_color_wide,trace_color_medium,trace_color_narrow,figure_title,file_name):
    fig=plt.figure(figsize=(12, 10))
    
    y_x_wide,y_x_std_wide=mean_relationship(cortical_distances_wide,performance_array_wide,np.arange(0.5,5.5,0.5))
    y_x_medium,y_x_std_medium=mean_relationship(cortical_distances_medium,performance_array_medium,np.arange(0.5,5.5,0.5))
    y_x_narrow,y_x_std_narrow=mean_relationship(cortical_distances_narrow,performance_array_narrow,np.arange(0.5,5.5,0.5))
    
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_wide,marker='o',linewidth=3.0,color=trace_color_wide)
    #plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_wide-y_x_std_wide,y_x_wide+y_x_std_wide,color=trace_color_wide)
    
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_medium,marker='o',linewidth=3.0,color=trace_color_medium)
    #plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_medium-y_x_std_medium,y_x_medium+y_x_std_medium,color=trace_color_medium)
    
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_narrow,marker='o',linewidth=3.0,color=trace_color_narrow)
    #plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_narrow-y_x_std_narrow,y_x_narrow+y_x_std_narrow,color=trace_color_narrow)
    
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3,1)
    plt.ylabel('Performance',fontweight='bold')
    plt.xlabel('Cortical distance (mm)',fontweight='bold')
    plt.title(figure_title,fontweight='bold',loc='center')
    plt.legend(('Wide','Medium','Narrow'),loc='lower right',fontsize=32)
    plt.show()
    fig.savefig(file_name, dpi=200)

######### Plots discrimination performance vs mean eccentricity of probes with specific separation

#linear function
def linear_function(x,a,b):
    return a*x+b


#This function fits a linear curve for discrimination performance in terms of mean eccentricity
#for 4 degrees separation
def Performance_ecc_fit(ecc_values,performance_array_values):
    popt,pcov = curve_fit(linear_function,ecc_values,performance_array_values)
    fitted_performance=linear_function(ecc_values,*popt)
    return fitted_performance,popt


#Plot the discrimination performance as a function of probes mean eccentricity for probes with specific separation
#it fits a linear function to estimate the performance vs eccentricity and calculates the linear correlation p-value
def Performance_eccentricity_relationship(distance_array,ecc_pairs,performance_array,dist_value,trace_color,ylim_min,ylim_max,file_name):
    ecc_values=ecc_pairs[distance_array==dist_value]
    performance_array_values=performance_array[distance_array==dist_value]
    sort_in=np.argsort(ecc_values)
    ecc_values=ecc_values[sort_in]
    performance_array_values=performance_array_values[sort_in]
    coeff,p_value=pearsonr(ecc_values,performance_array_values)
    fitted_performance,popt=Performance_ecc_fit(ecc_values,performance_array_values)
    fig=plt.figure(figsize=(12, 8))
    plt.scatter(ecc_values,performance_array_values,marker='o', c=trace_color)
    plt.plot(ecc_values,fitted_performance,linewidth=3.0, color=trace_color)
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(ylim_min,ylim_max)
    plt.ylabel('Performance',fontweight='bold', fontsize=30)
    plt.xlabel('Eccentricity (deg)',fontweight='bold')
    plt.text(28,0.94,'p-value={0}'.format(np.round_(p_value,4)),fontsize=20)
    plt.show()
    fig.savefig(file_name, dpi=350)
    return p_value,fitted_performance,popt

#Plot the discrimination performance as a function of probes mean eccentricity for probes with specific separation
#It calculates the linear correlation p-value between the performance and eccentricity
def Performance_eccentricity_plotter(distance_array_wide,distance_array_medium,distance_array_narrow,ecc_pairs_wide,ecc_pairs_medium,ecc_pairs_narrow,performance_array_wide,performance_array_medium,performance_array_narrow,dist_value,trace_color_wide,trace_color_medium,trace_color_narrow,figure_title,file_name):
    
    ecc_values_wide=ecc_pairs_wide[distance_array_wide==dist_value]
    performance_array_values_wide=performance_array_wide[distance_array_wide==dist_value]
    ecc_values_medium=ecc_pairs_medium[distance_array_medium==dist_value]
    performance_array_values_medium=performance_array_medium[distance_array_medium==dist_value]
    ecc_values_narrow=ecc_pairs_narrow[distance_array_narrow==dist_value]
    performance_array_values_narrow=performance_array_narrow[distance_array_narrow==dist_value]
    
    y_x_wide,y_x_std_wide=mean_relationship(ecc_values_wide,performance_array_values_wide,np.arange(3,27,3))
    y_x_medium,y_x_std_medium=mean_relationship(ecc_values_medium,performance_array_values_medium,np.arange(3,27,3))
    y_x_narrow,y_x_std_narrow=mean_relationship(ecc_values_narrow,performance_array_values_narrow,np.arange(3,27,3))
    
    coeff,p_value_wide=pearsonr(ecc_values_wide,performance_array_values_wide)
    coeff,p_value_medium=pearsonr(ecc_values_medium,performance_array_values_medium)
    coeff,p_value_narrow=pearsonr(ecc_values_narrow,performance_array_values_narrow)
    
    fig=plt.figure(figsize=(12, 10))
    
    plt.plot(np.arange(3,24,3)+1.5,y_x_wide,linewidth=3.0,marker='o',color=trace_color_wide)
    plt.plot(np.arange(3,24,3)+1.5,y_x_medium,linewidth=3.0,marker='o',color=trace_color_medium)
    plt.plot(np.arange(3,24,3)+1.5,y_x_narrow,linewidth=3.0,marker='o',color=trace_color_narrow)
    
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3,1)
    plt.ylabel('Performance',fontweight='bold')
    plt.xlabel('Eccentricity (deg)',fontweight='bold')
    plt.legend(('Wide p-value={0}'.format(np.round_(p_value_wide,4)),'Medium p-value={0}' .format(np.round_(p_value_medium,4)),'Narrow p-value={0}' .format(np.round_(p_value_narrow,4))),loc='lower right',fontsize=32)
    plt.title(figure_title,fontweight='bold',loc='center')
    plt.show()
    fig.savefig(file_name, dpi=200)
    

######### Clustering receptive fields

#This function for each given response dataset, clusters the responsive electrodes
#into no_cluster groups and takes cat_no-th group to produce output.
def RF_cluster(stim,dataset,responsive_electrodes,no_cluster,cat_no):
    averaged_responses=stimulus_averaged_responses(stim,dataset)
    clustering_labels = KMeans(init='k-means++',n_clusters=no_cluster, n_init=15, random_state=0).fit_predict(averaged_responses[:,responsive_electrodes].T)
    cat=responsive_electrodes[np.ravel(np.argwhere(clustering_labels==cat_no))]
    return cat

#Select 1-3 electrodes with the highest RF peak
def RF_representative_electrode(cat,stim,dataset):
    averaged_responses=stimulus_averaged_responses(stim,dataset)
    RF_peak=np.amax(averaged_responses,axis=0)
    sorted_cat=cat[np.argsort(RF_peak[cat])]
    sorted_cat=sorted_cat[::-1]
    three_highest_peak=sorted_cat[0:3]
    return three_highest_peak

def performance_corticaldistance_plotter_clustering(cortical_distances,performance_array,performance_array_clustering,trace_color,trace_color_clustering,figure_title,file_name):
    
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
    

######### Weight Analysis

#Determines linear SVM weights for a binary classification
def linear_svc_weights(X_train, X_test, y_train, y_test):
    param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12,50, 100,500, 1000]}
    clf = GridSearchCV(LinearSVC(penalty='l2',loss='squared_hinge',multi_class='ovr',class_weight='balanced'), param_grid, cv=5, scoring='roc_auc') 
    clf_train=clf.fit(X_train, y_train)
    SVMweights=clf_train.best_estimator_.coef_
    return SVMweights


#Weights of all the discriminations: Each row is a discrimination and 
#includes all the weights assigned to the electrodes for that discrimination
def SVM_weight_set(probes_pairs,stim,dataset):
    weight_set=np.zeros((len(probes_pairs),dataset.shape[1]))
    for i in np.arange(len(probes_pairs)):
        probe1, probe2 = probes_pairs[i,0], probes_pairs[i,1]
        X_train, X_test, y_train, y_test = train_test_generator(probe1, probe2, stim, dataset)
        SVMweights=linear_svc_weights(X_train, X_test, y_train, y_test)
        weight_set[i]=SVMweights
    return weight_set 


#Sort the electrodes according to the trial-averaged squared weights
#Values are sorted from the best to the worst electrodes
def electrodes_sorting_by_weight(weight_set,responsive_electrodes):
    weight_set=preprocessing.normalize(weight_set)
    electrodes_Importance=np.mean(weight_set**2,axis=0)
    electrode_rank=np.argsort(electrodes_Importance)
    electrode_rank=electrode_rank[::-1]
    weightsorted_electrodes=responsive_electrodes[electrode_rank]
    return weightsorted_electrodes

def electrodes_importance_calculator(weight_set):
    weight_set=preprocessing.normalize(weight_set)
    electrodes_Importance=np.mean(weight_set**2,axis=0)
    return electrodes_Importance

def electrode_pairs_importance(electrode_pairs,electrodes_importance,responsive_electrodes):
    pair_mean_importance=np.zeros(len(electrode_pairs))
    for i in np.arange(len(electrode_pairs)):
        pair_mean_importance[i]=(electrodes_importance[responsive_electrodes==electrode_pairs[i,0]]+electrodes_importance[responsive_electrodes==electrode_pairs[i,1]])/2
    return pair_mean_importance


def performance_corticaldistance_plotter_importance(cortical_distances,performance_array,performance_array_four,performance_array_six,performance_array_eight,performance_array_ten,performance_array_twelve,file_name):
    
    y_x,y_x_std=mean_relationship(cortical_distances,performance_array,np.arange(0.5,5.5,0.5))
    y_x_four,y_x_std_four=mean_relationship(cortical_distances,performance_array_four,np.arange(0.5,5.5,0.5))
    y_x_six,y_x_std_six=mean_relationship(cortical_distances,performance_array_six,np.arange(0.5,5.5,0.5))
    y_x_eight,y_x_std_eight=mean_relationship(cortical_distances,performance_array_eight,np.arange(0.5,5.5,0.5))
    y_x_ten,y_x_std_ten=mean_relationship(cortical_distances,performance_array_ten,np.arange(0.5,5.5,0.5))
    y_x_twelve,y_x_std_twelve=mean_relationship(cortical_distances,performance_array_twelve,np.arange(0.5,5.5,0.5))
    
    
    fig=plt.figure(figsize=(12, 10))
    
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x,marker='o',linewidth=3.0,color=(0.031, 0.031, 0.027))
    #plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x-y_x_std,y_x+y_x_std,color=(0.031, 0.031, 0.027))
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_four,marker='o',linewidth=3.0,color=(0.054, 0.925, 0.964))
    #plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_four-y_x_std_four,y_x_four+y_x_std_four,color=(0.054, 0.925, 0.964))
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_six,marker='o',linewidth=3.0,color=(0.427, 0.964, 0.054))
    #plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_six-y_x_std_six,y_x_six+y_x_std_six,color=(0.427, 0.964, 0.054))
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_eight,marker='o',linewidth=3.0,color=(0.964, 0.937, 0.054))
    #plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_eight-y_x_std_eight,y_x_eight+y_x_std_eight,color=(0.964, 0.937, 0.054))
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_ten,marker='o',linewidth=3.0,color=(0.964, 0.054, 0.780))
    #plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_ten-y_x_std_ten,y_x_ten+y_x_std_ten,color=(0.964, 0.054, 0.780))
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_twelve,marker='o',linewidth=3.0,color=(0.964, 0.098, 0.054))
    #plt.fill_between(np.arange(0.5,5,0.5)+0.25,y_x_twelve-y_x_std_twelve,y_x_twelve+y_x_std_twelve,color=(0.964, 0.098, 0.054))
    
    
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3,1)
    plt.ylabel('Performance',fontweight='bold')
    plt.xlabel('Cortical distance (mm)',fontweight='bold')
    plt.legend(('All','4','6','8','10','12'),loc='lower right',fontsize=32)
    
    plt.show()
    fig.savefig(file_name, dpi=200)



def Performance_eccentricity_plotter_importance(distance_array,ecc_pairs,performance_array,performance_array_four,performance_array_six,performance_array_eight,performance_array_ten,performance_array_twelve,dist_value,figure_title,file_name):
    
    ecc_values=ecc_pairs[distance_array==dist_value]
    
    performance_array_values=performance_array[distance_array==dist_value]
    performance_array_values_four=performance_array_four[distance_array==dist_value]
    performance_array_values_six=performance_array_six[distance_array==dist_value]
    performance_array_values_eight=performance_array_eight[distance_array==dist_value]
    performance_array_values_ten=performance_array_ten[distance_array==dist_value]
    performance_array_values_twelve=performance_array_twelve[distance_array==dist_value]
    
    y_x,y_x_std=mean_relationship(ecc_values,performance_array_values,np.arange(3,27,3))
    y_x_four,y_x_std_four=mean_relationship(ecc_values,performance_array_values_four,np.arange(3,27,3))
    y_x_six,y_x_std_six=mean_relationship(ecc_values,performance_array_values_six,np.arange(3,27,3))
    y_x_eight,y_x_std_eight=mean_relationship(ecc_values,performance_array_values_eight,np.arange(3,27,3))
    y_x_ten,y_x_std_ten=mean_relationship(ecc_values,performance_array_values_ten,np.arange(3,27,3))
    y_x_twelve,y_x_std_twelve=mean_relationship(ecc_values,performance_array_values_twelve,np.arange(3,27,3))
    
    
    coeff,p_value=pearsonr(ecc_values,performance_array_values)
    coeff,p_value_four=pearsonr(ecc_values,performance_array_values_four)
    coeff,p_value_six=pearsonr(ecc_values,performance_array_values_six)
    coeff,p_value_eight=pearsonr(ecc_values,performance_array_values_eight)
    coeff,p_value_ten=pearsonr(ecc_values,performance_array_values_ten)
    coeff,p_value_twelve=pearsonr(ecc_values,performance_array_values_twelve)
    
    
    fig=plt.figure(figsize=(12, 10))
    
    
    plt.plot(np.arange(3,24,3)+1.5,y_x,linewidth=3.0,marker='o',color=(0.031, 0.031, 0.027))
    plt.plot(np.arange(3,24,3)+1.5,y_x_four,linewidth=3.0,marker='o',color=(0.054, 0.925, 0.964))
    plt.plot(np.arange(3,24,3)+1.5,y_x_six,linewidth=3.0,marker='o',color=(0.427, 0.964, 0.054))
    plt.plot(np.arange(3,24,3)+1.5,y_x_eight,linewidth=3.0,marker='o',color=(0.964, 0.937, 0.054))
    plt.plot(np.arange(3,24,3)+1.5,y_x_ten,linewidth=3.0,marker='o',color=(0.964, 0.054, 0.780))
    plt.plot(np.arange(3,24,3)+1.5,y_x_twelve,linewidth=3.0,marker='o',color=(0.964, 0.098, 0.054))
    
    
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3,1)
    plt.ylabel('Performance',fontweight='bold')
    plt.xlabel('Eccentricity (deg)',fontweight='bold')
    plt.legend(('All p-value={0}'.format(np.round_(p_value,4)),'4 p-value={0}' .format(np.round_(p_value_four,4)),'6 p-value={0}'.format(np.round_(p_value_six,4)),'8 p-value={0}'.format(np.round_(p_value_eight,4)),'10 p-value={0}'.format(np.round_(p_value_ten,4)),'12 p-value={0}'.format(np.round_(p_value_twelve,4))),loc='lower right',fontsize=26)
    plt.title(figure_title,fontweight='bold',loc='center')
    
    plt.show()
    fig.savefig(file_name, dpi=200)
    
def Electrode_array_show(electrode_set,plxarray,filename):
    Array_=np.zeros(100)
    Array_[:]=np.nan
    Array_[electrode_set]=1
    fig=plt.figure(figsize=(3,3))
    plt.imshow(Array_[plxarray-1])
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlabel('Electrodes',fontsize=26,fontweight='bold')
    plt.ylabel('Electrodes',fontsize=26,fontweight='bold')
    plt.show()
    fig.savefig(filename, dpi=350)

def Electrodes_importance_histogram(weight_set,bar_color,filename):
    weight_set_normalized=preprocessing.normalize(weight_set)
    Importance_all=np.mean(weight_set_normalized**2,axis=0)
    mean_importance=np.mean(Importance_all)
    mean_importance=np.round_(mean_importance,4)
    std_importance=np.std(Importance_all)
    std_importance=np.round_(std_importance,4)
    fig=plt.figure(figsize=(9, 9))
    plt.hist(Importance_all,bins=np.arange(0,0.105,0.005),color=bar_color,edgecolor='black')
    plt.xlabel('Importance value',fontweight='bold')
    plt.ylabel('Number of electrodes',fontweight='bold')
    plt.ylim(0,40)
    plt.title('Mean=%.4f,SD=%.4f'%(mean_importance,std_importance),fontweight='bold',loc='center',fontsize=32)
    plt.show()
    fig.savefig(filename, dpi=200)

def Electrode_array_importance(weight_set,responsive_electrodes,plxarray,v_min,v_max,filename):
    weight_set_normalized=preprocessing.normalize(weight_set)
    Importance_all=np.mean(weight_set_normalized**2,axis=0)
    Aray_response=np.zeros(100)
    Aray_response[:]=np.nan
    Aray_response[responsive_electrodes]=Importance_all
    fig=plt.figure(figsize=(5,5))
    plt.imshow(Aray_response[plxarray-1],vmin=v_min, vmax=v_max)
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlabel('Electrodes',fontweight='bold')
    plt.ylabel('Electrodes',fontweight='bold')
    plt.subplots_adjust()
    cax = plt.axes([0.95, 0.13, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min,v_max])
    plt.show()
    fig.savefig(filename, dpi=200)

def probes_weight(weight_set,electrode,responsive_electrodes,probe_pairs,dist_value,distance_array,filename):
    weight_set=preprocessing.normalize(weight_set)
    electrode=np.ravel(np.argwhere(responsive_electrodes==electrode))[0]
    electrode_weights=weight_set[:,electrode]
    probes_weights=np.zeros(100)
    probes_weights[:]=np.nan
    for i in np.unique(probe_pairs[:,0]):
        probes_weights[i-1]=-np.mean(electrode_weights[(probe_pairs[:,0]==i)&(distance_array==dist_value)])
    v_min=np.nanmin(probes_weights)
    v_max=np.nanmax(probes_weights)
    fig=plt.figure(figsize=(5,5))
    plt.imshow(np.reshape(probes_weights,(10,10)).T)
    plt.xticks(np.array([0,9]),[-36,0])
    plt.yticks(np.array([0,9]),[2,-34])
    plt.xlabel('Position(deg)',fontweight='bold')
    plt.ylabel('Position(deg)',fontweight='bold')
    plt.subplots_adjust()
    cax = plt.axes([1, 0.23, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min,v_max])
    plt.show()
    fig.savefig(filename, dpi=150)

def weight_corticaldist(weight_set,cortical_distances,responsive_electrodes,electrode):
    weight_set=preprocessing.normalize(weight_set)
    electrode=np.ravel(np.argwhere(responsive_electrodes==electrode))[0]
    electrode_weights=weight_set[:,electrode]
    coeff,p_value=pearsonr(cortical_distances,electrode_weights)
    return p_value

def mean_importance_corticaldist(weight_set,cortical_distances,responsive_electrodes,electrode1,electrode2,electrode3,electrode4):
    weight_set=preprocessing.normalize(weight_set)
    importance_set=weight_set**2
    electrode1=np.ravel(np.argwhere(responsive_electrodes==electrode1))[0]
    electrode2=np.ravel(np.argwhere(responsive_electrodes==electrode2))[0]
    electrode3=np.ravel(np.argwhere(responsive_electrodes==electrode3))[0]
    electrode4=np.ravel(np.argwhere(responsive_electrodes==electrode4))[0]
    electrode_importance1=importance_set[:,electrode1]
    electrode_importance2=importance_set[:,electrode2]
    electrode_importance3=importance_set[:,electrode3]
    electrode_importance4=importance_set[:,electrode4]
    electrode_importance=0.25*(electrode_importance1+electrode_importance2+electrode_importance3+electrode_importance4)
    coeff,p_value=pearsonr(cortical_distances,electrode_importance)   
    return p_value

######### bandpass LFP

def bandpass_LFP(arr,lowcut,highcut):
    fs=500
    nyq = 0.5 * fs
    order=4
    low = lowcut / nyq
    high = highcut / nyq
    w_low, w_high = butter(order, [low, high], btype='band')
    arr_f = lfilter(w_low, w_high, arr)
    return arr_f

def bandpass_performance_corticaldistance_plotter(cortical_distances,performance_array_theta,performance_array_alpha,performance_array_beta,performance_array_gamma,performance_array_highgamma,figure_title,file_name):
    fig=plt.figure(figsize=(12, 10))
    
    y_x_theta,y_x_std_theta=mean_relationship(cortical_distances,performance_array_theta,np.arange(0.5,5.5,0.5))
    y_x_alpha,y_x_std_alpha=mean_relationship(cortical_distances,performance_array_alpha,np.arange(0.5,5.5,0.5))
    y_x_beta,y_x_std_beta=mean_relationship(cortical_distances,performance_array_beta,np.arange(0.5,5.5,0.5))
    y_x_gamma,y_x_std_gamma=mean_relationship(cortical_distances,performance_array_gamma,np.arange(0.5,5.5,0.5))
    y_x_highgamma,y_x_std_highgamma=mean_relationship(cortical_distances,performance_array_highgamma,np.arange(0.5,5.5,0.5))
    
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_theta,marker='o',linewidth=3.0,color=(0.035, 0.062, 0.682))
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_alpha,marker='o',linewidth=3.0,color=(0.298, 0.662, 0.941))
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_beta,marker='o',linewidth=3.0,color=(0.031, 0.568, 0.098))
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_gamma,marker='o',linewidth=3.0,color=(0.960, 0.050, 0.019))
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_highgamma,marker='o',linewidth=3.0,color=(0.960, 0.454, 0.019))
    
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3,1)
    plt.ylabel('Performance',fontweight='bold')
    plt.xlabel('Cortical distance (mm)',fontweight='bold')
    plt.title(figure_title,fontweight='bold',loc='center')
    plt.legend(('Theta','Alpha','Beta','Gamma','High-gamma'),loc='upper right',fontsize=28)
    plt.show()
    fig.savefig(file_name, dpi=200)

def Bandpass_Performance_eccentricity_plotter(distance_array,ecc_pairs,performance_array_theta,performance_array_alpha,performance_array_beta,performance_array_gamma,performance_array_highgamma,dist_value,figure_title,file_name):
    
    ecc_values=ecc_pairs[distance_array==dist_value]
    performance_array_values_theta=performance_array_theta[distance_array==dist_value]
    performance_array_values_alpha=performance_array_alpha[distance_array==dist_value]
    performance_array_values_beta=performance_array_beta[distance_array==dist_value]
    performance_array_values_gamma=performance_array_gamma[distance_array==dist_value]
    performance_array_values_highgamma=performance_array_highgamma[distance_array==dist_value]
    
    
    y_x_theta,y_x_std_theta=mean_relationship(ecc_values,performance_array_values_theta,np.arange(3,27,3))
    y_x_alpha,y_x_std_alpha=mean_relationship(ecc_values,performance_array_values_alpha,np.arange(3,27,3))
    y_x_beta,y_x_std_beta=mean_relationship(ecc_values,performance_array_values_beta,np.arange(3,27,3))
    y_x_gamma,y_x_std_gamma=mean_relationship(ecc_values,performance_array_values_gamma,np.arange(3,27,3))
    y_x_highgamma,y_x_std_highgamma=mean_relationship(ecc_values,performance_array_values_highgamma,np.arange(3,27,3))
    
    
    coeff,p_value_theta=pearsonr(ecc_values,performance_array_values_theta)
    coeff,p_value_alpha=pearsonr(ecc_values,performance_array_values_alpha)
    coeff,p_value_beta=pearsonr(ecc_values,performance_array_values_beta)
    coeff,p_value_gamma=pearsonr(ecc_values,performance_array_values_gamma)
    coeff,p_value_highgamma=pearsonr(ecc_values,performance_array_values_highgamma)
    
    
    fig=plt.figure(figsize=(12, 10))
    
    plt.plot(np.arange(3,24,3)+1.5,y_x_theta,linewidth=3.0,marker='o',color=(0.035, 0.062, 0.682))
    plt.plot(np.arange(3,24,3)+1.5,y_x_alpha,linewidth=3.0,marker='o',color=(0.298, 0.662, 0.941))
    plt.plot(np.arange(3,24,3)+1.5,y_x_beta,linewidth=3.0,marker='o',color=(0.031, 0.568, 0.098))
    plt.plot(np.arange(3,24,3)+1.5,y_x_gamma,linewidth=3.0,marker='o',color=(0.960, 0.050, 0.019))
    plt.plot(np.arange(3,24,3)+1.5,y_x_highgamma,linewidth=3.0,marker='o',color=(0.960, 0.454, 0.019))
    
    
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3,1)
    plt.ylabel('Performance',fontweight='bold')
    plt.xlabel('Eccentricity (deg)',fontweight='bold')
    plt.legend(('Theta p-value={0}'.format(np.round_(p_value_theta,4)),'Alpha p-value={0}' .format(np.round_(p_value_alpha,4)),'Beta p-value={0}' .format(np.round_(p_value_beta,4)),'Gamma p-value={0}' .format(np.round_(p_value_gamma,4)),'High-gamma p-value={0}' .format(np.round_(p_value_highgamma,4))),loc='upper right',fontsize=26)
    plt.title(figure_title,fontweight='bold',loc='center')
    plt.show()
    fig.savefig(file_name, dpi=200)
    

######### Correlation-blind performance

def Discrimination_performance_array_onblind(probes_pairs,stim,dataset):
    
    performance_array=np.zeros(len(probes_pairs))
    for i in np.arange(len(probes_pairs)):
        print(i)
        probe1, probe2 = probes_pairs[i,0], probes_pairs[i,1]
        
        X_clf, X_test, y_clf, y_test=train_test_generator(probe1,probe2,stim,dataset)
        
        skf = StratifiedKFold(n_splits=5)
        classification_performance=0
        for train_index, test_index in skf.split(X_clf, y_clf):
           X_train, X_test = X_clf[train_index], X_clf[test_index]
           y_train, y_test = y_clf[train_index], y_clf[test_index]
           
           X_train0=shuffler(X_train[y_train==0])
           X_train1=shuffler(X_train[y_train==1])
           
           X_train=np.vstack((X_train0,X_train1))
           y_train=np.hstack((np.zeros((len(X_train0)), dtype=int),np.ones((len(X_train1)), dtype=int)))
           
           order=np.argsort(np.random.random(len(X_train)))
           X_train=X_train[order]
           y_train=y_train[order]
           
           param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12,50, 100,500, 1000]}
           clf = GridSearchCV(LinearSVC(penalty='l2',loss='squared_hinge',multi_class='ovr',class_weight='balanced'), param_grid, cv=5, scoring='roc_auc') 
           clf_train=clf.fit(X_train, y_train)
           classification_performance=classification_performance+clf_train.score(X_test, y_test)
           
        classification_performance=classification_performance/5
        performance_array[i]=classification_performance
        
    return performance_array

def shuffler(dataset):
    shuffled_dataset=np.zeros((dataset.shape))
    for j in np.arange(dataset.shape[1]):
        order = np.argsort(np.random.random(len(dataset)))
        shuffled_dataset[:,j]=dataset[:,j][order]
    return shuffled_dataset

def performance_corticaldistance_onblind_plotter(cortical_distances,performance_array,performance_array_blind,trace_color,trace_color_blind,figure_title,file_name):
    
    fig=plt.figure(figsize=(12, 10))
    
    y_x,y_x_std=mean_relationship(cortical_distances,performance_array,np.arange(0.5,5.5,0.5))
    y_x_blind,y_x_std_blind=mean_relationship(cortical_distances,performance_array_blind,np.arange(0.5,5.5,0.5))
    
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x,marker='o',linewidth=3.0,color=trace_color)
    plt.plot(np.arange(0.5,5,0.5)+0.25,y_x_blind,marker='o',linewidth=3.0,color=trace_color_blind)
    
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3,1)
    plt.ylabel('Performance',fontweight='bold')
    plt.xlabel('Cortical distance (mm)',fontweight='bold')
    plt.title(figure_title,fontweight='bold',loc='center')
    plt.legend(('Correlation-aware','Correlation-blind'),loc='lower right',fontsize=32)
    plt.show()
    fig.savefig(file_name, dpi=200)

def Performance_eccentricity_onblind_plotter(distance_array,ecc_pairs,performance_array,performance_array_blind,dist_value,trace_color,trace_color_blind,figure_title,file_name):
    
    ecc_values=ecc_pairs[distance_array==dist_value]
    performance_array_values=performance_array[distance_array==dist_value]
    performance_array_values_blind=performance_array_blind[distance_array==dist_value]
    
    y_x,y_x_std=mean_relationship(ecc_values,performance_array_values,np.arange(3,27,3))
    y_x_blind,y_x_std_blind=mean_relationship(ecc_values,performance_array_values_blind,np.arange(3,27,3))
    
    coeff,p_value=pearsonr(ecc_values,performance_array_values)
    coeff,p_value_blind=pearsonr(ecc_values,performance_array_values_blind)
    
    fig=plt.figure(figsize=(12, 10))
    
    plt.plot(np.arange(3,24,3)+1.5,y_x,linewidth=3.0,marker='o',color=trace_color)
    plt.plot(np.arange(3,24,3)+1.5,y_x_blind,linewidth=3.0,marker='o',color=trace_color_blind)
    
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0.3,1)
    plt.ylabel('Performance',fontweight='bold')
    plt.xlabel('Eccentricity (deg)',fontweight='bold')
    plt.legend(('Correlation-aware p-value={0}'.format(np.round_(p_value,4)),'Correlation-blind p-value={0}' .format(np.round_(p_value_blind,4))),loc='lower right',fontsize=28)
    plt.title(figure_title,fontweight='bold',loc='center')
    plt.show()
    fig.savefig(file_name, dpi=200)
    
    
######### Noise correlation analysis

###Noise correlation

#this function creates the response set for repeated presentation of a
#particular stimulus. The responses are standardized.
def MUA_noise_correlation_dataset(MUA,stim,probe):
    arr_im = MUA[stim==probe]
    noise_dataset = np.sum(arr_im[:,:,16:22], axis=2)
    noise_dataset = preprocessing.StandardScaler().fit_transform(noise_dataset)
    return noise_dataset

def LFP_noise_correlation_dataset(LFP,stim,probe):
    arr_im = LFP[stim==probe]
    noise_dataset = np.sum(arr_im[:,:,200:225], axis=2)
    noise_dataset = preprocessing.StandardScaler().fit_transform(noise_dataset)
    return noise_dataset

#This function calculates Pearson correlation coefficient between 
#two electrodes. Since noise_data is standardized, this function determines 
#the outliers by comparing the absolute values to 3 and removes those trials
# for which the responses of either electrodes is an outlier.
def noise_correlation(noise_dataset,electrode1,electrode2):
    pair_dataset=noise_dataset[:,np.array([electrode1,electrode2])]
    outresponses=np.argwhere(np.absolute(pair_dataset)>3)
    out_trials=np.unique(outresponses[:,0])
    pair_dataset=np.delete(pair_dataset,out_trials,0)
    nc_value,p_val=pearsonr(pair_dataset[:,0],pair_dataset[:,1])
    return nc_value

#This function calculates noise correlation for all the pairs of electrodes
#for a particular stimulus
def MUA_all_pairs_noise_correlation(MUA,stim,probe,electrode_set):
    noise_dataset=MUA_noise_correlation_dataset(MUA,stim,probe)
    electrode_pairs = list(itertools.combinations(list(electrode_set),2))
    electrode_pairs = np.array(electrode_pairs,dtype=int)
    
    nc_array=np.zeros(len(electrode_pairs))
    for i in np.arange(len(electrode_pairs)):
        print(i)
        electrode1,electrode2=electrode_pairs[i,0],electrode_pairs[i,1] 
        nc_array[i]=noise_correlation(noise_dataset,electrode1,electrode2)
    return nc_array


def LFP_all_pairs_noise_correlation(LFP,stim,probe,electrode_set):
    noise_dataset=LFP_noise_correlation_dataset(LFP,stim,probe)
    electrode_pairs = list(itertools.combinations(list(electrode_set),2))
    electrode_pairs = np.array(electrode_pairs,dtype=int)
    
    nc_array=np.zeros(len(electrode_pairs))
    for i in np.arange(len(electrode_pairs)):
        print(i)
        electrode1,electrode2=electrode_pairs[i,0],electrode_pairs[i,1]
        nc_array[i]=noise_correlation(noise_dataset,electrode1,electrode2)
    return nc_array


#This function gives noise correlation values for all electrode pairs
#for all the presented stimuli
def MUA_all_probes_all_pairs_noise_correlation(MUA,stim,electrode_set,MUA_responsive_probes):
    electrode_pairs = list(itertools.combinations(list(electrode_set),2))
    electrode_pairs = np.array(electrode_pairs,dtype=int)
    nc_array=np.zeros((len(np.unique(stim)),len(electrode_pairs)))
    nc_array[:]=np.nan
    for probe in MUA_responsive_probes:
        nc_array[probe-1,:]=MUA_all_pairs_noise_correlation(MUA,stim,probe,electrode_set)
    return nc_array

def LFP_all_probes_all_pairs_noise_correlation(LFP,stim,electrode_set,LFP_responsive_probes):
    electrode_pairs = list(itertools.combinations(list(electrode_set),2))
    electrode_pairs = np.array(electrode_pairs,dtype=int)
    nc_array=np.zeros((len(np.unique(stim)),len(electrode_pairs)))
    nc_array[:]=np.nan
    for probe in LFP_responsive_probes:
        nc_array[probe-1,:]=LFP_all_pairs_noise_correlation(LFP,stim,probe,electrode_set)
    return nc_array

###Distances

#This function calculates the distance between two electrodes on the array
def electrode_dist(plxarray,electrode1,electrode2):
    plxarray=plxarray-1
    coord1=np.ravel(np.argwhere(plxarray==electrode1))
    coord2=np.ravel(np.argwhere(plxarray==electrode2))
    coord=coord1-coord2
    dist12=np.sqrt(coord[0]**2+coord[1]**2)
    return dist12

#This function calculates the distances between all the electrodes on
#a particular array
def all_pairs_electrodes_distances(plxarray,electrode_set):
    electrode_pairs = list(itertools.combinations(list(electrode_set),2))
    electrode_pairs = np.array(electrode_pairs,dtype=int)
    
    electrodes_dist_pairs=np.zeros(len(electrode_pairs))
    for i in np.arange(len(electrode_pairs)):
        print(i)
        electrode1,electrode2=electrode_pairs[i,0],electrode_pairs[i,1]
        electrodes_dist_pairs[i]=electrode_dist(plxarray,electrode1,electrode2)
    return 0.4*electrodes_dist_pairs


###plot noise correlation of probes

def Probes_noise_correlation_plotter(probes_nc,v_min,v_max,filename):
    probes_nc=np.reshape(probes_nc,(10,10)).T
    fig=plt.figure(figsize=(5, 5))
    plt.imshow(probes_nc,vmin=v_min, vmax=v_max)
    plt.xticks(np.array([0,9]),[-36,0])
    plt.yticks(np.array([0,9]),[2,-34])
    plt.xlabel('Position(deg)',fontweight='bold')
    plt.ylabel('Position(deg)',fontweight='bold')
    plt.subplots_adjust()
    cax = plt.axes([1, 0.23, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min,v_max])
    plt.show()
    fig.savefig(filename, dpi=150)

###plot noise correlation vs electrodes distances

def noisecorrelation_distance_plotter(noisecorrelation,electrodes_distance,y_lim_min,y_lim_max,figure_title,file_name):
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y_x=mean_relationship_twoD(electrodes_distance,noisecorrelation,np.arange(0,4.5,0.5))
    
    ax.plot(np.arange(0,4,0.5)+0.25,y_x[0],marker='o',linewidth=3.0,color=(0.419, 0.039, 0.741),label='Ecc<5 deg')
    ax.plot(np.arange(0,4,0.5)+0.25,y_x[1],marker='o',linewidth=3.0,color=(0.419, 0.039, 0.741))
    ax.plot(np.arange(0,4,0.5)+0.25,y_x[2],marker='o',linewidth=3.0,color=(0.741, 0.701, 0.039),label='8<Ecc<12')
    ax.plot(np.arange(0,4,0.5)+0.25,y_x[3],marker='o',linewidth=3.0,color=(0.741, 0.701, 0.039))
    ax.plot(np.arange(0,4,0.5)+0.25,y_x[4],marker='o',linewidth=3.0,color=(0.039, 0.741, 0.525),label='16<Ecc<20')
    ax.plot(np.arange(0,4,0.5)+0.25,y_x[5],marker='o',linewidth=3.0,color=(0.039, 0.741, 0.525))
    
    
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(y_lim_min,y_lim_max)
    plt.ylabel('Noise correlation',fontweight='bold')
    plt.xlabel('Electrodes distances (mm)',fontweight='bold')
    plt.title(figure_title,fontweight='bold',loc='center')
    ax.legend(loc='lower right',fontsize=28)
    plt.show()
    fig.savefig(file_name, dpi=200)

###plot noise correlation vs electrodes mean importance

def noisecorrelation_mean_importance_plotter(noisecorrelation,pair_mean_importance,bins_values,centre_points,y_lim_min,y_lim_max,figure_title,file_name):
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y_x=mean_relationship_twoD(pair_mean_importance,noisecorrelation,bins_values)
    
    ax.plot(centre_points,y_x[0],marker='o',linewidth=3.0,color=(0.419, 0.039, 0.741),label='Ecc<5 deg')
    ax.plot(centre_points,y_x[1],marker='o',linewidth=3.0,color=(0.419, 0.039, 0.741))
    ax.plot(centre_points,y_x[2],marker='o',linewidth=3.0,color=(0.741, 0.701, 0.039),label='8<Ecc<12')
    ax.plot(centre_points,y_x[3],marker='o',linewidth=3.0,color=(0.741, 0.701, 0.039))
    ax.plot(centre_points,y_x[4],marker='o',linewidth=3.0,color=(0.039, 0.741, 0.525),label='16<Ecc<20')
    ax.plot(centre_points,y_x[5],marker='o',linewidth=3.0,color=(0.039, 0.741, 0.525))
    
    
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(y_lim_min,y_lim_max)
    plt.ylabel('Noise correlation',fontweight='bold')
    plt.xlabel('Electrodes importance',fontweight='bold')
    plt.title(figure_title,fontweight='bold',loc='center')
    ax.legend(loc='lower right',fontsize=28)
    plt.show()
    fig.savefig(file_name, dpi=200)


###################### Main ##########################


######### Load data

#probe numbering on the grid starts from 1 for the probe at the upper left
#corner and increase to 10 to the bottom and continues to the lower right corner that is 100
#stim is an array that gives the corresponding probe number to each trial
#1769 trials was recorded and the label of each trial is in the stim and
#the corresponding recorded activity is in MUA and LFP.
#plxarray represents the location of each electrode on the array
#Edges are the time edges for MUA and LFP_Edges are the time samples for the LFPs

MUA=np.load('MUA.npy')
LFP=np.load('LFP.npy')
stim=np.load('stim.npy')
Edges=np.load('Edges.npy')
LFP_Edges=np.load('LFP_Edges.npy')
plxarray=np.load('plxarray.npy')

###################### Section 0: Visualizing ensemble responses and receptive fields

# a selected group of electrodes that respond well to stimuli for both
# MUA and LFP are selected
electrode_set=np.array([ 3,  4,  5,  7, 10, 12, 18, 19, 20, 21, 24, 26, 27, 29, 31, 37, 45,
       48, 49, 50, 54, 58, 60, 72, 75, 89, 90, 91], dtype=int)

######### Visualization of recordings

# selected probes are distributed over the grid 
Probe_set=[23,29,55,82,88]

# MUA spike count over time
edge_data=Edges+0.0125
ylab='Spikes'
ylimit_bottom,ylimit_top=-0.01,3.7
Signal_visualization(MUA,stim,Probe_set,electrode_set,edge_data,ylab,ylimit_bottom,ylimit_top,'r','Spike_count_traces.png')

# LFP activity over time
edge_data=LFP_Edges
ylab='LFP (\u03BC V)'
ylimit_bottom,ylimit_top=-300,300
Signal_visualization(LFP,stim,Probe_set,electrode_set,edge_data,ylab,ylimit_bottom,ylimit_top,'b','LFP_traces.png')

######### Building the response dataset and visualizing it

### MUA

# MUA response data are the z-scored responses of all the electrodes
#in all the trials. Response window=wide
MUA_dataset=MUA_Response_dataset(MUA,'w')

# Mean MUA responses over the electrode array for a set of probes
v_min=-0.27
v_max=0.27
Probe_set=[23,29,55,82,88]
Aray_response_visualization(stim,MUA_dataset,Probe_set,plxarray,v_min,v_max,'MUA_array_response_wide.png')

# MUA response data are the z-scored responses of all the electrodes
#in all the trials. Response window=medium
MUA_dataset=MUA_Response_dataset(MUA,'m')

# Mean MUA responses over the electrode array for a set of probes
v_min=-0.27
v_max=0.27
Probe_set=[23,29,55,82,88]
Aray_response_visualization(stim,MUA_dataset,Probe_set,plxarray,v_min,v_max,'MUA_array_response_medium.png')

# MUA response data are the z-scored responses of all the electrodes
#in all the trials. Response window=narrow
MUA_dataset=MUA_Response_dataset(MUA,'n')

# Mean MUA responses over the electrode array for a set of probes
v_min=-0.27
v_max=0.27
Probe_set=[23,29,55,82,88]
Aray_response_visualization(stim,MUA_dataset,Probe_set,plxarray,v_min,v_max,'MUA_array_response_narrow.png')


### LFP


# LFP response data are the z-scored responses of all the electrodes
#in all the trials. Response window=wide
LFP_dataset=LFP_Response_dataset(LFP,'w')

# Mean responses over the electrode array for a set of probes.
#Since the LFP responses are negative, a - was applied to the response values when 
#they are presented on the array

v_min=-0.34
v_max=0.34
Probe_set=[23,29,55,82,88]
Aray_response_visualization(stim,-LFP_dataset,Probe_set,plxarray,v_min,v_max,'LFP_array_response_wide.png')

# LFP response data are the z-scored responses of all the electrodes
#in all the trials. Response window=medium
LFP_dataset=LFP_Response_dataset(LFP,'m')

# Mean responses over the electrode array for a set of probes
#Since the LFP responses are negative, a - was applied to the response values when 
#they are presented on the array
v_min=-0.34
v_max=0.34
Probe_set=[23,29,55,82,88]
Aray_response_visualization(stim,-LFP_dataset,Probe_set,plxarray,v_min,v_max,'LFP_array_response_medium.png')

# LFP response data are the z-scored responses of all the electrodes
#in all the trials. Response window=narrow
LFP_dataset=LFP_Response_dataset(LFP,'n')

# Mean responses over the electrode array for a set of probes
#Since the LFP responses are negative, a - was applied to the response values when 
#they are presented on the array
v_min=-0.34
v_max=0.34
Probe_set=[23,29,55,82,88]
Aray_response_visualization(stim,-LFP_dataset,Probe_set,plxarray,v_min,v_max,'LFP_array_response_narrow.png')


######### Receptive fields

#Contour plot of MUA receptive fields for wide, medium, and narrow response windows
#The population_contour function also calculates the mean of long and short diameter
#of the fitted RF ellipses

MUA_dataset=MUA_Response_dataset(MUA,'w')
LFP_dataset=LFP_Response_dataset(LFP,'w')
fig=plt.figure(figsize=(12, 12))
RF_mean_diameter_MUA_wide=population_contours(MUA_dataset,stim,electrode_set,4,'r')
RF_mean_diameter_LFP_wide=population_contours(-LFP_dataset,stim,electrode_set,4,'b')
plt.show()
fig.savefig('RF_contours_wide.png', dpi=150)


MUA_dataset=MUA_Response_dataset(MUA,'m')
LFP_dataset=LFP_Response_dataset(LFP,'m')
fig=plt.figure(figsize=(12, 12))
RF_mean_diameter_MUA_medium=population_contours(MUA_dataset,stim,electrode_set,4,'r')
RF_mean_diameter_LFP_medium=population_contours(-LFP_dataset,stim,electrode_set,4,'b')
plt.show()
fig.savefig('RF_contours_medium.png', dpi=150)


MUA_dataset=MUA_Response_dataset(MUA,'n')
LFP_dataset=LFP_Response_dataset(LFP,'n')
fig=plt.figure(figsize=(12, 12))
RF_mean_diameter_MUA_narrow=population_contours(MUA_dataset,stim,electrode_set,4,'r')
RF_mean_diameter_LFP_narrow=population_contours(-LFP_dataset,stim,electrode_set,4,'b')
plt.show()
fig.savefig('RF_contours_narrow.png', dpi=150)


###################### Section 1: Spatial precision of MUA and LFP in area V4

######### Responsive electrodes

#For each response dataset which is MUA or LFP and wide, medium, or narrow window
#we determine the set of responsive electrodes

MUA_dataset=MUA_Response_dataset(MUA,'w')
MUA_responsive_electrodes_wide=Select_responsive_electrodes(MUA_dataset,stim)
MUA_dataset=MUA_Response_dataset(MUA,'m')
MUA_responsive_electrodes_medium=Select_responsive_electrodes(MUA_dataset,stim)
MUA_dataset=MUA_Response_dataset(MUA,'n')
MUA_responsive_electrodes_narrow=Select_responsive_electrodes(MUA_dataset,stim)

LFP_dataset=LFP_Response_dataset(LFP,'w')
LFP_responsive_electrodes_wide=Select_responsive_electrodes(LFP_dataset,stim)
LFP_dataset=LFP_Response_dataset(LFP,'m')
LFP_responsive_electrodes_medium=Select_responsive_electrodes(LFP_dataset,stim)
LFP_dataset=LFP_Response_dataset(LFP,'n')
LFP_responsive_electrodes_narrow=Select_responsive_electrodes(LFP_dataset,stim)

######### Responsive probes (probes that elicit response)
#Select those probes that elicit response in the population of responsive electrodes

MUA_dataset=MUA_Response_dataset(MUA,'w')
MUA_responsive_probes_wide=responsive_probes(MUA_dataset[:,MUA_responsive_electrodes_wide],stim)
MUA_dataset=MUA_Response_dataset(MUA,'m')
MUA_responsive_probes_medium=responsive_probes(MUA_dataset[:,MUA_responsive_electrodes_medium],stim)
MUA_dataset=MUA_Response_dataset(MUA,'n')
MUA_responsive_probes_narrow=responsive_probes(MUA_dataset[:,MUA_responsive_electrodes_narrow],stim)

LFP_dataset=LFP_Response_dataset(LFP,'w')
LFP_responsive_probes_wide=responsive_probes(-LFP_dataset[:,LFP_responsive_electrodes_wide],stim)
LFP_dataset=LFP_Response_dataset(LFP,'m')
LFP_responsive_probes_medium=responsive_probes(-LFP_dataset[:,LFP_responsive_electrodes_medium],stim)
LFP_dataset=LFP_Response_dataset(LFP,'n')
LFP_responsive_probes_narrow=responsive_probes(-LFP_dataset[:,LFP_responsive_electrodes_narrow],stim)

######### Pairs of responsive probes
#Permutation of responsive probes to prepare probe pairs considering
#replacement of labels

MUA_probe_pairs_wide = list(itertools.permutations(list(MUA_responsive_probes_wide),2))
MUA_probe_pairs_wide = np.array(MUA_probe_pairs_wide,dtype=int)
MUA_probe_pairs_medium = list(itertools.permutations(list(MUA_responsive_probes_medium),2))
MUA_probe_pairs_medium = np.array(MUA_probe_pairs_medium,dtype=int)
MUA_probe_pairs_narrow = list(itertools.permutations(list(MUA_responsive_probes_narrow),2))
MUA_probe_pairs_narrow = np.array(MUA_probe_pairs_narrow,dtype=int)

LFP_probe_pairs_wide = list(itertools.permutations(list(LFP_responsive_probes_wide),2))
LFP_probe_pairs_wide = np.array(LFP_probe_pairs_wide,dtype=int)
LFP_probe_pairs_medium = list(itertools.permutations(list(LFP_responsive_probes_medium),2))
LFP_probe_pairs_medium = np.array(LFP_probe_pairs_medium,dtype=int)
LFP_probe_pairs_narrow = list(itertools.permutations(list(LFP_responsive_probes_narrow),2))
LFP_probe_pairs_narrow = np.array(LFP_probe_pairs_narrow,dtype=int)

######### Distances of all pairs of responsive probes
#For each pair of probes measure the Euclidean distance in visual degrees

MUA_distance_array_wide=probe_distances_array(MUA_probe_pairs_wide)
MUA_distance_array_medium=probe_distances_array(MUA_probe_pairs_medium)
MUA_distance_array_narrow=probe_distances_array(MUA_probe_pairs_narrow)

LFP_distance_array_wide=probe_distances_array(LFP_probe_pairs_wide)
LFP_distance_array_medium=probe_distances_array(LFP_probe_pairs_medium)
LFP_distance_array_narrow=probe_distances_array(LFP_probe_pairs_narrow)

######### Pairs of responsive probes with <15 distances
#Only select pairs with less than 15 degree separation

MUA_probe_pairs_wide=MUA_probe_pairs_wide[MUA_distance_array_wide<15]
MUA_probe_pairs_medium=MUA_probe_pairs_medium[MUA_distance_array_medium<15]
MUA_probe_pairs_narrow=MUA_probe_pairs_narrow[MUA_distance_array_narrow<15]

LFP_probe_pairs_wide=LFP_probe_pairs_wide[LFP_distance_array_wide<15]
LFP_probe_pairs_medium=LFP_probe_pairs_medium[LFP_distance_array_medium<15]
LFP_probe_pairs_narrow=LFP_probe_pairs_narrow[LFP_distance_array_narrow<15]

######### <15 distances
#From the corresponding distance array only select lower than 15 degree
#distances

MUA_distance_array_wide=MUA_distance_array_wide[MUA_distance_array_wide<15]
MUA_distance_array_medium=MUA_distance_array_medium[MUA_distance_array_medium<15]
MUA_distance_array_narrow=MUA_distance_array_narrow[MUA_distance_array_narrow<15]

LFP_distance_array_wide=LFP_distance_array_wide[LFP_distance_array_wide<15]
LFP_distance_array_medium=LFP_distance_array_medium[LFP_distance_array_medium<15]
LFP_distance_array_narrow=LFP_distance_array_narrow[LFP_distance_array_narrow<15]

######### Mean eccentricity, magnification factor, and cortical distances
#Here we first calculate the mean eccentricity for each pair of probes.
#using equation M=3.01E^(-0.9) from Gattass paper we calculate the
#cortical magnification factor corresponding to each mean eccentricity (of the pairs)
#we then calculate the cortical distance between each pair of probes
#by multiplying their visual separation in degree and the corresponding
#magnification factor.

#MUA

MUA_ecc_pairs_wide=Pair_eccentricity(MUA_probe_pairs_wide)
MUA_mgf_pairs_wide=Magnification_factor(MUA_ecc_pairs_wide)
MUA_cortical_distances_wide=cortical_distance(MUA_distance_array_wide,MUA_mgf_pairs_wide)

MUA_ecc_pairs_medium=Pair_eccentricity(MUA_probe_pairs_medium)
MUA_mgf_pairs_medium=Magnification_factor(MUA_ecc_pairs_medium)
MUA_cortical_distances_medium=cortical_distance(MUA_distance_array_medium,MUA_mgf_pairs_medium)

MUA_ecc_pairs_narrow=Pair_eccentricity(MUA_probe_pairs_narrow)
MUA_mgf_pairs_narrow=Magnification_factor(MUA_ecc_pairs_narrow)
MUA_cortical_distances_narrow=cortical_distance(MUA_distance_array_narrow,MUA_mgf_pairs_narrow)

#LFP

LFP_ecc_pairs_wide=Pair_eccentricity(LFP_probe_pairs_wide)
LFP_mgf_pairs_wide=Magnification_factor(LFP_ecc_pairs_wide)
LFP_cortical_distances_wide=cortical_distance(LFP_distance_array_wide,LFP_mgf_pairs_wide)

LFP_ecc_pairs_medium=Pair_eccentricity(LFP_probe_pairs_medium)
LFP_mgf_pairs_medium=Magnification_factor(LFP_ecc_pairs_medium)
LFP_cortical_distances_medium=cortical_distance(LFP_distance_array_medium,LFP_mgf_pairs_medium)

LFP_ecc_pairs_narrow=Pair_eccentricity(LFP_probe_pairs_narrow)
LFP_mgf_pairs_narrow=Magnification_factor(LFP_ecc_pairs_narrow)
LFP_cortical_distances_narrow=cortical_distance(LFP_distance_array_narrow,LFP_mgf_pairs_narrow)


######### Discrimination of all pairs of responsive probes
#For each pair of probes we calculate the 5-fold cross-validation performance
#and grid search to find the best parameters for the estimator. We measured
#the performance using the area under the ROC curve to consider sensitivity and specificity together.
#We only use those probes with <15 distance as it higher distances are not applicable for prosthetic applications


MUA_dataset=MUA_Response_dataset(MUA,'w')
MUA_performance_array_wide=Discrimination_performance_array(MUA_probe_pairs_wide,stim,MUA_dataset[:,MUA_responsive_electrodes_wide],'Linear SVM')
MUA_dataset=MUA_Response_dataset(MUA,'m')
MUA_performance_array_medium=Discrimination_performance_array(MUA_probe_pairs_medium,stim,MUA_dataset[:,MUA_responsive_electrodes_medium],'Linear SVM')
MUA_dataset=MUA_Response_dataset(MUA,'n')
MUA_performance_array_narrow=Discrimination_performance_array(MUA_probe_pairs_narrow,stim,MUA_dataset[:,MUA_responsive_electrodes_narrow],'Linear SVM')

LFP_dataset=LFP_Response_dataset(LFP,'w')
LFP_performance_array_wide=Discrimination_performance_array(LFP_probe_pairs_wide,stim,LFP_dataset[:,LFP_responsive_electrodes_wide],'Linear SVM')
LFP_dataset=LFP_Response_dataset(LFP,'m')
LFP_performance_array_medium=Discrimination_performance_array(LFP_probe_pairs_medium,stim,LFP_dataset[:,LFP_responsive_electrodes_medium],'Linear SVM')
LFP_dataset=LFP_Response_dataset(LFP,'n')
LFP_performance_array_narrow=Discrimination_performance_array(LFP_probe_pairs_narrow,stim,LFP_dataset[:,LFP_responsive_electrodes_narrow],'Linear SVM')

######### Plot discrimination performance vs cortical distance

performance_corticaldistance_plotter(MUA_cortical_distances_wide,MUA_cortical_distances_medium,MUA_cortical_distances_narrow,MUA_performance_array_wide,MUA_performance_array_medium,MUA_performance_array_narrow,(0.435, 0.043, 0.043),(0.913, 0.145, 0.145),(0.960, 0.639, 0.639),'MUA','MUA_performance_corticaldist.png')
performance_corticaldistance_plotter(LFP_cortical_distances_wide,LFP_cortical_distances_medium,LFP_cortical_distances_narrow,LFP_performance_array_wide,LFP_performance_array_medium,LFP_performance_array_narrow,(0.047, 0.062, 0.372),(0.145, 0.188, 0.894),(0.611, 0.631, 0.949),'LFP','LFP_performance_corticaldist.png')

#for 1 deg separation at eccentricity 1 deg, the cortical distance is
#3.01 mm which is quite big cortical distance with high performance
#The performance for this separation for MUA and LFP are 92% and 82% for
#linear SVM and 94% and 85% for LFP for RBF SVM.

######### Plot discrimination performance vs mean eccentricity of probes with specific separation

separation_dist=4
Performance_eccentricity_plotter(MUA_distance_array_wide,MUA_distance_array_medium,MUA_distance_array_narrow,MUA_ecc_pairs_wide,MUA_ecc_pairs_medium,MUA_ecc_pairs_narrow,MUA_performance_array_wide,MUA_performance_array_medium,MUA_performance_array_narrow,separation_dist,(0.435, 0.043, 0.043),(0.913, 0.145, 0.145),(0.960, 0.639, 0.639),'MUA 4 deg separation','Performance_Eccentricity_MUA_four.png')
Performance_eccentricity_plotter(LFP_distance_array_wide,LFP_distance_array_medium,LFP_distance_array_narrow,LFP_ecc_pairs_wide,LFP_ecc_pairs_medium,LFP_ecc_pairs_narrow,LFP_performance_array_wide,LFP_performance_array_medium,LFP_performance_array_narrow,separation_dist,(0.047, 0.062, 0.372),(0.145, 0.188, 0.894),(0.611, 0.631, 0.949),'LFP 4 deg separation','Performance_Eccentricity_LFP_four.png')


separation_dist=5.656854249492381
Performance_eccentricity_plotter(MUA_distance_array_wide,MUA_distance_array_medium,MUA_distance_array_narrow,MUA_ecc_pairs_wide,MUA_ecc_pairs_medium,MUA_ecc_pairs_narrow,MUA_performance_array_wide,MUA_performance_array_medium,MUA_performance_array_narrow,separation_dist,(0.435, 0.043, 0.043),(0.913, 0.145, 0.145),(0.960, 0.639, 0.639),'MUA 5.6 deg separation','Performance_Eccentricity_MUA_five.png')
Performance_eccentricity_plotter(LFP_distance_array_wide,LFP_distance_array_medium,LFP_distance_array_narrow,LFP_ecc_pairs_wide,LFP_ecc_pairs_medium,LFP_ecc_pairs_narrow,LFP_performance_array_wide,LFP_performance_array_medium,LFP_performance_array_narrow,separation_dist,(0.047, 0.062, 0.372),(0.145, 0.188, 0.894),(0.611, 0.631, 0.949),'LFP 5.6 deg separation','Performance_Eccentricity_LFP_five.png')


separation_dist=8
Performance_eccentricity_plotter(MUA_distance_array_wide,MUA_distance_array_medium,MUA_distance_array_narrow,MUA_ecc_pairs_wide,MUA_ecc_pairs_medium,MUA_ecc_pairs_narrow,MUA_performance_array_wide,MUA_performance_array_medium,MUA_performance_array_narrow,separation_dist,(0.435, 0.043, 0.043),(0.913, 0.145, 0.145),(0.960, 0.639, 0.639),'MUA 8 deg separation','Performance_Eccentricity_MUA_eight.png')
Performance_eccentricity_plotter(LFP_distance_array_wide,LFP_distance_array_medium,LFP_distance_array_narrow,LFP_ecc_pairs_wide,LFP_ecc_pairs_medium,LFP_ecc_pairs_narrow,LFP_performance_array_wide,LFP_performance_array_medium,LFP_performance_array_narrow,separation_dist,(0.047, 0.062, 0.372),(0.145, 0.188, 0.894),(0.611, 0.631, 0.949),'LFP 8 deg separation','Performance_Eccentricity_LFP_eight.png')

###################### Section 2: Minimizing the number of electrodes
#The goal here is to select the best group of electrodes for discrimination and
#select minimum number of them without impairing the performance. 

#Conclusion: we use medium window for LFP and wide window for MUA
MUA_dataset=MUA_Response_dataset(MUA,'w')
LFP_dataset=LFP_Response_dataset(LFP,'m')

######### Method 1: Clustering similar electrodes
#We clustered the receptive fields to find the similar receptive fields
#then from each group we selected 2 or 3 representatives with highest
#peak of receptive field. We used the obtained group of electrodes to 
#repeat the discrimination analysis. 


no_cluster=3
cat_no=2
MUA_cat=RF_cluster(stim,MUA_dataset,MUA_responsive_electrodes_wide,no_cluster,cat_no)

fig=plt.figure(figsize=(12, 12))
population_contours(MUA_dataset,stim,MUA_cat,4,'r')
plt.show()
fig.savefig('MUA_Clustered_RF_contours2.png', dpi=150)

MUA_three_highest_peak=RF_representative_electrode(MUA_cat,stim,MUA_dataset)
MUA_clustering_selected_electrodes=np.array([49, 19, 18, 58, 10, 45, 54, 72])
RF_set_viewer(stim,MUA_dataset,MUA_clustering_selected_electrodes,-0.27,0.27,'sure')
MUA_performance_array_clustering_selected=Discrimination_performance_array(MUA_probe_pairs_wide,stim,MUA_dataset[:,MUA_clustering_selected_electrodes],'Linear SVM')


performance_corticaldistance_plotter_clustering(MUA_cortical_distances_wide,MUA_performance_array_wide,MUA_performance_array_clustering_selected,(0.435, 0.043, 0.043),(0.952, 0.568, 0.349),'MUA','MUA_performance_corticaldist_clustering.png')
separation_dist=4
Performance_eccentricity_plotter_clustering(MUA_distance_array_wide,MUA_ecc_pairs_wide,MUA_performance_array_wide,MUA_performance_array_clustering_selected,separation_dist,(0.435, 0.043, 0.043),(0.952, 0.568, 0.349),'MUA 4 deg separation','Performance_Eccentricity_MUA_four_clustering.png')
separation_dist=5.656854249492381
Performance_eccentricity_plotter_clustering(MUA_distance_array_wide,MUA_ecc_pairs_wide,MUA_performance_array_wide,MUA_performance_array_clustering_selected,separation_dist,(0.435, 0.043, 0.043),(0.952, 0.568, 0.349),'MUA 5.6 deg separation','Performance_Eccentricity_MUA_five_clustering.png')
separation_dist=8
Performance_eccentricity_plotter_clustering(MUA_distance_array_wide,MUA_ecc_pairs_wide,MUA_performance_array_wide,MUA_performance_array_clustering_selected,separation_dist,(0.435, 0.043, 0.043),(0.952, 0.568, 0.349),'MUA 8 deg separation','Performance_Eccentricity_MUA_eight_clustering.png')



no_cluster=3
cat_no=2
LFP_cat=RF_cluster(stim,-LFP_dataset,LFP_responsive_electrodes_medium,no_cluster,cat_no)

fig=plt.figure(figsize=(12, 12))
population_contours(-LFP_dataset,stim,LFP_cat,4,'b')
plt.show()
fig.savefig('LFP_Clustered_RF_contours2.png', dpi=150)

LFP_three_highest_peak=RF_representative_electrode(LFP_cat,stim,-LFP_dataset)
LFP_clustering_selected_electrodes=np.array([95, 79, 50, 19, 20, 18, 69, 68, 85])
RF_set_viewer(stim,-LFP_dataset,LFP_clustering_selected_electrodes,-0.34,0.34,'sure')
LFP_performance_array_clustering_selected=Discrimination_performance_array(LFP_probe_pairs_medium,stim,LFP_dataset[:,LFP_clustering_selected_electrodes],'Linear SVM')


performance_corticaldistance_plotter_clustering(LFP_cortical_distances_medium,LFP_performance_array_medium,LFP_performance_array_clustering_selected,(0.145, 0.188, 0.894),(0.635, 0.349, 0.952),'LFP','LFP_performance_corticaldist_clustering.png')
separation_dist=4
Performance_eccentricity_plotter_clustering(LFP_distance_array_medium,LFP_ecc_pairs_medium,LFP_performance_array_medium,LFP_performance_array_clustering_selected,separation_dist,(0.145, 0.188, 0.894),(0.635, 0.349, 0.952),'LFP 4 deg separation','Performance_Eccentricity_LFP_four_clustering.png')
separation_dist=5.656854249492381
Performance_eccentricity_plotter_clustering(LFP_distance_array_medium,LFP_ecc_pairs_medium,LFP_performance_array_medium,LFP_performance_array_clustering_selected,separation_dist,(0.145, 0.188, 0.894),(0.635, 0.349, 0.952),'LFP 5.6 deg separation','Performance_Eccentricity_LFP_five_clustering.png')
separation_dist=8
Performance_eccentricity_plotter_clustering(LFP_distance_array_medium,LFP_ecc_pairs_medium,LFP_performance_array_medium,LFP_performance_array_clustering_selected,separation_dist,(0.145, 0.188, 0.894),(0.635, 0.349, 0.952),'LFP 8 deg separation','Performance_Eccentricity_LFP_eight_clustering.png')


######### Method 2: Using electrodes importance

#When weights of all the discriminations of <15 deg separation are used

MUA_weight_set=SVM_weight_set(MUA_probe_pairs_wide,stim,MUA_dataset[:,MUA_responsive_electrodes_wide])
MUA_weightsorted_electrodes=electrodes_sorting_by_weight(MUA_weight_set,MUA_responsive_electrodes_wide)
MUA_performance_array_importance_four_selected=Discrimination_performance_array(MUA_probe_pairs_wide,stim,MUA_dataset[:,MUA_weightsorted_electrodes[0:4]],'Linear SVM')
MUA_performance_array_importance_six_selected=Discrimination_performance_array(MUA_probe_pairs_wide,stim,MUA_dataset[:,MUA_weightsorted_electrodes[0:6]],'Linear SVM')
MUA_performance_array_importance_eight_selected=Discrimination_performance_array(MUA_probe_pairs_wide,stim,MUA_dataset[:,MUA_weightsorted_electrodes[0:8]],'Linear SVM')
MUA_performance_array_importance_ten_selected=Discrimination_performance_array(MUA_probe_pairs_wide,stim,MUA_dataset[:,MUA_weightsorted_electrodes[0:10]],'Linear SVM')
MUA_performance_array_importance_twelve_selected=Discrimination_performance_array(MUA_probe_pairs_wide,stim,MUA_dataset[:,MUA_weightsorted_electrodes[0:12]],'Linear SVM')


performance_corticaldistance_plotter_importance(MUA_cortical_distances_wide,MUA_performance_array_wide,MUA_performance_array_importance_four_selected,MUA_performance_array_importance_six_selected,MUA_performance_array_importance_eight_selected,MUA_performance_array_importance_ten_selected,MUA_performance_array_importance_twelve_selected,'MUA_performance_corticaldist_importance_selected.png')


separation_dist=4
Performance_eccentricity_plotter_importance(MUA_distance_array_wide,MUA_ecc_pairs_wide,MUA_performance_array_wide,MUA_performance_array_importance_four_selected,MUA_performance_array_importance_six_selected,MUA_performance_array_importance_eight_selected,MUA_performance_array_importance_ten_selected,MUA_performance_array_importance_twelve_selected,separation_dist,'4 degree separation','Performance_Eccentricity_MUA_importance_four_deg.png')
separation_dist=8
Performance_eccentricity_plotter_importance(MUA_distance_array_wide,MUA_ecc_pairs_wide,MUA_performance_array_wide,MUA_performance_array_importance_four_selected,MUA_performance_array_importance_six_selected,MUA_performance_array_importance_eight_selected,MUA_performance_array_importance_ten_selected,MUA_performance_array_importance_twelve_selected,separation_dist,'8 degree separation','Performance_Eccentricity_MUA_importance_eight_deg.png')




LFP_weight_set=SVM_weight_set(LFP_probe_pairs_medium,stim,LFP_dataset[:,LFP_responsive_electrodes_medium])
LFP_weightsorted_electrodes=electrodes_sorting_by_weight(LFP_weight_set,LFP_responsive_electrodes_medium)
LFP_performance_array_importance_four_selected=Discrimination_performance_array(LFP_probe_pairs_medium,stim,LFP_dataset[:,LFP_weightsorted_electrodes[0:4]],'Linear SVM')
LFP_performance_array_importance_six_selected=Discrimination_performance_array(LFP_probe_pairs_medium,stim,LFP_dataset[:,LFP_weightsorted_electrodes[0:6]],'Linear SVM')
LFP_performance_array_importance_eight_selected=Discrimination_performance_array(LFP_probe_pairs_medium,stim,LFP_dataset[:,LFP_weightsorted_electrodes[0:8]],'Linear SVM')
LFP_performance_array_importance_ten_selected=Discrimination_performance_array(LFP_probe_pairs_medium,stim,LFP_dataset[:,LFP_weightsorted_electrodes[0:10]],'Linear SVM')
LFP_performance_array_importance_twelve_selected=Discrimination_performance_array(LFP_probe_pairs_medium,stim,LFP_dataset[:,LFP_weightsorted_electrodes[0:12]],'Linear SVM')


performance_corticaldistance_plotter_importance(LFP_cortical_distances_medium,LFP_performance_array_medium,LFP_performance_array_importance_four_selected,LFP_performance_array_importance_six_selected,LFP_performance_array_importance_eight_selected,LFP_performance_array_importance_ten_selected,LFP_performance_array_importance_twelve_selected,'LFP_performance_corticaldist_importance_selected.png')

separation_dist=4
Performance_eccentricity_plotter_importance(LFP_distance_array_medium,LFP_ecc_pairs_medium,LFP_performance_array_medium,LFP_performance_array_importance_four_selected,LFP_performance_array_importance_six_selected,LFP_performance_array_importance_eight_selected,LFP_performance_array_importance_ten_selected,LFP_performance_array_importance_twelve_selected,separation_dist,'4 degree separation','Performance_Eccentricity_LFP_importance_four_deg.png')
separation_dist=8
Performance_eccentricity_plotter_importance(LFP_distance_array_medium,LFP_ecc_pairs_medium,LFP_performance_array_medium,LFP_performance_array_importance_four_selected,LFP_performance_array_importance_six_selected,LFP_performance_array_importance_eight_selected,LFP_performance_array_importance_ten_selected,LFP_performance_array_importance_twelve_selected,separation_dist,'8 degree separation','Performance_Eccentricity_LFP_importance_eight_deg.png')


#When weights of all the discriminations of ==4 deg separation and eccentricity<=8 deg are used

MUA_probe_pairs_separation_four_ecc_eight=MUA_probe_pairs_wide[(MUA_distance_array_wide==4)&(MUA_ecc_pairs_wide<8.1),:]
MUA_weight_set_separation_four_ecc_eight=SVM_weight_set(MUA_probe_pairs_separation_four_ecc_eight,stim,MUA_dataset[:,MUA_responsive_electrodes_wide])
MUA_weightsorted_electrodes_separation_four_ecc_eight=electrodes_sorting_by_weight(MUA_weight_set_separation_four_ecc_eight,MUA_responsive_electrodes_wide)

MUA_performance_array_importance_four_selected_separation_four_ecc_eight=Discrimination_performance_array(MUA_probe_pairs_separation_four_ecc_eight,stim,MUA_dataset[:,MUA_weightsorted_electrodes_separation_four_ecc_eight[0:4]],'Linear SVM')
MUA_performance_array_importance_six_selected_separation_four_ecc_eight=Discrimination_performance_array(MUA_probe_pairs_separation_four_ecc_eight,stim,MUA_dataset[:,MUA_weightsorted_electrodes_separation_four_ecc_eight[0:6]],'Linear SVM')
MUA_performance_array_importance_eight_selected_separation_four_ecc_eight=Discrimination_performance_array(MUA_probe_pairs_separation_four_ecc_eight,stim,MUA_dataset[:,MUA_weightsorted_electrodes_separation_four_ecc_eight[0:8]],'Linear SVM')
MUA_performance_array_importance_ten_selected_separation_four_ecc_eight=Discrimination_performance_array(MUA_probe_pairs_separation_four_ecc_eight,stim,MUA_dataset[:,MUA_weightsorted_electrodes_separation_four_ecc_eight[0:10]],'Linear SVM')
MUA_performance_array_importance_twelve_selected_separation_four_ecc_eight=Discrimination_performance_array(MUA_probe_pairs_separation_four_ecc_eight,stim,MUA_dataset[:,MUA_weightsorted_electrodes_separation_four_ecc_eight[0:12]],'Linear SVM')

separation_dist=4
Performance_eccentricity_plotter_importance(MUA_distance_array_wide[(MUA_distance_array_wide==4)&(MUA_ecc_pairs_wide<8.1)],MUA_ecc_pairs_wide[(MUA_distance_array_wide==4)&(MUA_ecc_pairs_wide<8.1)],MUA_performance_array_wide[(MUA_distance_array_wide==4)&(MUA_ecc_pairs_wide<8.1)],MUA_performance_array_importance_four_selected_separation_four_ecc_eight,MUA_performance_array_importance_six_selected_separation_four_ecc_eight,MUA_performance_array_importance_eight_selected_separation_four_ecc_eight,MUA_performance_array_importance_ten_selected_separation_four_ecc_eight,MUA_performance_array_importance_twelve_selected_separation_four_ecc_eight,separation_dist,'4 degree separation','Performance_Eccentricity_MUA_importance_four_deg_separation_four_ecc_eight.png')

Electrode_array_show(MUA_weightsorted_electrodes_separation_four_ecc_eight[0:6],plxarray,'MUA_enough_electrodes')



LFP_probe_pairs_separation_four_ecc_eight=LFP_probe_pairs_medium[(LFP_distance_array_medium==4)&(LFP_ecc_pairs_medium<8.1),:]
LFP_weight_set_separation_four_ecc_eight=SVM_weight_set(LFP_probe_pairs_separation_four_ecc_eight,stim,LFP_dataset[:,LFP_responsive_electrodes_medium])
LFP_weightsorted_electrodes_separation_four_ecc_eight=electrodes_sorting_by_weight(LFP_weight_set_separation_four_ecc_eight,LFP_responsive_electrodes_medium)

LFP_performance_array_importance_four_selected_separation_four_ecc_eight=Discrimination_performance_array(LFP_probe_pairs_separation_four_ecc_eight,stim,LFP_dataset[:,LFP_weightsorted_electrodes_separation_four_ecc_eight[0:4]],'Linear SVM')
LFP_performance_array_importance_six_selected_separation_four_ecc_eight=Discrimination_performance_array(LFP_probe_pairs_separation_four_ecc_eight,stim,LFP_dataset[:,LFP_weightsorted_electrodes_separation_four_ecc_eight[0:6]],'Linear SVM')
LFP_performance_array_importance_eight_selected_separation_four_ecc_eight=Discrimination_performance_array(LFP_probe_pairs_separation_four_ecc_eight,stim,LFP_dataset[:,LFP_weightsorted_electrodes_separation_four_ecc_eight[0:8]],'Linear SVM')
LFP_performance_array_importance_ten_selected_separation_four_ecc_eight=Discrimination_performance_array(LFP_probe_pairs_separation_four_ecc_eight,stim,LFP_dataset[:,LFP_weightsorted_electrodes_separation_four_ecc_eight[0:10]],'Linear SVM')
LFP_performance_array_importance_twelve_selected_separation_four_ecc_eight=Discrimination_performance_array(LFP_probe_pairs_separation_four_ecc_eight,stim,LFP_dataset[:,LFP_weightsorted_electrodes_separation_four_ecc_eight[0:12]],'Linear SVM')

separation_dist=4
Performance_eccentricity_plotter_importance(LFP_distance_array_medium[(LFP_distance_array_medium==4)&(LFP_ecc_pairs_medium<8.1)],LFP_ecc_pairs_medium[(LFP_distance_array_medium==4)&(LFP_ecc_pairs_medium<8.1)],LFP_performance_array_medium[(LFP_distance_array_medium==4)&(LFP_ecc_pairs_medium<8.1)],LFP_performance_array_importance_four_selected_separation_four_ecc_eight,LFP_performance_array_importance_six_selected_separation_four_ecc_eight,LFP_performance_array_importance_eight_selected_separation_four_ecc_eight,LFP_performance_array_importance_ten_selected_separation_four_ecc_eight,LFP_performance_array_importance_twelve_selected_separation_four_ecc_eight,separation_dist,'4 degree separation','Performance_Eccentricity_LFP_importance_four_deg_separation_four_ecc_eight.png')

Electrode_array_show(LFP_weightsorted_electrodes_separation_four_ecc_eight[0:16],plxarray,'LFP_enough_electrodes')


#Further analysis (not shown) with more than 12 electrodes showed that performance
#reached to its maximum (90%) with 16 electrodes for 4 deg separation
#in 4.5 deg eccentricity. Using magnification factor in V4, 4 deg separation
#at 4.5 deg eccentricity has 3.11 mm cortical distance. This amount of
#distance can contain 8 electrodes. The result of this analysis shows
#that precise decoding doesn't need all the electrodes (sparser distribution of electrodes will be better).


###################### Section 3: Weight analysis
#The goal here is to show how weights are assigned and distributed
#Specifically we study the coding strategy for small and large separations

######### Distribution of weights over the electrode array

#Histogram of importance values for MUA and LFP

Electrodes_importance_histogram(MUA_weight_set,'r','MUA_importance_histogram.png')
Electrodes_importance_histogram(LFP_weight_set,'b','LFP_importance_histogram.png')

#Distribution of importance values over electrode arrays

v_min,v_max=0.0189,0.0525
Electrode_array_importance(MUA_weight_set,MUA_responsive_electrodes_wide,plxarray,v_min,v_max,'MUA_importance_on_array.png')
v_min,v_max=0.0074,0.0156
Electrode_array_importance(LFP_weight_set,LFP_responsive_electrodes_medium,plxarray,v_min,v_max,'LFP_importance_on_array.png')


######### Comparison of tuning curve and weight field of an electrode

#MUA

electrode=10

separation_dist=4
probes_weight(MUA_weight_set,electrode,MUA_responsive_electrodes_wide,MUA_probe_pairs_wide,separation_dist,MUA_distance_array_wide,'MUA_weight_field_elec10_separation4.png')
separation_dist=8
probes_weight(MUA_weight_set,electrode,MUA_responsive_electrodes_wide,MUA_probe_pairs_wide,separation_dist,MUA_distance_array_wide,'MUA_weight_field_elec10_separation8.png')
separation_dist=12
probes_weight(MUA_weight_set,electrode,MUA_responsive_electrodes_wide,MUA_probe_pairs_wide,separation_dist,MUA_distance_array_wide,'MUA_weight_field_elec10_separation12.png')

tuning_plotter(stim,MUA_dataset,electrode,-0.27,0.27,'MUA_tuning_curve_elec10.png')

electrode=58
separation_dist=4
probes_weight(MUA_weight_set,electrode,MUA_responsive_electrodes_wide,MUA_probe_pairs_wide,separation_dist,MUA_distance_array_wide,'MUA_weight_field_elec58_separation4.png')
separation_dist=8
probes_weight(MUA_weight_set,electrode,MUA_responsive_electrodes_wide,MUA_probe_pairs_wide,separation_dist,MUA_distance_array_wide,'MUA_weight_field_elec58_separation8.png')
separation_dist=12
probes_weight(MUA_weight_set,electrode,MUA_responsive_electrodes_wide,MUA_probe_pairs_wide,separation_dist,MUA_distance_array_wide,'MUA_weight_field_elec58_separation12.png')

tuning_plotter(stim,MUA_dataset,electrode,-0.27,0.27,'MUA_tuning_curve_elec58.png')


electrode=19
separation_dist=4
probes_weight(MUA_weight_set,electrode,MUA_responsive_electrodes_wide,MUA_probe_pairs_wide,separation_dist,MUA_distance_array_wide,'MUA_weight_field_elec19_separation4.png')
separation_dist=8
probes_weight(MUA_weight_set,electrode,MUA_responsive_electrodes_wide,MUA_probe_pairs_wide,separation_dist,MUA_distance_array_wide,'MUA_weight_field_elec19_separation8.png')
separation_dist=12
probes_weight(MUA_weight_set,electrode,MUA_responsive_electrodes_wide,MUA_probe_pairs_wide,separation_dist,MUA_distance_array_wide,'MUA_weight_field_elec19_separation12.png')

tuning_plotter(stim,MUA_dataset,electrode,-0.27,0.27,'MUA_tuning_curve_elec19.png')


#LFP

electrode=2
separation_dist=4
probes_weight(LFP_weight_set,electrode,LFP_responsive_electrodes_medium,LFP_probe_pairs_medium,separation_dist,LFP_distance_array_medium,'LFP_weight_field_elec2_separation4.png')
separation_dist=8
probes_weight(LFP_weight_set,electrode,LFP_responsive_electrodes_medium,LFP_probe_pairs_medium,separation_dist,LFP_distance_array_medium,'LFP_weight_field_elec2_separation8.png')
separation_dist=12
probes_weight(LFP_weight_set,electrode,LFP_responsive_electrodes_medium,LFP_probe_pairs_medium,separation_dist,LFP_distance_array_medium,'LFP_weight_field_elec2_separation12.png')

tuning_plotter(stim,-LFP_dataset,electrode,-0.34,0.34,'LFP_tuning_curve_elec2.png')


electrode=30
separation_dist=4
probes_weight(LFP_weight_set,electrode,LFP_responsive_electrodes_medium,LFP_probe_pairs_medium,separation_dist,LFP_distance_array_medium,'LFP_weight_field_elec30_separation4.png')
separation_dist=8
probes_weight(LFP_weight_set,electrode,LFP_responsive_electrodes_medium,LFP_probe_pairs_medium,separation_dist,LFP_distance_array_medium,'LFP_weight_field_elec30_separation8.png')
separation_dist=12
probes_weight(LFP_weight_set,electrode,LFP_responsive_electrodes_medium,LFP_probe_pairs_medium,separation_dist,LFP_distance_array_medium,'LFP_weight_field_elec30_separation12.png')

tuning_plotter(stim,-LFP_dataset,electrode,-0.34,0.34,'LFP_tuning_curve_elec30.png')



electrode=55
separation_dist=4
probes_weight(LFP_weight_set,electrode,LFP_responsive_electrodes_medium,LFP_probe_pairs_medium,separation_dist,LFP_distance_array_medium,'LFP_weight_field_elec55_separation4.png')
separation_dist=8
probes_weight(LFP_weight_set,electrode,LFP_responsive_electrodes_medium,LFP_probe_pairs_medium,separation_dist,LFP_distance_array_medium,'LFP_weight_field_elec55_separation8.png')
separation_dist=12
probes_weight(LFP_weight_set,electrode,LFP_responsive_electrodes_medium,LFP_probe_pairs_medium,separation_dist,LFP_distance_array_medium,'LFP_weight_field_elec55_separation12.png')

tuning_plotter(stim,-LFP_dataset,electrode,-0.34,0.34,'LFP_tuning_curve_elec55.png')


electrode=48
separation_dist=4
probes_weight(LFP_weight_set,electrode,LFP_responsive_electrodes_medium,LFP_probe_pairs_medium,separation_dist,LFP_distance_array_medium,'LFP_weight_field_elec48_separation4.png')
separation_dist=8
probes_weight(LFP_weight_set,electrode,LFP_responsive_electrodes_medium,LFP_probe_pairs_medium,separation_dist,LFP_distance_array_medium,'LFP_weight_field_elec48_separation8.png')
separation_dist=12
probes_weight(LFP_weight_set,electrode,LFP_responsive_electrodes_medium,LFP_probe_pairs_medium,separation_dist,LFP_distance_array_medium,'LFP_weight_field_elec48_separation12.png')

tuning_plotter(stim,-LFP_dataset,electrode,-0.34,0.34,'LFP_tuning_curve_elec48.png')

#Relationship between weights of an electrode and cortical distances

MUA_p_values_weight_corticaldist=weight_corticaldist(MUA_weight_set,MUA_cortical_distances_wide,MUA_responsive_electrodes_wide,19)
LFP_p_values_weight_corticaldist=weight_corticaldist(LFP_weight_set,LFP_cortical_distances_medium,LFP_responsive_electrodes_medium,55)


MUA_p_values_importance_corticaldist=mean_importance_corticaldist(MUA_weight_set,MUA_cortical_distances_wide,MUA_responsive_electrodes_wide,10,58,19,18)
LFP_p_values_importance_corticaldist=mean_importance_corticaldist(LFP_weight_set,LFP_cortical_distances_medium,LFP_responsive_electrodes_medium,2,30,55,48)


###################### Section 4: Decoding bandpass LFP

#theta

LFP_theta=bandpass_LFP(LFP,4,8)
LFP_dataset_theta=LFP_Response_dataset(LFP_theta,'m')
LFP_performance_array_theta=Discrimination_performance_array(LFP_probe_pairs_medium,stim,LFP_dataset_theta[:,LFP_responsive_electrodes_medium],'Linear SVM')

#alpha

LFP_alpha=bandpass_LFP(LFP,8,12)
LFP_dataset_alpha=LFP_Response_dataset(LFP_alpha,'m')
LFP_performance_array_alpha=Discrimination_performance_array(LFP_probe_pairs_medium,stim,LFP_dataset_alpha[:,LFP_responsive_electrodes_medium],'Linear SVM')

#beta

LFP_beta=bandpass_LFP(LFP,12,30)
LFP_dataset_beta=LFP_Response_dataset(LFP_beta,'m')
LFP_performance_array_beta=Discrimination_performance_array(LFP_probe_pairs_medium,stim,LFP_dataset_beta[:,LFP_responsive_electrodes_medium],'Linear SVM')

#gamma

LFP_gamma=bandpass_LFP(LFP,30,50)
LFP_dataset_gamma=LFP_Response_dataset(LFP_gamma,'m')
LFP_performance_array_gamma=Discrimination_performance_array(LFP_probe_pairs_medium,stim,LFP_dataset_gamma[:,LFP_responsive_electrodes_medium],'Linear SVM')

#high gamma

LFP_highgamma=bandpass_LFP(LFP,50,80)
LFP_dataset_highgamma=LFP_Response_dataset(LFP_highgamma,'m')
LFP_performance_array_highgamma=Discrimination_performance_array(LFP_probe_pairs_medium,stim,LFP_dataset_highgamma[:,LFP_responsive_electrodes_medium],'Linear SVM')

#plot performance vs cortical distance

bandpass_performance_corticaldistance_plotter(LFP_cortical_distances_medium,LFP_performance_array_theta,LFP_performance_array_alpha,LFP_performance_array_beta,LFP_performance_array_gamma,LFP_performance_array_highgamma,'Bandpass Performance','Performance_bandpass_LFP.png')

separation_dist=4
Bandpass_Performance_eccentricity_plotter(LFP_distance_array_medium,LFP_ecc_pairs_medium,LFP_performance_array_theta,LFP_performance_array_alpha,LFP_performance_array_beta,LFP_performance_array_gamma,LFP_performance_array_highgamma,separation_dist,'4 deg separation','LFP_Bandpass_Performance_eccentricity_four.png')

separation_dist=8
Bandpass_Performance_eccentricity_plotter(LFP_distance_array_medium,LFP_ecc_pairs_medium,LFP_performance_array_theta,LFP_performance_array_alpha,LFP_performance_array_beta,LFP_performance_array_gamma,LFP_performance_array_highgamma,separation_dist,'8 deg separation','LFP_Bandpass_Performance_eccentricity_eight.png')


###################### Section 5-1: Correlation-blind performance

#all the responsive electrodes

MUA_performance_array_blind=Discrimination_performance_array_onblind(MUA_probe_pairs_wide,stim,MUA_dataset[:,MUA_responsive_electrodes_wide])
LFP_performance_array_blind=Discrimination_performance_array_onblind(LFP_probe_pairs_medium,stim,LFP_dataset[:,LFP_responsive_electrodes_medium])

#six most important responsive electrodes

MUA_performance_array_importance_six_blind=Discrimination_performance_array_onblind(MUA_probe_pairs_wide,stim,MUA_dataset[:,MUA_weightsorted_electrodes[0:6]])
LFP_performance_array_importance_six_blind=Discrimination_performance_array_onblind(LFP_probe_pairs_medium,stim,LFP_dataset[:,LFP_weightsorted_electrodes[0:6]])

#Plot discrimination performance vs cortical distance

performance_corticaldistance_onblind_plotter(MUA_cortical_distances_wide,MUA_performance_array_wide,MUA_performance_array_blind,(0.435, 0.043, 0.043),(0.992, 0.576, 0.070),'MUA','MUA_performance_corticaldist_correlationblind.png')
performance_corticaldistance_onblind_plotter(LFP_cortical_distances_medium,LFP_performance_array_medium,LFP_performance_array_blind,(0.145, 0.188, 0.894),(0.964, 0.070, 0.992),'LFP','LFP_performance_corticaldist_correlationblind.png')

#Plot discrimination performance vs Eccentricity

separation_dist=4
Performance_eccentricity_onblind_plotter(MUA_distance_array_wide,MUA_ecc_pairs_wide,MUA_performance_array_wide,MUA_performance_array_blind,separation_dist,(0.435, 0.043, 0.043),(0.992, 0.576, 0.070),'MUA 4 deg separation','MUA_performance_eccentricity_correlationblind_fourdeg.png')

separation_dist=8
Performance_eccentricity_onblind_plotter(MUA_distance_array_wide,MUA_ecc_pairs_wide,MUA_performance_array_wide,MUA_performance_array_blind,separation_dist,(0.435, 0.043, 0.043),(0.992, 0.576, 0.070),'MUA 8 deg separation','MUA_performance_eccentricity_correlationblind_eightdeg.png')


separation_dist=4
Performance_eccentricity_onblind_plotter(LFP_distance_array_medium,LFP_ecc_pairs_medium,LFP_performance_array_medium,LFP_performance_array_blind,separation_dist,(0.145, 0.188, 0.894),(0.964, 0.070, 0.992),'LFP 4 deg separation','LFP_performance_eccentricity_correlationblind_fourdeg.png')

separation_dist=8
Performance_eccentricity_onblind_plotter(LFP_distance_array_medium,LFP_ecc_pairs_medium,LFP_performance_array_medium,LFP_performance_array_blind,separation_dist,(0.145, 0.188, 0.894),(0.964, 0.070, 0.992),'LFP 8 deg separation','LFP_performance_eccentricity_correlationblind_eightdeg.png')


#Plot discrimination performance vs cortical distance for six most important responsive electrodes

performance_corticaldistance_onblind_plotter(MUA_cortical_distances_wide,MUA_performance_array_importance_six_selected,MUA_performance_array_importance_six_blind,(0.435, 0.043, 0.043),(0.992, 0.576, 0.070),'MUA six best electrodes','MUA_performance_importance_six_corticaldist_correlationblind.png')
performance_corticaldistance_onblind_plotter(LFP_cortical_distances_medium,LFP_performance_array_importance_six_selected,LFP_performance_array_importance_six_blind,(0.145, 0.188, 0.894),(0.964, 0.070, 0.992),'LFP six best electrodes','LFP_performance_importance_six_corticaldist_correlationblind.png')


###################### Section 5-2: Noise correlation analysis

#build the set of responsive electrode pairs

MUA_electrode_pairs = list(itertools.combinations(list(MUA_responsive_electrodes_wide),2))
MUA_electrode_pairs = np.array(MUA_electrode_pairs,dtype=int)

LFP_electrode_pairs = list(itertools.combinations(list(LFP_responsive_electrodes_medium),2))
LFP_electrode_pairs = np.array(LFP_electrode_pairs,dtype=int)

#Noise correlation for all the pairs of responsive electrode (row) and
#for all the probes (column)

MUA_nc_array=MUA_all_probes_all_pairs_noise_correlation(MUA,stim,MUA_responsive_electrodes_wide,MUA_responsive_probes_wide)
MUA_nc_array=np.nan_to_num(MUA_nc_array)
LFP_nc_array=LFP_all_probes_all_pairs_noise_correlation(LFP,stim,LFP_responsive_electrodes_medium,LFP_responsive_probes_medium)

#Cortical distances of all the pairs of responsive electrodes

MUA_electrodes_distance=all_pairs_electrodes_distances(plxarray,MUA_responsive_electrodes_wide)
LFP_electrodes_distance=all_pairs_electrodes_distances(plxarray,LFP_responsive_electrodes_medium)

#Select the best two electrodes for MUA and two for LFP and plot their 
#noise correlation corresponding to each probe position 

#MUA: 58, 10 which is pair 118

electrode=58
tuning_plotter(stim,MUA_dataset,electrode,-0.27,0.27,'MUA_tuning_curve_electrode_58.png')
electrode=10
tuning_plotter(stim,MUA_dataset,electrode,-0.27,0.27,'MUA_tuning_curve_electrode_10.png')


MUA_probes_nc=MUA_nc_array[:,118]
MUA_probes_nc[MUA_probes_nc==0] = np.nan
Probes_noise_correlation_plotter(MUA_probes_nc,np.nanmean(MUA_probes_nc)-1.5*np.nanstd(MUA_probes_nc),np.nanmean(MUA_probes_nc)+1.5*np.nanstd(MUA_probes_nc),'MUA_probes_nc.png')

#LFP: 10, 48 which is pair 772

electrode=10
tuning_plotter(stim,-LFP_dataset,electrode,-0.34,0.34,'LFP_tuning_curve_electrode_10.png')
electrode=48
tuning_plotter(stim,-LFP_dataset,electrode,-0.34,0.34,'LFP_tuning_curve_electrode_48.png')


LFP_probes_nc=LFP_nc_array[:,772]
Probes_noise_correlation_plotter(LFP_probes_nc,np.nanmean(LFP_probes_nc)-1.5*np.nanstd(LFP_probes_nc),np.nanmean(LFP_probes_nc)+1.5*np.nanstd(LFP_probes_nc),'LFP_probes_nc.png')

#Plot noise correlation vs electrodes distances in response to probes at
#different eccentricities

#Under 5 deg ecc: 91, 81
#between 8 and 12 deg ecc: 72, 83
#between 16 and 20 deg ecc: 53, 75

noisecorrelation_distance_plotter(MUA_nc_array[np.array([91, 81, 72, 83, 53, 75]),:],MUA_electrodes_distance,-0.3,0.3,'MUA','MUA_Noisecorrelation_electrodes_distance.png')
noisecorrelation_distance_plotter(LFP_nc_array[np.array([91, 81, 72, 83, 53, 75]),:],LFP_electrodes_distance,0.2,0.8,'LFP','LFP_Noisecorrelation_electrodes_distance.png')

#plot noise correlation vs electrodes mean importance

MUA_electrodes_importance=electrodes_importance_calculator(MUA_weight_set)
MUA_pair_mean_importance=electrode_pairs_importance(MUA_electrode_pairs,MUA_electrodes_importance,MUA_responsive_electrodes_wide)
noisecorrelation_mean_importance_plotter(MUA_nc_array[np.array([91, 81, 72, 83, 53, 75]),:],MUA_pair_mean_importance,np.arange(0.02,0.095,0.015),np.arange(0.02,0.08,0.015)+0.0075,-0.3,0.3,'MUA','MUA_Noisecorrelation_electrodes_mean_importance.png')


LFP_electrodes_importance=electrodes_importance_calculator(LFP_weight_set)
LFP_pair_mean_importance=electrode_pairs_importance(LFP_electrode_pairs,LFP_electrodes_importance,LFP_responsive_electrodes_medium)
noisecorrelation_mean_importance_plotter(LFP_nc_array[np.array([91, 81, 72, 83, 53, 75]),:],LFP_pair_mean_importance,np.arange(0.004,0.024,0.004),np.arange(0.004,0.02,0.004)+0.002,0.2,0.8,'LFP','LFP_Noisecorrelation_electrodes_mean_importance.png')

