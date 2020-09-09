import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from probes_coord import dim_calculator


def stimulus_averaged_responses(stim, dataset):
    """Average response to a stimulus
    This function calculates the average response to each stimulus. For each 
    probe location, 96 response values (each for an electrode) are calculated.
    """
    averaged_responses=np.zeros((100,96))
    for i in np.arange(1,101):
        averaged_responses[i-1,:]=np.mean(dataset[stim==i,:],axis=0)
    return averaged_responses


def RF(averaged_responses,electrode):
    """Electrode receptive field data
    This function chooses the mean responses of an electrode to all the probe 
    locations.
    """
    RF_electrode=averaged_responses[:,electrode]
    return RF_electrode


def RF_plotter(RF_electrode,v_min,v_max,filename):
    """Receptive field Plotter
    This function plots receptive field of an electrode
    """
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


def RF_set_viewer(stim,dataset,electrode_set,v_min,v_max,filename):
    """Group RF plotter
    This function plots receptive fields of a group of electrodes
    """
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
    """Tuning curve plotter
    Plots tuning curve (region) of an electrode
    """
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


def tuning_plotterP3(stim,dataset,electrode,v_min,v_max,filename):
    """Tuning curve plotter
    Plots tuning curve (region) of an electrode
    """
    averaged_responses=stimulus_averaged_responses(stim,dataset)
    RF_electrode=RF(averaged_responses,electrode)
    RF_electrode=np.reshape(RF_electrode,(10,10)).T
    fig=plt.figure(figsize=(5, 5))
    plt.imshow(RF_electrode,vmin=v_min, vmax=v_max)
    plt.xticks(np.array([0,9]),[-36+20,0+20])
    plt.yticks(np.array([0,9]),[2,-34])
    plt.xlabel('Position(deg)',fontweight='bold')
    plt.ylabel('Position(deg)',fontweight='bold')
    plt.subplots_adjust()
    cax = plt.axes([1, 0.23, 0.03, 0.1])
    plt.colorbar(cax=cax).set_ticks([v_min,v_max])
    plt.show()
    fig.savefig(filename, dpi=150)


def two_d_gaussian(x,A,x1_0,x2_0,sigma_x1,sigma_x2,d):
    """2D Gaussian
    2D Gaussian function that takes x as input and outputs a corresponding 
    Gaussian value. This will be called by gaussian_RF function.
    """
    return A*np.exp(-((x[:,0]-x1_0)**2/(2*sigma_x1**2))-((x[:,1]-x2_0)**2/(2*sigma_x2**2)))+d


def gaussian_RF(RF_electrode,x1_0,x2_0,sigma_x1,sigma_x2):
    """
    This function fits a 2D Gaussian to the Receptive field of an electrode in 
    terms of its x-y locations on the grid
    """
    dim=dim_calculator()
    popt,pcov = curve_fit(two_d_gaussian,dim,RF_electrode,p0=[1,x1_0,x2_0,sigma_x1,sigma_x2,0],bounds=([-2,-36,-34,2,2,-2], [3,0,2,30,30,2]))
    RF_gaussian=two_d_gaussian(dim,*popt)
    return RF_gaussian,popt

def gaussian_RF_P3(RF_electrode,x1_0,x2_0,sigma_x1,sigma_x2):
    """
    This function fits a 2D Gaussian to the Receptive field of an electrode in 
    terms of its x-y locations on the grid
    """
    dim=dim_calculator()
    popt,pcov = curve_fit(two_d_gaussian,dim,RF_electrode,p0=[1,x1_0,x2_0,sigma_x1,sigma_x2,0],bounds=([-2,-36+20,-34,2,2,-2], [3,0+20,2,30,30,2]))
    RF_gaussian=two_d_gaussian(dim,*popt)
    return RF_gaussian,popt

def FWHM_gaussian(param_G):
    """Gaussian FWHM
    This function calculates full width at half maximum (FWHM) of the fitted 
    Gaussian to a receptive field. It takes as input parameters of the fitted 
    Gaussian
    """
    X=np.arange(-36,4,4)
    Y=np.arange(2,-38,-4)
    X,Y = np.meshgrid(X,Y)
    Z=param_G[0]*np.exp(-((X-param_G[1])**2/(2*param_G[3]**2))-((Y-param_G[2])**2/(2*param_G[4]**2)))+param_G[5]-(param_G[5]+param_G[0])/2
    return X,Y,Z

def FWHM_gaussian_P3(param_G):
    """Gaussian FWHM
    This function calculates full width at half maximum (FWHM) of the fitted 
    Gaussian to a receptive field. It takes as input parameters of the fitted 
    Gaussian
    """
    X=np.arange(-36+20,4+20,4)
    Y=np.arange(2,-38,-4)
    X,Y = np.meshgrid(X,Y)
    Z=param_G[0]*np.exp(-((X-param_G[1])**2/(2*param_G[3]**2))-((Y-param_G[2])**2/(2*param_G[4]**2)))+param_G[5]-(param_G[5]+param_G[0])/2
    return X,Y,Z

def RF_diameter(param_G):
    """RF diameter
    This function takes the parameters of the fitted Gaussian to calculate 
    diameters of FWHM ellipse
    """
    Dx=2*np.abs(param_G[3])*np.sqrt(-2*np.log(1/2-(1/2)*(param_G[5]/param_G[0])))
    Dy=2*np.abs(param_G[4])*np.sqrt(-2*np.log(1/2-(1/2)*(param_G[5]/param_G[0])))
    return Dx,Dy


def population_contours(dataset,stim,electrode_set,sigm,clr):
    """RF contours from a group of electrodes
    This function plots fitted FWHM ellipse to receptive fields of a group of 
    electrodes. In addition, it returns mean RF diameter of all the electrodes 
    in the electrode_set.
    """
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


def RFCenterPosition(dataset,stim,electrode_set,sigm):
    """RF centers
    This function determines coordinations of the receptive fields centers 
    obtained by the fitted Gaussian
    """
    dim=dim_calculator()
    averaged_responses=stimulus_averaged_responses(stim,dataset)
    RF_center_position=np.zeros((len(electrode_set),2))
    c=0
    for electrode in electrode_set:
        RF_electrode=RF(averaged_responses,electrode)
        ind=np.argmax(RF_electrode)
        x1_0,x2_0,sigma_x1,sigma_x2=dim[ind,0],dim[ind,1],sigm,sigm
        RF_G,param_G=gaussian_RF(RF_electrode,x1_0,x2_0,sigma_x1,sigma_x2)
        RF_center_position[c,:]=param_G[1],param_G[2]
        c=c+1
    return RF_center_position


def RFCenterEccentricity(RF_center_position):
    """RF centers eccentricity
    This function calculates eccentricity of the receptive fields centers
    """
    RF_center_eccentricity=np.sqrt(RF_center_position[:,0]**2+RF_center_position[:,1]**2)
    return RF_center_eccentricity


def RFDiameterEccentricityPlot(RF_center_eccentricity,RF_mean_diameter,trace_color,figure_title,file_name):
    """RF diameters vs eccentricity
    This function plots the diameter of the receptive fields versus their 
    center eccentricity
    """
    fig=plt.figure(figsize=(12, 8))
    ind=np.argsort(RF_center_eccentricity)
    plt.scatter(RF_center_eccentricity[ind],RF_mean_diameter[ind],marker="^", s=200, c=trace_color)
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0,40)
    plt.xlim(0,40)
    plt.ylabel('RF Diameter',fontweight='bold')
    plt.xlabel('RF Center Eccentricity',fontweight='bold')
    plt.title(figure_title,fontweight='bold',loc='center')
    plt.show()
    fig.savefig(file_name, dpi=350)


def ECConArray(RF_center_eccentricity,electrode_set,plxarray,filename):
    """Eccentricity plotted on the array
    This function plots the eccentricity of each electrode on its location on 
    the electrode array
    """
    fig=plt.figure(figsize=(9,12))
    Aray_ecc=np.zeros(100)
    Aray_ecc[electrode_set]=RF_center_eccentricity
    plt.imshow(Aray_ecc[plxarray-1],vmin=0, vmax=40)
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.title('RF Eccentricity',fontweight='bold')
    plt.subplots_adjust()
    cax = plt.axes([1, 0.25, 0.05, 0.2])
    plt.colorbar(cax=cax).set_ticks([0,40]) #rect = [left, bottom, width, height]
    plt.show()
    fig.savefig(filename, dpi=350)
