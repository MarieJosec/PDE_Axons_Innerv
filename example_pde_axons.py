# python program computing the PDE-ODE coupled system for pancreatic cancer progression and axon interaction 
# Marie Jose Chaaya (2024) marie-jose.chaaya@univ-amu.fr
# Mathieu Mezache (2024) 
import numpy as np

import functions_pde_axons as fct



def parameters_used():
    """
    Sets of parameters used to compute the coupled model

    Returns:
        list of lists consisting of values of the parameters
    """    
    all_par =  [   [2.005,0.73,0.398,1.5,2.68,172.295,-0.176,0.214,0.055,0.241,32.97,5.357,15.131,4.151],
                   [4.589,0.504,0.829,1.160,2.775,177.807,0.609,0.139,0.077,0.928,34.56,6.555,14.733,1.105],
                   [1.795,0.535,0.398,4.537,6.097,150.576,0.176,0.678,0.032,0.29,30,4.385,17.609,6.49]
                   ] 
    return all_par


if __name__ == "__main__":
    plt.close('all')
    A1_0 = 0.15445 
    A2_0 = 0.004
    init_data = [A1_0, A2_0] # list of initial consitions for A1 and A2
    
    temp = parameters_used()  # we define the sets of parameters that we wish to test
    index_set_parameters = 0
    par = fct.set_parameters(temp[index_set_parameters]) # we choose a set of parameters from all sets of parameters provided

    # Dynamical system 
    fct.axon_plot(init_data, par) # we plot the axons remodelling over time
    fct.plot_pdes(init_data, par)   # we plot the dynamics of the evolution of Q with respect to time 

    # Denervated system
    # The following in silico denervation times should be between 0 and 70 as the final time is 70.
    time_denerv1 = 40 # time chosen for the denervation of sympathetic axons
    time_denerv2 = 60 # time chosen for the denervation of sensory axons
    # we plot the evolution of cancer cells subject to the in silico denervation
    denerv_dyn_system(time_denerv1,time_denerv2,init_data,par) 
   
    
