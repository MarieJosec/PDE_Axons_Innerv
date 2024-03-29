# library of functions to compute the coupled PDE-ODE system for pancreatic cancer progression and axon interaction
# Marie Jose Chaaya (2024) marie-jose.chaaya@univ-amu.fr
# Mathieu Mezache (2024) 
import numpy as np
import numba
import matplotlib.pyplot as plt



# Parameters setting
def set_parameters(par):
    """
    sets the parameters

    Args:
        par (list): sets of parameters not set 
    Returns:
        out (array float): sets of parameters set to be used for computing the model
    """
    out = np.zeros((18,))
    # Transport
    out[0] = par[0]  # \pi_0
    out[1] = par[1]  # \beta
    out[2] = par[2]  # \delta
    out[3] = 0  # \epsilon_0
    out[4] = 0  # s_\epsilon
    # Proliferaition
    out[5] = par[3]  # \gamma_R
    out[6] = par[4]  # s_R
    out[7] = par[5]  # \tau_C
    out[8] = par[6]  # \mu_1
    out[9] = par[7]  # \mu_2
    # Axon growth
    out[10] = 0.154442  # \A_1^eq
    out[11] = par[8]  # r_A1
    out[12] = par[9]  # r_A2
    # Transport function $\pi$ parameters
    out[13] = par[10]  # x_1
    out[14] = par[11]  # \epsilon_{1,\pi}
    # Allee effect parameters
    out[15] = par[12]  # s_theta
    # Initial distribution parameters
    out[16] = 0.8  # s_init
    # Sensory axons growth parameter
    out[17] = par[13]  # s_{A_2}
    return out


def denerv_param(par, axondenerv):
    """modifying the parameters to apply in silico denervation based on the type of axon denervated 

    Args:
        par (array float): sets of parameters that needs to be modified based on the denervation type
        axondenerv (string): the type of axon to be denervated, only sympathetic, only sensory, or both axons
    """

    out = np.copy(par)

    if axondenerv == "A1":
        out[1] = 0
        out[8] = 0
        out[11] = 0

    if axondenerv == "A2":
        out[2] = 0
        out[9] = 0
        out[12] = 0
    if axondenerv == "A1A2":
        out[1] = 0
        out[2] = 0
        out[8] = 0
        out[9] = 0
        out[11] = 0
        out[12] = 0
    return out

#--------------------------------------------------------
# System computation

def init_distribution(xhalf,L,par):
    """
    A function that determines the initial distribution of Q

    Args:
        xhalf (1-D vector): a vector of points that discritize the space
        L (int): bound of the domain
        par (array float): set of parameters used

    Returns:
        The initial distribution of Q for all values in xhalf
    """    
    return 100 * np.exp(-((xhalf + par[16] * L) ** 2 / 4)) / np.sqrt(4 * np.pi)

def pi_transport(x, par):
    """
    A function that corresponds to pi(x) in the model

    Args:
        x (1-D vector): a vector of points that discritize the space
        par (array float): set of parameters used

    Returns:
        The function pi(x) evaluated for all values in x
    """    
    #out = (np.tanh(par[14] * (x + par[13])) + np.tanh(10 * (40 - x))) / 2
    out = (np.tanh(par[14] * (x + par[13])) + np.tanh(0.1 * (20 - x))) / 2
    return out



def compute_pde(tf1, par, init_data):
    """Numerical computation of the dynamical system

    Args:
        tf1 (int): Final time of computation
        par (array float): tuple of parameters of the dynamical system
        init_data (list): Initial data for the dynamical system
    """
    # Discretization inputs
    dx = 0.05 # space step
    
    L = 50 # space bounds
    x = np.arange(-L,L+dx,dx)# vector discretized space
    N = len(x) - 1 # max size space vector
    xhalf = (x[0:N] + x[1:N +1])/2# vector middle points
    
    k = np.abs(x)
    xcind = k.argmin()  

    # Time step fulfilling the CFL
    #TO CHANGE TO FULFILL BETA DELTA MOVING NOW
    dt = min(0.8 * dx / (par[0] * (1 + par[2])), 0.005)
    # Time and space discretization
    t = np.arange(0, tf1 + dt/2, dt)
    NT = len(t)
    # stability conditions
    cfl = dt / dx
    # Initialization variables
    A1 = np.zeros((NT,))
    A2 = np.zeros((NT,))
    Q = np.zeros((N,))
    # Initialization
    A1[0] = init_data[0]  # initial state A1(0)=0.6
    A2[0] = init_data[1]  # initial state A2(0)=0.2
    Q = init_distribution(xhalf,L,par)[: len(xhalf)]
    init_first_mom = dx * (Q[0] / 2 + np.sum(Q[1:-1]) + Q[-1] / 2)
    # Indices on the discretize scales
    # advection and reaction terms
    transfer = par[0] * pi_transport(x , par)
    epsilon = par[3] * np.exp(-par[4] * ((x + L) ** 2))
    prol = 0.5 * (par[5]) * (1 + np.tanh(par[6] * xhalf))
    # Compute the PDE and return the necessary values for the cost function
    A1, A2, NT0, NTc, Q, All_Q = loop_pde(
        dt,
        cfl,
        NT,
        transfer,
        prol,
        epsilon,
        par,
        A1,
        A2,
        Q,
        init_first_mom,
        dx,
        N,
        xcind
    )
    return (Q, t, dt, A1, A2, NTc, NT0,xhalf,All_Q)   



#--------------------------------------------------------
# In silico denervation 
def compute_pde_den(tf1, dt, tf2, par_old, par, init_data):
    """
    Numerical computation of the in silico denervation of the dynamical system

    Args:
        tf1 (int): time at which we do the first denervation for a particular type of axon
        dt (float): time step
        tf2 (int): final time
        par_old (array float): set of parameters used
        par (array float): set of parameters obtained after first denervation at time tf1
        init_data (list): initial data for A1 and A2

    Returns:
        td : a 1-D vector of time discritized
        NT0d : 1-D time vector for NT0d total concentration of cells 
        NTcd : 1-D time vector for the total concentration of cancer cells 
        all_Q : a list of arrays that saves the cell distribution at each time step
    """



    # Discretization inputs
    dx = 0.05 # space step
    
    L = 50 # space bounds
    x = np.arange(-L,L+dx,dx)# vector discretized space
    N = len(x) - 1 # max size space vector
    xhalf = (x[0:N] + x[1:N +1])/2# vector middle points
    
    k = np.abs(x)
    xcind = k.argmin()  


    # stability conditions
    cfl = dt /dx
    # Initialization variables

    Qold, _told, _dtold, A1old, A2old, _NTcold, NT0old , xhalf,All_Q= compute_pde(
        tf1, 
        par_old, 
        init_data
    )
    
    mom = NT0old[0]
    
    
    t = np.arange(_told[-1], tf2 + dt/2, dt)
    NTnew = len(t)
    A1 = np.zeros((NTnew,))
    A2 = np.zeros((NTnew,))
    Q = np.zeros((N,))
    # Initialization
    A2[0] = A2old[-1]  # initial state
    A1[0] = A1old[-1]
    Q = Qold
    # advection and reaction terms
    transfer = par[0] * pi_transport(x,par)
    epsilon = par[3] * np.exp(-par[4] * ((x + L) ** 2))
    prol = 0.5 * (par[5]) * (1 + np.tanh(par[6] * xhalf))



    _A1, _A2, NT0, NTc, Q , t_Q = loop_pde(
        dt, 
        cfl, 
        NTnew,  
        transfer, 
        prol, 
        epsilon, 
        par, 
        A1, 
        A2, 
        Q, 
        mom,
        dx,
        N,
        xcind
    )
    td = np.concatenate((_told[:-1],t))
    NTcd = np.concatenate((_NTcold[:-1],NTc))
    NT0d = np.concatenate((NT0old[:-1], NT0))
    all_Q = np.concatenate((All_Q[:-1], t_Q))

    return (td, NT0d, NTcd, all_Q)

def compute_pde_both_den(tf1,dt, tf2, tf3, par_old,par,par2, init_data):
    """
    Numerical computation of the in silico denervation of the dynamical system

    Args:
        tf1 (int): time at which we do the first denervation for a particular type of axon
        dt (float): time step
        tf2 (int): time at which we do the second denervation  for a particular type of axon following the first
        tf3 (int): final time
        par_old (array float): set of parameters used
        par (array float): set of parameters obtained after first denervation at time tf1
        par2 (array float): set of parameters obtained after second denervation at time tf2
        init_data (list): initial data for A1 and A2

    Returns:
        td : a 1-D vector of time discritized
        NT0d : 1-D time vector for NT0d total concentration of cells 
        NTcd : 1-D time vector for the total concentration of cancer cells 
        all_Q : a list of arrays that saves the cell distribution at each time step
    """



    # Discretization inputs
    dx = 0.05 # space step
    
    L = 50 # space bounds
    x = np.arange(-L,L+dx,dx)# vector discretized space
    N = len(x) - 1 # max size space vector
    xhalf = (x[0:N] + x[1:N +1])/2# vector middle points
    
    k = np.abs(x)
    xcind = k.argmin()  


    # stability conditions
    cfl = dt /dx
    # Initialization variables

    Qold, _told, _dtold, A1old, A2old, _NTcold, NT0old , xhalf,All_Q= compute_pde(
        tf1, 
        par_old, 
        init_data
    )
    
    mom = NT0old[0]
    
    
    t = np.arange(_told[-1], tf2 + dt/2, dt)
    NTnew = len(t)
    A1 = np.zeros((NTnew,))
    A2 = np.zeros((NTnew,))
    Q = np.zeros((N,))
    # Initialization
    A2[0] = A2old[-1]  # initial state
    A1[0] = A1old[-1]
    Q = Qold
    # advection and reaction terms
    transfer = par[0] * pi_transport(x,par)
    epsilon = par[3] * np.exp(-par[4] * ((x + L) ** 2))
    prol = 0.5 * (par[5]) * (1 + np.tanh(par[6] * xhalf))



    _A1, _A2, NT0, NTc, Q , t_Q = loop_pde(
        dt, 
        cfl, 
        NTnew,  
        transfer, 
        prol, 
        epsilon, 
        par, 
        A1, 
        A2, 
        Q, 
        mom,
        dx,
        N,
        xcind
    )

    cfl = dt / dx
    tn = np.arange(t[-1], tf3 + dt/2, dt)
    NTn = len(tn)
    A1den = np.zeros((NTn,))
    A2den = np.zeros((NTn,))
    Qden = np.zeros((N,))
    # Initialization
    A2den[0] = _A2[-1]  # initial state
    A1den[0] = _A1[-1]
    Qden = Q
    # advection and reaction terms
    transferd = par2[0] * pi_transport(x,par2)
    epsilond = par2[3] * np.exp(-par2[4] * ((x + L) ** 2))
    prold = 0.5 * (par2[5]) * (1 + np.tanh(par2[6] * xhalf))
    _A1den, _A2den, NT0den, NTcden, Qden, t_Qn = loop_pde(
        dt, 
        cfl, 
        NTn,  
        transferd, 
        prold, 
        epsilond, 
        par2, 
        A1den, 
        A2den, 
        Qden, 
        mom,
        dx,
        N,
        xcind
    )

    td = np.concatenate((_told[:-1],t[:-1],tn))
    NTcd = np.concatenate((_NTcold[:-1],NTc[:-1],NTcden))
    NT0d = np.concatenate((NT0old[:-1], NT0[:-1],NT0den))
    all_Q = np.concatenate((All_Q[:-1], t_Q[:-1],t_Qn))

    return (td, NT0d, NTcd, all_Q)



@staticmethod
@numba.jit(nopython=True)
def loop_pde(dt, cfl, NT, transfer, prol, epsilon, par, A1, A2, Q, mom,dx,N,xcind):
    """
    Loop over the discritized time for the descritized model that solves the coupled system

    Args:
        dt (float): time step
        cfl (float): multiplicative coefficient
        NT (int):  length of the time vector
        transfer (float): return of pi_transport function multiplied by parameter \pi_0
        prol (float): return of  function multiplies by parameter \pi_0
        epsilon (float): term found in the speed term of the model 
        par (array float): tuple of the parameters of the dynamical system
        A1 (1-D time vector): sympathetic axons density
        A2 (1-D time vector): sensory axons density
        Q (1-D time vector): concentration of cells at time t with phenotype x
        mom (float): saves the initial value of NT0 the total concentration of cells present
        dx (float): space step
        N (int): length of the space vector
        xcind (float): taken to be the closest to 0 

    Returns:
        A1 : 1-D time vector for A1
        A2 : 1-D time vector for A2
        NT0 : 1-D time vector for the total concentration of cells 
        NTc : 1-D time vector for the total concentration of cancerous cells 
        Q : 1-D space vector for the phenotypic distribution at the final time
        all_Q : a list of arrays that saves the cell distribution at each time step
    """    

    # Initialization of the video outputs
    all_Q = []
    # Initialization of the ouputs
    NT0 = np.zeros((NT,))
    NTc = np.zeros((NT,))
    Fplus = np.zeros((N + 1,))
    AX = np.log(A1)
    AY = np.log(A2)
    # Loop to compute the time evolution of the dynamical system
    for n in np.arange(0, NT - 1):
        
        Q_old = np.copy(Q)
        all_Q.append(np.copy(Q_old))
        NT0[n] = dx * (
            Q_old[0] / 2 
            + np.sum(Q_old[1:-1]) 
            + Q_old[-1] / 2
        )
        NTc[n] = dx * (
            Q_old[xcind] / 2 
            + np.sum(Q_old[xcind + 1 : -1]) 
            + Q_old[-1] / 2
        )
        Fplus[0] = 0
        Q[0] = 0
        rho_A1 = max(A1[n] - par[10], 0)
        Fplus[1 : N + 1] = transfer[1 : N + 1] * (
            1 - par[1] * rho_A1 + par[2] * A2[n]
        ) + epsilon[1 : N + 1] * (NTc[n] / NT0[n])
        temp_prol = (
            dt
            * prol[1:N]
            * Q_old[1:N]
            * (1 - (NT0[n] / par[7]) - par[8] * A1[n] + par[9] * A2[n])
        )
        Q[1:N] = (
            (1 - cfl * Fplus[2 : N + 1]) * Q_old[1:N]
            + cfl * Fplus[1 : N] * Q_old[0 : N - 1]
            + temp_prol
        )
        Theta = (
            par[10] / 2
            + (np.tanh(par[15] * (NT0[n] / mom - 1 - 0.1))) / 2.0
            + 0.5
        )
        # Finite differences for the axons
        # 1. Change of variables
        AX[n + 1] = AX[n] + dt * par[11] * (np.exp(AX[n]) / Theta - 1) * (
            1 - np.exp(AX[n])
        )
        y = np.tanh(par[17] * (NTc[n] / NT0[n]))
        AY[n + 1] = AY[n] + dt * par[12] * y * (1 - np.exp(AY[n]))
        # 2. Get back to the original variables
        A1[n + 1] = np.exp(AX[n + 1])
        A2[n + 1] = np.exp(AY[n + 1])
        
    # Summation for the underlying variables
    NT0[NT - 1] = dx * (Q[0] / 2 + np.sum(Q[1:-1]) + Q[-1] / 2)
    NTc[NT - 1] = dx * (
        Q[xcind] / 2 
        + np.sum(Q[xcind + 1 : -1]) 
        + Q[-1] / 2
    )
    return (A1, A2, NT0, NTc, Q, all_Q)



#--------------------------------------------------------
# Visualization

def axon_plot(init_data, par):
    """
    A figure that shows the axons dynamics in time 

    Args:
        init_data (list): initial data for A1 and A2
        par (array float): set of parameters used 
    """    
    Q, t, dt, A1, A2, NTc, NT0,xhalf,All_Q = compute_pde(70, par, init_data)
    plt.figure()
    plt.rcParams['axes.formatter.min_exponent']=2
    plt.rcParams.update({'font.size':14})
    plt.rcParams['text.usetex'] = True
    plt.rc('legend', fontsize = 14)
    plt.plot(t,A1,color='black', label=r'$A_1$')
    plt.plot(t,A2, color='black',linestyle=':',label=r'$A_2$')
    plt.legend()
    plt.xlabel('t days')
    plt.grid()
    plt.ylim(0, 1)
    plt.show()
    return()

def plot_cancercells_den_1(init_data, par, t1_den, par_1):
    """
    A figure that shows the cancer cells dynamics in time for the control and for the denervated case

    Args:
        init_data (list): initial data for A1 and A2
        par (array float): set of parameters used 
        t1_den (integer): time of denervation
        par_1 (array float): set of parameters obtained after first denervation at time t1_den
    """    
    Q, t, dt, A1, A2, NTc, NT0,xhalf,All_Q = compute_pde(70, par, init_data)
    td, NT0d, NTcd, all_Q = compute_pde_den(t1_den,dt, 70, par,par_1, init_data)
    plt.figure()
    plt.plot(t,NTc, color='black', label='control')
    plt.plot(td,NTcd, color='red', label='denervated')
    plt.xlabel('t days', fontsize=15)
    plt.ylabel(r'$NT_c$', fontsize=15)
    plt.legend()
    plt.grid() 
    plt.tight_layout()
    plt.show()
    return()

def plot_cancercells_den(init_data, par, t1_den,t2_den, par_1,par_2):
    """
    A figure that shows the cancer cells dynamics in time for the control and for the denervated case

    Args:
        init_data (list): initial data for A1 and A2
        par (array float): set of parameters used 
        t1_den (integer): time at which we do the first denervation for a particular type of axon
        t2_den (integer): time at which we do the second denervation  for a particular type of axon following the first
        par_1 (array float): set of parameters obtained after first denervation at time t1_den
        par_2 (array float): set of parameters obtained after second denervation at time t2_den
    """    
    Q, t, dt, A1, A2, NTc, NT0,xhalf,All_Q = compute_pde(70, par, init_data)
    td, NT0d, NTcd, all_Q = compute_pde_both_den(t1_den,dt, t2_den, 70, par,par_1,par_2, init_data)
    plt.figure()
    plt.plot(t,NTc, color='black', label='control')
    plt.plot(td,NTcd, color='red', label='denervated')
    plt.xlabel('t days', fontsize=15)
    plt.ylabel(r'$NT_c$', fontsize=15)
    plt.legend()
    plt.grid() 
    plt.tight_layout()
    plt.show()
    return()



def plot_pdes(init_data, par):
    """
    A video of the evolution with repect to time of Q(t,x) over x

    Args:
        init_data (list): initial data for A1 and A2
        par (array float): set of parameters used for plotting
    """    
    Q, t, dt, A1, A2, NTc, NT0,xhalf,All_Q = compute_pde(70, par, init_data)
    skip_frame = int(len(All_Q)/100)
    n_end = int(len(All_Q)/skip_frame)
    plt.figure()
    for n in range(0,n_end):
        plt.clf()
        plt.subplot(111)
        plt.suptitle('t='+str(t[skip_frame*n]))
        plt.plot(xhalf,All_Q[skip_frame*n], color = 'blue', label='control')
        plt.xlabel('x')
        plt.ylabel('Q(t,x)')
        plt.legend()
        plt.pause(0.1)
           
    plt.show()   
    return()

def denerv_dyn_system(t1,t2,init_data,par):
    """
    In silico denervation at defined time points

    Args:
        t1 (float): time of denervation of A1
        t2 (float): time of denervation of A2
        init_data (list): initial data for A1 and A2
        par (array float): set of parameters used for plotting
    """  
    if (t1>t2):
        par_1 = denerv_param(par,'A2') 
        par_2 = denerv_param(par,'A1A2') 
        plot_cancercells_den(init_data, par, t2 , t1 , par_1, par_2) 

    if (t1<t2):
        par_1 = denerv_param(par,'A1') 
        par_2 = denerv_param(par,'A1A2') 
        plot_cancercells_den(init_data, par, t1 , t2 , par_1, par_2)

    if (t1==t2):
        par_d = denerv_param(par,'A1A2')
        plot_cancercells_den_1(init_data, par, t1, par_d)
    return()

