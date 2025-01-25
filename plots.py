import numpy as np 
from scipy.special import betainc
from scipy.stats import binom  
import matplotlib.pyplot as plt 
from matplotlib import rcParams


#----- CONTROLS -----

show =True
save = True 
pdf = True

k_plot = True
ck_plot = True
rho_plot = True

CSC_plot = True 

#----- CALCULATIONS -----
N = 1000
overlaps = np.linspace(0,1,N)
probs = 0.5 * (1 + overlaps)

if k_plot:
        
    k_arr = np.array([10,100,1000])
    rho = 0.5

    j_star_arr =(0.5 * k_arr * (1+ rho)).astype(int)
    fidelities_k = np.empty(len(k_arr), dtype=object)
    for i in np.arange(len(k_arr)):
        fidelities_k[i] = 2 * (1 - betainc(j_star_arr[i],k_arr[i]-j_star_arr[i] +1, probs) ) -1
    ideal = np.ones(N) - 2 * (overlaps >= rho)

if ck_plot: 

    k = 100
    rho = 0.5
    ck_arr = np.array([0, k**0.1, k**0.2])

    j_star_arr =(0.5 * k * (1+ rho) - 0.5 * ck_arr * np.sqrt(k)).astype(int)
    fidelities_ck = np.empty(len(ck_arr), dtype=object)
    for i in np.arange(len(ck_arr)):
        fidelities_ck[i] = 2 * (1 - betainc(j_star_arr[i],k-j_star_arr[i] +1, probs) ) -1

if rho_plot: 

    k = 100
    rho_arr = np.array([0.05,0.1,0.3,0.5])
    ck=0

    j_star_arr =(0.5 * k * (1+ rho_arr) - 0.5 * ck * np.sqrt(k)).astype(int)
    fidelities_rho = np.empty(len(rho_arr), dtype=object)
    for i in np.arange(len(rho_arr)):
        fidelities_rho[i] = 2 * (1 - betainc(j_star_arr[i],k-j_star_arr[i] +1, probs) ) -1        

#----- PLOTTING -----
rcParams['mathtext.fontset'] = 'stix' 
rcParams['font.family'] = 'STIXGeneral' 
rcParams["text.usetex"] = True

width=0.75 
fontsize=28 
titlesize=32
ticksize=22
figsize=(10,6)
pdf_str=".pdf" if pdf else ""

if k_plot:
    colours = ["tab:blue", "limegreen","tab:red", "Black"]
    styles = ["solid", "dashed", "dashdot", "solid"]
    labels = [r"$10$", r"$10^{2}$", r"$10^{3}$"]

    plt.figure(figsize=figsize)
    for i in np.arange(len(k_arr)):
        plt.plot(overlaps, fidelities_k[i], color=colours[i], label=r'$k=$'f'{labels[i]}', ls=styles[i])
    plt.plot(overlaps, ideal, color=colours[-1], label=r'$k \to \infty$', ls=styles[-1])
    plt.xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$',fontsize=fontsize)
    plt.ylabel(r'$F_i$',fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.legend(fontsize=titlesize)
    plt.tight_layout()
    if save:
        plt.savefig(f"fidelity_plot_k{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close()

if ck_plot:
    colours = ["tab:blue", "limegreen","tab:red", ]
    styles = ["solid", "dashed", "dashdot", ]
    labels = [r"$0$", r"$k^{0.1}$", r"$k^{0.2}$"]

    plt.figure(figsize=figsize)
    for i in np.arange(len(ck_arr)):
        plt.plot(overlaps, fidelities_ck[i], color=colours[i], label=r'$c(k)=$'f'{labels[i]}', ls=styles[i])

    min, max = plt.ylim()
    plt.ylim(min, max)
    plt.vlines(x=rho, ymin=min, ymax=max, colors="black", label=r'$\rho_{thr}$')
    plt.xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$',fontsize=fontsize)
    plt.ylabel(r'$F_i$',fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.legend(fontsize=titlesize)
    plt.tight_layout()
    if save:
        plt.savefig(f"fidelity_plot_ck{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close() 

if rho_plot:
    colours = ["tab:blue", "limegreen","tab:red", "black"]
    styles = ["solid", "dashed", "dashdot", "solid"]
    labels = [r"$0.05$", r"$0.1$", r"$0.3$", r"$0.5$"]

    plt.figure(figsize=figsize)
    for i in np.arange(len(rho_arr)):
        plt.plot(overlaps, fidelities_rho[i], color=colours[i], label=r'$\rho_{thr}=$'f'{labels[i]}', ls=styles[i])
    plt.xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$',fontsize=fontsize)
    plt.ylabel(r'$F_i$',fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.legend(fontsize=titlesize)
    plt.tight_layout()
    if save:
        plt.savefig(f"fidelity_plot_rho{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close()      

if CSC_plot:

    def arrowed_spines(fig, ax):

        xmin, xmax = ax.get_xlim() 
        ymin, ymax = ax.get_ylim()

        # removing the default axis on all sides:
        for side in ['bottom','right','top','left']:
            ax.spines[side].set_visible(False)

        # removing the axis ticks
        plt.xticks([]) # labels 
        plt.yticks([])
        ax.xaxis.set_ticks_position('none') # tick markers
        ax.yaxis.set_ticks_position('none')

        # get width and height of axes object to compute 
        # matching arrowhead length and width
        dps = fig.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(dps)
        width, height = bbox.width, bbox.height

        # manual arrowhead width and length
        hw = 1./20.*(ymax-ymin) 
        hl = 1./20.*(xmax-xmin)
        lw = 1. # axis line width
        ohg = 0.3 # arrow overhang

        # compute matching arrowhead length and width
        yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
        yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

        # draw x and y axis
        ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
                head_width=hw, head_length=hl, overhang = ohg, 
                length_includes_head= True, clip_on = False) 

        ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
                head_width=yhw, head_length=yhl, overhang = ohg, 
                length_includes_head= True, clip_on = False)


    #####
    k = 30 # 300
    j_star = k / 2
    rho = 0.2

    p = 0.5*(1+rho)
    x = np.arange(0, k, step=1) # step =3
    y = binom.pmf(x,k,p)
    
    ### BEFORE PLOT ###
    plt.figure(figsize=figsize)
    plt.scatter(x , np.sqrt(y),color="tab:blue")
    plt.vlines(x, 0, np.sqrt(y),colors="tab:blue")
    plt.xlim(0,k+1)
    fig = plt.gcf()
    fig.set_facecolor('white') 
    ax = plt.gca()
    plt.xlabel(r'$\vert j \rangle \vert \Omega_j \rangle$',fontsize=fontsize)
    plt.ylabel(r'Amplitude',fontsize=fontsize)
    arrowed_spines(fig, ax)

    plt.tight_layout()
    if save:
        plt.savefig(f"CSC_plot_before{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close() 

    ### AFTER PLOT ###
    plt.figure(figsize=figsize)
    plt.scatter(x * (x >= j_star ), -np.sqrt(y)* (x >= j_star ),color="tab:red")
    plt.vlines(x* (x >= j_star ), 0, -np.sqrt(y)* (x >= j_star ),colors="tab:red")
    plt.scatter(x * (x < j_star ), np.sqrt(y)* (x < j_star ),color="tab:blue")
    plt.vlines(x* (x < j_star ), 0, np.sqrt(y)* (x < j_star ),colors="tab:blue")
    plt.xlim(0,k+1)
    fig = plt.gcf()
    fig.set_facecolor('white') 
    ax = plt.gca()
    plt.xlabel(r'$\vert j \rangle \vert \Omega_j \rangle$',fontsize=fontsize)
    plt.ylabel(r'Amplitude',fontsize=fontsize)
    arrowed_spines(fig, ax)

    plt.tight_layout()
    if save:
        plt.savefig(f"CSC_plot_after{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close() 

    
