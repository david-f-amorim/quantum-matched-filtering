import numpy as np 
from scipy.special import erfc, betainc  
import matplotlib.pyplot as plt 
from matplotlib import rcParams

#----- CONTROLS -----

show =True
save = True 
pdf = True

k_plot = False
ck_plot = False 
rho_plot = False 

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