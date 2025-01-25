import numpy as np 
from scipy.special import erfc, betainc  
import matplotlib.pyplot as plt 
from matplotlib import rcParams


#----- VARIABLES -----

k_arr = np.array([10,100,1000])
rho = 0.5

#----- CONTROLS -----

show =True
save = True 
pdf = True

#----- CALCULATIONS -----
N = 1000
overlaps = np.linspace(0,1,N)
probs = 0.5 * (1 + overlaps)

j_star_arr =(0.5 * k_arr * (1+ rho)).astype(int)

fidelities = np.empty(len(k_arr), dtype=object)

for i in np.arange(len(k_arr)):
    fidelities[i] = 2 * (1 - betainc(j_star_arr[i],k_arr[i]-j_star_arr[i] +1, probs) ) -1

ideal = np.ones(N) - 2 * (overlaps >= rho)

#----- PLOTTING -----
rcParams['mathtext.fontset'] = 'stix' 
rcParams['font.family'] = 'STIXGeneral' 

width=0.75 
fontsize=28 
titlesize=32
ticksize=22
figsize=(12,10)
pdf_str=".pdf" if pdf else ""
plt.figure(figsize=figsize)


colours = ["tab:blue", "limegreen","tab:red", "Black"]
styles = ["solid", "dashed", "dashdot", "solid"]

labels = [r"$10$", r"$10^{2}$", r"$10^{3}$"]

for i in np.arange(len(k_arr)):
    plt.plot(overlaps, fidelities[i], color=colours[i], label=r'$k=$'f'{labels[i]}', ls=styles[i])
plt.plot(overlaps, ideal, color=colours[-1], label=r'$k \to \infty$', ls=styles[-1])

#plt.vlines(x=rho, ymin=-1.1, ymax=+1.2, colors="black", label=r'$\rho_{thr}$')
plt.xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$',fontsize=fontsize)
plt.ylabel(r'$F_i$',fontsize=fontsize)
plt.tick_params(axis="both", labelsize=ticksize)
plt.legend(fontsize=titlesize)
plt.tight_layout()
if save:
    plt.savefig(f"fidelity_plot{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close()