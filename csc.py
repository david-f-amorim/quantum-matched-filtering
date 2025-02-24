import numpy as np 
from scipy.special import betainc
from scipy import stats 
import matplotlib.pyplot as plt 
from matplotlib import rcParams

#----- VARIABLES -----

k = 100
rho = np.sqrt(0.5) # overlap, not overlap squared
v_min = 5/2
v_max = 9/2

def theta(j): 
    j_min =  int(0.5 * k * (1+ rho**2)- v_min * np.sqrt(k))
    j_max =  int(0.5 * k * (1+ rho**2)+ v_max * np.sqrt(k))

    return np.pi * (j >= j_max) + 0 * (j < j_min) + np.pi * ( j - j_min)/(np.sqrt(k)*(v_max + v_min)) * ( j < j_max) * (j >= j_min)   
  

#----- CONTROLS -----

show =True
save = True 
pdf = True

#----- CALCULATIONS -----
N = 1000
overlaps = np.linspace(0,1,N)
probs = 0.5 * (1 + overlaps)
fidelities = np.zeros(N, dtype="complex")

for i in np.arange(N):
    for j in np.arange(k+1):
        fidelities[i]+=np.exp(1j * theta(j)) * stats.binom.pmf(j,k,probs[i])

F = np.abs(fidelities)
arg = np.angle(fidelities)

#----- PLOTTING -----
rcParams['mathtext.fontset'] = 'stix' 
rcParams['font.family'] = 'STIXGeneral'
rcParams["text.usetex"] = True 

width=0.75 
fontsize=1.3*28 
titlesize=32
ticksize=1.3*22
figsize=(10,6)
pdf_str=".pdf" if pdf else ""

fig, ax1 = plt.subplots(figsize=figsize)

colour1 = 'tab:blue'
ax1.set_xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$', fontsize=fontsize)
ax1.set_ylabel(r'$\vert F_i \vert$', color=colour1, fontsize=fontsize)
ax1.plot(overlaps, F, color=colour1)
ax1.vlines(x=rho**2, ymin=0, ymax=1, linestyles="dashed", color="black")
ax1.tick_params(axis='y', labelcolor=colour1, labelsize=ticksize)
ax1.tick_params(axis='x', labelsize=ticksize)

ax2 = ax1.twinx()  
colour2 = 'tab:red'
ax2.set_ylabel(r'arg($F_i$)', color=colour2, fontsize=fontsize) 
ax2.plot(overlaps, arg, color=colour2)
ax2.tick_params(axis='y', labelcolor=colour2,labelsize=ticksize)
ax2.set_yticks(np.arange(- np.pi, np.pi+np.pi/2, step=(np.pi / 2)))
ax2.set_yticklabels([r'$-\pi$', r'$-0.5 \pi$', r'$0\pi$', r'$+0.5\pi$', r'$+\pi$'])

fig.tight_layout()  
if save:
    plt.savefig(f"complex_F{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close() 
