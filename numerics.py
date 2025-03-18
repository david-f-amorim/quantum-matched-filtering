import numpy as np 
import sys
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.special import erfc, betainc  
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'stix' 
rcParams['font.family'] = 'STIXGeneral' 

width=0.75 
fontsize=28 * 1.3
titlesize=32
ticksize=22 * 1.3
figsize=(10,6)
show =True
save = True 
pdf = True
pdf_str=".pdf" if pdf else ""

N = 10000
rho = np.sqrt(0.5) # realistic: 0.15 ;  overlap, not overlap squared!
k = 1000
use_theta1 = True
alternate = True 

v = 5
v_min = v/2
v_max = v/2 + 0.0001 * (v==0)

std = 0.3 * rho   # std of overlap distribution (assumed to be Gaussian with zero mean)

def c(k): 
    return 0 *k**(0.2)

j_star = int(0.5 * k * (1+ rho**2) - 0.5 * c(k) * np.sqrt(k))

if j_star ==0:
    print('\nWARNING: j_star=0\n')
elif j_star==k:
    print('\nWARNING: j_star=k\n')
if j_star <0:
    print('\nNegative j_star. Aborting.\n')
    sys.exit()

def theta(j): 
    j_min =  int(0.5 * k * (1+ rho**2)- v_min * np.sqrt(k))
    j_max =  int(0.5 * k * (1+ rho**2)+ v_max * np.sqrt(k))

    return np.pi * (j >= j_max) + 0 * (j < j_min) + np.pi * ( j - j_min)/(np.sqrt(k)*(v_max + v_min)) * ( j < j_max) * (j >= j_min)   
  
print("==================================")
print(f"k:\t{k}")
print(f"j_star:\t{j_star}")

## randomly sample overlaps 
rng = np.random.default_rng(seed = 12345678)
rho_arr = np.abs(rng.normal(size=N, scale=std)) 
rho_arr = np.sort(rho_arr)

## make sure overlaps less than one (in case std is large)
for i in np.arange(N):
    if rho_arr[i]>1:
        rho_arr[i]=1

if np.sum(rho_arr > 1):
    print("\nUnphysical overlap (>1). Aborting.\n")
    sys.exit() 

## count number of marked entries 
M = np.sum(rho_arr**2 >= rho**2)
if M == 0:
    print("\nNo matches. Aborting.\n")
    sys.exit() 

## count number of entries in set D_k 
D = np.sum( (rho_arr**2 >= rho**2- 2 * c(k)/ np.sqrt(k)) * (rho_arr**2 < rho**2) )
delta = D / N

## get number of iterations    
s = int(np.pi /4 * np.sqrt(N/M))

## get ideal Grover fidelities 
F_ideal = np.ones(N)-2* (rho_arr**2 >= rho**2)

## get CSO fidelities
p_0_arr = 0.5 * (1+ rho_arr**2)

if use_theta1 == False:
    F_CSO = 2 * (1 - betainc(j_star,k-j_star +1, p_0_arr) ) -1
else:
    fidelities = np.zeros(N, dtype="complex")
    theta_arr = np.array([theta(j) for j in np.arange(k+1)])
    phase_arr = np.cos(theta_arr)+1j * np.sin(theta_arr)

    for i in np.arange(N):
        arr = np.array([ phase_arr[j]* stats.binom._pmf(j,k,p_0_arr[i]) for j in np.arange(k+1)],dtype="complex")
        fidelities[i]= np.sum(arr)

    F = np.abs(fidelities)
    arg = np.angle(fidelities)

    F_CSO = fidelities  
    F_CSO_inverse = np.exp(-1j * arg) * F  

    #ang = 0.5 * np.arccos( np.abs(np.sum(F_CSO / F)) /N)
    #ang = 0.5 * np.arccos( np.abs(np.sum(F_CSO)) /N )
    ang = np.sqrt(M/N)
    s = int(np.pi/4 / ang)
    
if use_theta1 == False:
    ## get actual epsilon (defined as F_CSO at threshold)
    epsilon_actual = F_CSO[np.where(rho_arr == np.min(rho_arr[np.where(rho_arr**2 >= rho**2)]))][0] +1  

    ## get epsilon erfc 
    epsilon_erfc = erfc( c(k) / np.sqrt(8))

    ## get epsilon bound
    if c(k)!=0:
        epsilon_bound = np.sqrt(8/np.pi) * np.exp(-c(k)**2 /8) / c(k)
    else:
        epsilon_bound=0    

## set up coefficient arrays 
c_ideal = np.ones((N, s+1)) * np.sqrt(1/N)

if use_theta1:
    c_CSO = np.ones((N, s+1), dtype="complex") * np.sqrt(1/N)
else:
    c_CSO = np.ones((N, s+1)) * np.sqrt(1/N)    

## apply operators 
for i in np.arange(s):

    c_ideal[:,i+1] = 2* np.mean(c_ideal[:,i]*F_ideal)- c_ideal[:,i]*F_ideal 
    if i % 2 == 0 or alternate==False or use_theta1==False:
        c_CSO[:,i+1] = 2* np.mean(c_CSO[:,i]*F_CSO)- c_CSO[:,i]*F_CSO
    else:
        c_CSO[:,i+1] = 2* np.mean(c_CSO[:,i]*F_CSO_inverse)- c_CSO[:,i]*F_CSO_inverse    

## extract useful quantities:
c_CSO_norm = c_CSO / np.sqrt(np.sum(np.abs(c_CSO)**2, axis =0))

error = np.sqrt(np.sum(np.abs(c_ideal - c_CSO_norm)**2, axis =0)) 

P_failure = 1 - np.sum(np.abs(c_CSO)**2, axis =0) 

P_ideal_marked = np.empty(s+1)
P_CSO_marked = np.empty(s+1)

for i in np.arange(s+1):
    P_ideal_marked[i] = np.sum(np.abs(c_ideal[:,i] * (rho_arr**2 >= rho**2))**2)
    P_CSO_marked[i] = np.sum(np.abs(c_CSO_norm[:,i] * (rho_arr**2 >= rho**2))**2)

Pi = P_CSO_marked*(1 - P_failure)

# print relevant info 
print("----------------------------------")
print(f"M/N:\t{M/N: .3e}")
print(f"s: \t{s}")
print("----------------------------------")
print(f"P*_S:\t{P_CSO_marked[-1]:.3f}")
print(f"Pi:\t{Pi[-1]:.3f}")
print(f"P_S:\t{P_ideal_marked[-1]:.3f}")
print(f"P_T:\t{P_failure[-1]:.3f}")
print("----------------------------------")
print(f"Delta P_S:\t{P_ideal_marked[-1]-P_CSO_marked[-1]:.3f}")
print(f"Delta Pi:\t{P_ideal_marked[-1]-Pi[-1]:.3f}")
print("==================================")

######
s_arr = np.arange(s+1)

plt.figure(figsize=figsize)
plt.plot(s_arr, P_ideal_marked, label=r"$P_S$", color="grey", ls="dashed")
plt.plot(s_arr, P_CSO_marked, label=r"$P_S^*$", color="tab:red")
plt.plot(s_arr, Pi, label=r'$\Pi$', color="tab:blue")
plt.legend(fontsize=fontsize)
plt.tick_params(axis="both", labelsize=ticksize)
plt.xlabel(r"$s$",fontsize=fontsize)
plt.tight_layout()
if save:
    plt.savefig(f"probs_k{k}_v_{v}{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close()

if use_theta1:
    fig, ax1 = plt.subplots(figsize=figsize)

    colour1 = 'tab:blue'
    ax1.set_xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$', fontsize=fontsize)
    ax1.set_ylabel(r'$\vert F_i \vert$', color=colour1, fontsize=fontsize)
    ax1.scatter(rho_arr**2, F, color=colour1, marker="x")
    ax1.vlines(x=rho**2, ymin=0, ymax=1, linestyles="dashed", color="black")
    ax1.tick_params(axis='y', labelcolor=colour1, labelsize=ticksize)
    ax1.tick_params(axis='x', labelsize=ticksize)

    ax2 = ax1.twinx()  
    colour2 = 'tab:red'
    ax2.set_ylabel(r'arg($F_i$)', color=colour2, fontsize=fontsize) 
    ax2.scatter(rho_arr**2, arg, color=colour2, marker="x")
    ax2.tick_params(axis='y', labelcolor=colour2,labelsize=ticksize)
    ax2.set_yticks(np.arange(- np.pi, np.pi+np.pi/2, step=(np.pi / 2)))
    ax2.set_yticklabels([r'$-\pi$', r'$-0.5 \pi$', r'$0\pi$', r'$+0.5\pi$', r'$+\pi$'])

    fig.tight_layout()  
    if save:
        plt.savefig(f"complex_F{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close() 

if use_theta1==False:
    plt.figure(figsize=figsize)
    hist = plt.hist(rho_arr**2, color="blue")
    plt.vlines(x=rho**2- 2 * c(k)/ np.sqrt(k), ymin=0, ymax=np.max(hist[0]),linestyles="dashed", colors="black") 
    plt.vlines(x=rho**2, ymin=0, ymax=np.max(hist[0]), colors="black", label=r'$\rho_\text{thr}$')
    plt.xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$',fontsize=fontsize)
    plt.title(f"Overlap distribution (N={N}, M={M})",fontsize=titlesize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.legend(fontsize=titlesize)
    plt.tight_layout()
    if save:
        plt.savefig(f"overlap_distribution{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=figsize)
    plt.plot(rho_arr**2, F_CSO, color="red", label="CSO")
    plt.plot(rho_arr**2, F_ideal, color="blue", label="ideal")
    plt.vlines(x=rho**2- 2 * c(k)/ np.sqrt(k), ymin=-1, ymax=+1,linestyles="dashed", colors="black")
    plt.vlines(x=rho**2, ymin=-1, ymax=+1, colors="black")
    plt.xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$',fontsize=fontsize)
    plt.xlim(0, np.min(np.array([1,np.max(rho**2*1.2)])))
    plt.title(f"Fidelity distribution (signed)",fontsize=titlesize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.legend(fontsize=titlesize)
    plt.tight_layout()
    if save:
        plt.savefig(f"fidelity_distribution{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=figsize)
    hf =plt.hist([F_CSO, F_ideal], color=["red","blue"], label=["CSO", "ideal"],bins=20, rwidth=0.6, align="mid")
    plt.xlabel(r'Fidelity (signed)',fontsize=fontsize)
    plt.title(f"Fidelity distribution (N={N}, M={M})",fontsize=titlesize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.yscale("log")
    plt.legend(fontsize=titlesize)
    plt.tight_layout()
    if save:
        plt.savefig(f"fidelity_hist{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close()

plt.figure(figsize=figsize)
plt.plot(s_arr, s_arr * error[1], label=r'$s\Vert \mathcal{G} - \mathcal{G}^* \Vert_\Psi$', color="black", ls="--")
plt.plot(s_arr, error, label=r'$\Vert \mathcal{G}^s - \mathcal{G}^{*s} \Vert_\Psi$', color="red")
plt.plot(s_arr, -P_CSO_marked+P_ideal_marked, label=r'$\Delta P_S$', color="blue")
plt.plot(s_arr, -P_CSO_marked*(1 - P_failure)+P_ideal_marked, label=r'$\Delta \Pi$', color="green")
plt.legend(fontsize=fontsize)
plt.tick_params(axis="both", labelsize=ticksize)
plt.xlabel("Iteration",fontsize=fontsize)
plt.title("Error",fontsize=titlesize)
plt.tight_layout()
if save:
    plt.savefig(f"error{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close()

plt.figure(figsize=figsize)
plt.plot(s_arr, 1 - (1 - P_failure[1])**s_arr, label=r'$ 1 -(1-P_T^{(1)})^s$', color="black", ls="--")
plt.plot(s_arr, P_failure, label=r'$P_T^{(s)}$', color="red")
plt.legend(fontsize=fontsize)
plt.tick_params(axis="both", labelsize=ticksize)
plt.xlabel("Iteration",fontsize=fontsize)
plt.title("Termination probability",fontsize=titlesize)
if save:
    plt.savefig(f"termination_probability{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close()