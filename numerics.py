import numpy as np 
import sys
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.special import erfc, betainc  
from matplotlib import rcParams

"""
This code is used to produce numerical simulations of quantum matched filtering. 




"""

# --- CONFIGURE OUTPUT PLOTS ---  
show =True              # toggle to show/not show plots
save = True             # toggle to save/not save plots
pdf = True              # toggle to save/not save plots as PDFs (if not PDF, output is PNG)

pdf_str=".pdf" if pdf else ""
rcParams['mathtext.fontset'] = 'stix' 
rcParams['font.family'] = 'STIXGeneral' 
fontsize=28 * 1.3
titlesize=32
ticksize=22 * 1.3
figsize=(10,6)   

# --- CONFIGURE SIMULATION ---

N = 10000               # number of templates
rho = 0.5               # threshold overlap
k = 1000                # number of copies in the k-copy SWAP test 
use_complex = True      # use (or not) CSC with complex phases 


## set CSC threshold value j_star (irrelevant if use_complex==True)

ck = 0                  # offset function c(k) used to define CSC threshold j_star 
                        # (note: has been replaced with q(k) in the report)

j_star = int(0.5 * k * (1+ rho) - 0.5 * ck * np.sqrt(k))

if j_star <0:
    print('\nNegative j_star. Aborting.\n') # j_star must be non-negative
    sys.exit()

## configure CSC phase gradient (irrelevant if use_complex==False)

v_min = 3/2 # see j_min below
v_max = 5/2 # see j_max below

def theta(j): # CSC phase assigned to a given value of j
    j_min =  int(0.5 * k * (1+ rho)- v_min * np.sqrt(k)) # j value at which phase gradient sets in 
    j_max =  int(0.5 * k * (1+ rho)+ v_max * np.sqrt(k)) # j value at which phase gradient is turned off

    return np.pi * (j >= j_max) + 0 * (j < j_min) + np.pi * ( j - j_min)/(np.sqrt(k)*(v_max + v_min)) * ( j < j_max) * (j >= j_min)   

## print simulation information
print("==================================")
print(f"k:\t{k}")


# --- SAMPLE OVERLAP DISTRIBUTION ---
# 
# NOTE: change this code when generalising to different overlap distributions!

std = 0.3 * rho   # std of overlap distribution (assumed to be Gaussian with zero mean)
rng = np.random.default_rng(seed = 12345678)
rho_arr = np.abs(rng.normal(size=N, scale=std)) # remove negative overlaps
rho_arr = np.sort(rho_arr) # sorted array of random overlaps

for i in np.arange(N):
    if rho_arr[i]>1:
        rho_arr[i]=1 # make sure all overlaps are bounded

M = np.sum(rho_arr >= rho) # count number of marked entries 
if M == 0:
    print("\nNo matches. Aborting.\n")
    sys.exit() 

s = int(np.pi /4 * np.sqrt(N/M)) # get number of iterations (assuming same number of iterations as in ideal Grover)   

p_0_arr = 0.5 * (1+ rho_arr) # array of SWAP test probabilities associated with overlaps


# --- FIND ORACLE FIDELITIES ---

F_ideal = np.ones(N)-2* (rho_arr >= rho) # get ideal fidelities (+1 for unmarked items, -1 for marked)

if use_complex == False:
    F_CSO = 2 * (1 - betainc(j_star,k-j_star +1, p_0_arr) ) -1 # uses CDF of the binomial distribution to 
                                                               # efficiently calculate CSO fidelities in the 
                                                               # absense of complex phases
else:
    # when using the complex CSC, fidelities have to be obtained 
    # by explicitly summing over the different binomial components
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

    # NOTE: if experimenting with a different number of iterations (s),
    # change the code below
    ang = np.sqrt(M/N)
    s = int(np.pi/4 / ang)


# --- SIMULATE GROVER'S ALGORITHM ---

c_ideal = np.ones((N, s+1)) * np.sqrt(1/N) # coefficients of initial superposition (ideal Grover)

if use_complex:
    c_CSO = np.ones((N, s+1), dtype="complex") * np.sqrt(1/N)  # coefficients of initial superposition (CSO)
else:
    c_CSO = np.ones((N, s+1)) * np.sqrt(1/N)     # coefficients of initial superposition (CSO)

# repeatedly apply Grover's operator
for i in np.arange(s):

    c_ideal[:,i+1] = 2* np.mean(c_ideal[:,i]*F_ideal)- c_ideal[:,i]*F_ideal 

    if i % 2 == 0 or use_complex==False:
        c_CSO[:,i+1] = 2* np.mean(c_CSO[:,i]*F_CSO)- c_CSO[:,i]*F_CSO
    else:
        # for complex CSO, alternate between oracle and its conjugate
        c_CSO[:,i+1] = 2* np.mean(c_CSO[:,i]*F_CSO_inverse)- c_CSO[:,i]*F_CSO_inverse    

# extract useful quantities:

c_CSO_norm = c_CSO / np.sqrt(np.sum(np.abs(c_CSO)**2, axis =0)) # normalise CSO coefficients 
P_failure = 1 - np.sum(np.abs(c_CSO)**2, axis =0) # probability of terminating early 

P_ideal_marked = np.empty(s+1)
P_CSO_marked = np.empty(s+1)

for i in np.arange(s+1):
    P_ideal_marked[i] = np.sum(np.abs(c_ideal[:,i] * (rho_arr**2 >= rho**2))**2) # probability of finding a marked item (ideal Grover)
    P_CSO_marked[i] = np.sum(np.abs(c_CSO_norm[:,i] * (rho_arr**2 >= rho**2))**2)# probability of finding a marked item (CSO, not counting probability of early termination)

Pi = P_CSO_marked*(1 - P_failure) # weighted success probability: probability of not terminating early AND finding a marked item

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


# --- PLOTTING ---
s_arr = np.arange(s+1)

# plot of ideal success probability, CSO success probability, and weighted CSO success probability
# (as a function of iteration)
plt.figure(figsize=figsize)
plt.plot(s_arr, P_ideal_marked, label=r"$P_S$", color="grey", ls="dashed")
plt.plot(s_arr, P_CSO_marked, label=r"$P_S^*$", color="tab:red")
plt.plot(s_arr, Pi, label=r'$\Pi$', color="tab:blue")
plt.legend(fontsize=fontsize)
plt.tick_params(axis="both", labelsize=ticksize)
plt.xlabel(r"$s$",fontsize=fontsize)
plt.tight_layout()
if save:
    plt.savefig(f"probabilities{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close()

if use_complex:

    # plot of CSO fidelity (showing magnitude and phase)
    fig, ax1 = plt.subplots(figsize=figsize)

    colour1 = 'tab:blue'
    ax1.set_xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$', fontsize=fontsize)
    ax1.set_ylabel(r'$\vert F_i \vert$', color=colour1, fontsize=fontsize)
    ax1.scatter(rho_arr, F, color=colour1, marker="x")
    ax1.vlines(x=rho, ymin=0, ymax=1, linestyles="dashed", color="black")
    ax1.tick_params(axis='y', labelcolor=colour1, labelsize=ticksize)
    ax1.tick_params(axis='x', labelsize=ticksize)

    ax2 = ax1.twinx()  
    colour2 = 'tab:red'
    ax2.set_ylabel(r'arg($F_i$)', color=colour2, fontsize=fontsize) 
    ax2.scatter(rho_arr, arg, color=colour2, marker="x")
    ax2.tick_params(axis='y', labelcolor=colour2,labelsize=ticksize)
    ax2.set_yticks(np.arange(- np.pi, np.pi+np.pi/2, step=(np.pi / 2)))
    ax2.set_yticklabels([r'$-\pi$', r'$-0.5 \pi$', r'$0\pi$', r'$+0.5\pi$', r'$+\pi$'])

    fig.tight_layout()  
    if save:
        plt.savefig(f"complex_F{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close() 


# plot of overlap distribution
plt.figure(figsize=figsize)
hist = plt.hist(rho_arr, color="blue")
plt.vlines(x=rho, ymin=0, ymax=np.max(hist[0]), colors="black", label=r'$\rho_\text{thr}$')
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

if use_complex==False:

    # plot of fidelity distribution
    plt.figure(figsize=figsize)
    plt.plot(rho_arr, F_CSO, color="red", label="CSO")
    plt.plot(rho_arr, F_ideal, color="blue", label="ideal")
    plt.vlines(x=rho, ymin=-1, ymax=+1, colors="black")
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

    # hisogram of fidelity distribution
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

# plot of termination probability (as a function of iteration)
plt.figure(figsize=figsize)
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