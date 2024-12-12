import numpy as np 
import sys
import matplotlib.pyplot as plt 
from scipy.special import comb, erfc, betainc  
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'stix' 
rcParams['font.family'] = 'STIXGeneral' 

width=0.75 
fontsize=28 
titlesize=32
ticksize=22
figsize=(10,10)
show = False
save = True 
pdf = True
pdf_str=".pdf" if pdf else ""

N = 1000
rho = 0.85 # overlap, not overlap squared!
k = 100

std = 0.35  #0.06 # std of overlap distribution (assumed to be Gaussian with zero mean)

def c(k): 
    return  0.  * k**(0.4)

j_star = int(0.5 * k * (1+ rho**2) - 0.5 * c(k) * np.sqrt(k))

if j_star ==0:
    print('\nWARNING: j_star=0\n')
elif j_star==k:
    print('\nWARNING: j_star=k\n')
if j_star <0:
    print('\nNegative j_star. Aborting.\n')
    sys.exit()

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
theta = np.arcsin(np.sqrt(M/N))

## print relevant info 

print("----------------------------------")
print(f"M/N:\t{M/N: .3f}")
print(f"s: \t{s}")

## get ideal Grover fidelities 
F_ideal = np.ones(N)-2* (rho_arr**2 >= rho**2)

## get CSO fidelities
F_CSO = np.empty(N) 
for i in np.arange(N):
    p_0 = 0.5 * (1+ rho_arr[i]**2)
    F_CSO[i] = 2 * (1 - betainc(j_star,k-j_star +1, p_0) ) -1
 
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
c_CSO = np.ones((N, s+1)) * np.sqrt(1/N)

## apply operators 
for i in np.arange(s):

    c_ideal[:,i+1] = 2* np.mean(c_ideal[:,i]*F_ideal)- c_ideal[:,i]*F_ideal 
    c_CSO[:,i+1] = 2* np.mean(c_CSO[:,i]*F_CSO)- c_CSO[:,i]*F_CSO

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
print(f"P*_S:\t{P_CSO_marked[-1]:.3f}")
print(f"Pi:\t{Pi[-1]:.3f}")
print(f"P_S:\t{P_ideal_marked[-1]:.3f}")
print("==================================")

######
s_arr = np.arange(s+1)

plt.figure(figsize=figsize)
hist = plt.hist(rho_arr**2, color="blue")
plt.vlines(x=rho**2- 2 * c(k)/ np.sqrt(k), ymin=0, ymax=np.max(hist[0]),linestyles="dashed", colors="black", label=r'$\rho_\text{thr} - 2\frac{c(k)}{\sqrt{k}}$')
plt.vlines(x=rho**2, ymin=0, ymax=np.max(hist[0]), colors="black", label=r'$\rho_\text{thr}$')
plt.xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$',fontsize=fontsize)
plt.title(f"Overlap distribution (N={N}, M={M}, D={D})",fontsize=titlesize)
plt.tick_params(axis="both", labelsize=ticksize)
#plt.legend(fontsize=titlesize)
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
plt.hlines(y=-1+epsilon_erfc, xmin=0, xmax=1, linestyles="dashed", colors="gray")
plt.hlines(y=1-epsilon_erfc, xmin=0, xmax=1, linestyles="dashed", colors="gray")
plt.xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$',fontsize=fontsize)
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
hf =plt.hist([F_CSO, F_ideal], color=["red","blue"], label=["CSO", "ideal"], histtype="barstacked", bins=20, rwidth=0.6, align="mid")
plt.vlines(x=-1+epsilon_erfc, ymin=0, ymax=np.max(hf[0]), linestyles="dashed", colors="gray")
plt.vlines(x=1-epsilon_erfc, ymin=0, ymax=np.max(hf[0]), linestyles="dashed", colors="gray")
plt.xlabel(r'Fidelity (signed)',fontsize=fontsize)
plt.title(f"Fidelity distribution (N={N}, M={M}, D={D})",fontsize=titlesize)
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
plt.plot(s_arr, P_ideal_marked, label=r"$P_S$", color="blue")
plt.plot(s_arr, P_CSO_marked, label=r"$P_S^*$", color="red")
plt.plot(s_arr, Pi, label=r'$\Pi$', color="green")
plt.legend(fontsize=fontsize)
plt.tick_params(axis="both", labelsize=ticksize)
plt.xlabel("Iteration",fontsize=fontsize)
plt.title("Success probability",fontsize=titlesize)
plt.tight_layout()
if save:
    plt.savefig(f"success_probability{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close()

plt.figure(figsize=figsize)
plt.plot(s_arr, s_arr * error[1], label=r'$s\Vert \mathcal{G} - \mathcal{G}^* \Vert_\Psi$', color="black", ls="--")
#plt.plot(s_arr, s_arr * 2 * np.sqrt(delta + epsilon_erfc), label=r'$s 2 \sqrt{\epsilon + \Delta}  $', color="gray", ls="--")
plt.plot(s_arr, error, label=r'$\Vert \mathcal{G}^s - \mathcal{G}^{*s} \Vert_\Psi$', color="red")
plt.plot(s_arr, -P_CSO_marked+P_ideal_marked, label=r'$P_S - P_S^* $', color="blue")
plt.plot(s_arr, -P_CSO_marked*(1 - P_failure)+P_ideal_marked, label=r'$P_S - \Pi $', color="green")
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
plt.plot(s_arr, 1 - (1 - P_failure[1])**s_arr, label=r'$ (P_T^{(1)})^s$', color="black", ls="--")
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