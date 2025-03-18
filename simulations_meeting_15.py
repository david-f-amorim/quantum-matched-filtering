import numpy as np 
import matplotlib.pyplot as plt 
from scipy.special import erf 
from matplotlib import rcParams

k = 10000 
v_min = 3/2
v_max = 7/2
max_overlap = 0.01
rho = max_overlap /2

overlaps = np.linspace(0,max_overlap,1000,dtype=complex)

if v_min+v_max != 0:
    gamma = np.pi / (v_min + v_max)
else:
    gamma = -1    

### write F = A + B + C * (D - E)
A = 0.5 * erf( np.sqrt(k/8) * ( rho - overlaps - v_min / np.sqrt(k) ) )
B = 0.5 * erf( np.sqrt(k/8) * ( rho - overlaps + v_max / np.sqrt(k) ) )
F = A +B

if gamma != -1:
    C = 0.5 * np.exp( 1j * ( np.sqrt(k) /2 * (1+ overlaps) + v_min ) - gamma**2 / 4)
    D = erf( np.sqrt(k/8) * ( rho - overlaps + v_max / np.sqrt(k) - 1j * gamma / np.sqrt(k) ) )
    E = erf( np.sqrt(k/8) * ( rho - overlaps - v_min / np.sqrt(k) - 1j * gamma / np.sqrt(k) ) )
    F +=  C * (D - E)

abs = np.abs(F)
arg = np.angle(F) + 2* np.pi * (np.angle(F) < -0.1)

AB_abs = np.abs(A +B)
AB_arg = np.angle(A +B) + 2* np.pi * (np.angle(A +B) < -0.1)

if gamma != -1:
    CDE_abs = np.abs( C * (D - E))
    CDE_arg = np.angle( C * (D - E)) + 2* np.pi * (np.angle( C * (D - E)) < -0.1)

    DE_abs = np.abs( D - E)
    DE_arg = np.angle( D - E) + 2* np.pi * (np.angle( D - E) < -0.1)

    C_abs = np.abs( C )
    C_arg = np.angle( C ) + 2* np.pi * (np.angle( C ) < -0.1)

#----- PLOTTING -----
save = True 
pdf = True 
show = True 

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
ax1.plot(np.abs(overlaps), abs, color=colour1)
ax1.vlines(x=rho, ymin=0, ymax=1, linestyles="dashed", color="black")
ax1.tick_params(axis='y', labelcolor=colour1, labelsize=ticksize)
ax1.tick_params(axis='x', labelsize=ticksize)

ax2 = ax1.twinx()  
colour2 = 'tab:red'
ax2.set_ylabel(r'arg($F_i$)', color=colour2, fontsize=fontsize) 
ax2.plot(np.abs(overlaps), arg, color=colour2)
ax2.tick_params(axis='y', labelcolor=colour2,labelsize=ticksize)
ax2.set_yticks(np.arange(0, 2*np.pi+np.pi/2, step=(np.pi / 2)))
ax2.set_yticklabels([r'$0\pi$', r'$0.5 \pi$', r'$\pi$', r'$1.5\pi$', r'$2\pi$'])

fig.tight_layout()  
if save:
    plt.savefig(f"complex_F{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close() 

fig, ax1 = plt.subplots(figsize=figsize)
colour1 = 'tab:blue'
ax1.set_xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$', fontsize=fontsize)
ax1.set_ylabel(r'$\vert A+B \vert$', color=colour1, fontsize=fontsize)
ax1.plot(np.abs(overlaps), AB_abs, color=colour1)
ax1.vlines(x=rho, ymin=0, ymax=1, linestyles="dashed", color="black")
ax1.tick_params(axis='y', labelcolor=colour1, labelsize=ticksize)
ax1.tick_params(axis='x', labelsize=ticksize)

ax2 = ax1.twinx()  
colour2 = 'tab:red'
ax2.set_ylabel(r'arg($A+B$)', color=colour2, fontsize=fontsize) 
ax2.plot(np.abs(overlaps), AB_arg, color=colour2)
ax2.tick_params(axis='y', labelcolor=colour2,labelsize=ticksize)
ax2.set_yticks(np.arange(0, 2*np.pi+np.pi/2, step=(np.pi / 2)))
ax2.set_yticklabels([r'$0\pi$', r'$0.5 \pi$', r'$\pi$', r'$1.5\pi$', r'$2\pi$'])

fig.tight_layout()  
if save:
    plt.savefig(f"complex_AB{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close() 

if gamma != -1:
    fig, ax1 = plt.subplots(figsize=figsize)
    colour1 = 'tab:blue'
    ax1.set_xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$', fontsize=fontsize)
    ax1.set_ylabel(r'$\vert C(D-E) \vert$', color=colour1, fontsize=fontsize)
    ax1.plot(np.abs(overlaps), CDE_abs, color=colour1)
    ax1.vlines(x=rho, ymin=0, ymax=np.max(CDE_abs), linestyles="dashed", color="black")
    ax1.tick_params(axis='y', labelcolor=colour1, labelsize=ticksize)
    ax1.tick_params(axis='x', labelsize=ticksize)

    ax2 = ax1.twinx()  
    colour2 = 'tab:red'
    ax2.set_ylabel(r'arg($C(D-E)$)', color=colour2, fontsize=fontsize) 
    ax2.plot(np.abs(overlaps), CDE_arg, color=colour2)
    ax2.tick_params(axis='y', labelcolor=colour2,labelsize=ticksize)
    ax2.set_yticks(np.arange(0, 2*np.pi+np.pi/2, step=(np.pi / 2)))
    ax2.set_yticklabels([r'$0\pi$', r'$0.5 \pi$', r'$\pi$', r'$1.5\pi$', r'$2\pi$'])

    fig.tight_layout()  
    if save:
        plt.savefig(f"complex_CDE{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close() 

    fig, ax1 = plt.subplots(figsize=figsize)
    colour1 = 'tab:blue'
    ax1.set_xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$', fontsize=fontsize)
    ax1.set_ylabel(r'$\vert D-E \vert$', color=colour1, fontsize=fontsize)
    ax1.plot(np.abs(overlaps), DE_abs, color=colour1)
    ax1.vlines(x=rho, ymin=0, ymax=np.max(DE_abs), linestyles="dashed", color="black")
    ax1.tick_params(axis='y', labelcolor=colour1, labelsize=ticksize)
    ax1.tick_params(axis='x', labelsize=ticksize)

    ax2 = ax1.twinx()  
    colour2 = 'tab:red'
    ax2.set_ylabel(r'arg($D-E$)', color=colour2, fontsize=fontsize) 
    ax2.plot(np.abs(overlaps), DE_arg, color=colour2)
    ax2.tick_params(axis='y', labelcolor=colour2,labelsize=ticksize)
    ax2.set_yticks(np.arange(0, 2*np.pi+np.pi/2, step=(np.pi / 2)))
    ax2.set_yticklabels([r'$0\pi$', r'$0.5 \pi$', r'$\pi$', r'$1.5\pi$', r'$2\pi$'])

    fig.tight_layout()  
    if save:
        plt.savefig(f"complex_DE{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close() 

    fig, ax1 = plt.subplots(figsize=figsize)
    colour1 = 'tab:blue'
    ax1.set_xlabel(r'$\vert \langle \psi | \phi_i \rangle \vert^2$', fontsize=fontsize)
    ax1.set_ylabel(r'$\vert C \vert$', color=colour1, fontsize=fontsize)
    ax1.plot(np.abs(overlaps), C_abs, color=colour1)
    ax1.vlines(x=rho, ymin=0, ymax=1, linestyles="dashed", color="black")
    ax1.tick_params(axis='y', labelcolor=colour1, labelsize=ticksize)
    ax1.tick_params(axis='x', labelsize=ticksize)

    ax2 = ax1.twinx()  
    colour2 = 'tab:red'
    ax2.set_ylabel(r'arg($C$)', color=colour2, fontsize=fontsize) 
    ax2.plot(np.abs(overlaps), C_arg, color=colour2)
    ax2.tick_params(axis='y', labelcolor=colour2,labelsize=ticksize)
    ax2.set_yticks(np.arange(0, 2*np.pi+np.pi/2, step=(np.pi / 2)))
    ax2.set_yticklabels([r'$0\pi$', r'$0.5 \pi$', r'$\pi$', r'$1.5\pi$', r'$2\pi$'])

    fig.tight_layout()  
    if save:
        plt.savefig(f"complex_C{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close() 

