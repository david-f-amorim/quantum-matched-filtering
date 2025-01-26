import numpy as np 
from scipy.special import betainc
from scipy.stats import binom, norm  
import matplotlib.pyplot as plt 
from matplotlib import rcParams


#----- CONTROLS -----

show =True
save = True 
pdf = True

k_plot = False
ck_plot = False
rho_plot = False

CSC_plot = False 
CSC_phase_plot =False

error_plot = False 
gauss_plot = True 

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
    plt.vlines(x=rho, ymin=min, ymax=max, colors="black", label=r'$\rho_{\mathrm{thr}}$')
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
        plt.plot(overlaps, fidelities_rho[i], color=colours[i], label=r'$\rho_{\mathrm{thr}}=$'f'{labels[i]}', ls=styles[i])
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
    plt.xlabel(r'$\vert j \rangle \vert \Omega_j \rangle$',fontsize=1.3*fontsize)
    plt.ylabel(r'Amplitude',fontsize=1.3*fontsize)
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
    plt.xlabel(r'$\vert j \rangle \vert \Omega_j \rangle$',fontsize=1.3*fontsize)
    plt.ylabel(r'Amplitude',fontsize=1.3*fontsize)
    arrowed_spines(fig, ax)

    plt.tight_layout()
    if save:
        plt.savefig(f"CSC_plot_after{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close() 

if CSC_phase_plot:

    def arrowed_spines2(fig, ax,ax2):

        xmin, xmax = ax.get_xlim() 
        ymin, ymax = ax.get_ylim()
        ymin2, ymax2 = ax2.get_ylim()

        ymin = min(ymin, ymin2)
        ymax=max(ymax,ymax2)

        # removing the default axis on all sides:
        for side in ['bottom','right','top','left']:
            ax.spines[side].set_visible(False)
            ax2.spines[side].set_visible(False)

        # removing the axis ticks
        plt.xticks([]) # labels 
        plt.yticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none') # tick markers
        ax.yaxis.set_ticks_position('none')
        ax2.xaxis.set_ticks_position('none') # tick markers
        ax2.yaxis.set_ticks_position('none')

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
        
        ax2.arrow(xmax - yhw-ohg*0.9, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
                head_width=yhw, head_length=yhl, overhang = ohg, 
                length_includes_head= True, clip_on = False)


    #####
    k = 30 # 300
    j_star = k / 2
    rho = 0.2

    p = 0.5*(1+rho)
    x = np.arange(0, k, step=1) # step =3
    y = binom.pmf(x,k,p)
    
    plt.figure(figsize=figsize)
    plt.scatter(x , np.sqrt(y),color="tab:blue")
    plt.vlines(x, 0, np.sqrt(y),colors="tab:blue")
    
    plt.xlim(0,k+1)
    
    fig = plt.gcf()
    fig.set_facecolor('white') 
    ax = plt.gca()

    plt.xlabel(r'$\vert j \rangle \vert \Omega_j \rangle$',fontsize=1.3*fontsize)
    plt.ylabel(r'Amplitude',fontsize=1.3*fontsize,color='tab:blue')
   
    ax2 = ax.twinx()  
    ax2.set_ylabel(r'Phase', color='tab:red', fontsize=1.3*fontsize,labelpad=-7) 
    x2= np.linspace(0,k,1000)

    ## standard 
    #ax2.plot(x2,0.05*np.ones(len(x2))+ 0.4 * (x2 >= j_star), color='tab:red')
    ## linear gradient 
    q = 0.1
    ax2.plot(x2,0.05*np.ones(len(x2))+ 0.4 * (x2 -j_star + q*k) / (2*q*k) * ( (x2 >= j_star- q*k) * (x2 < j_star+ q*k))  + 0.4 * (x2 >= j_star+ q*k), color='tab:red') 
    
    arrowed_spines2(fig,ax,ax2)

    plt.tight_layout()
    if save:
        plt.savefig(f"CSC_plot_phase{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close() 

if error_plot:

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
    k = 20 # 300
    j_star = 9
    rho = 0.3

    p = 0.5*(1+rho)
    x = np.linspace(0, k, 1000)
    y = norm.pdf(x,loc=p*k,scale=np.sqrt(k*p*(1-p)))
    
  
    plt.figure(figsize=figsize)
    plt.plot(x , y,color="black")
    plt.fill_between(x, y, where= (0 < x)&(x < j_star), color= "tab:blue",alpha=0.2)
    plt.vlines(x=j_star, ymin=0, ymax=1.2*np.max(y), color="tab:blue")

    plt.vlines(x=p*k-np.sqrt(k*p*(1-p)), ymin=0, ymax=1.2*np.max(y), color="tab:red")
    plt.fill_between(x, y, where= (p*k-np.sqrt(k*p*(1-p)) < x)&(x < p*k+np.sqrt(k*p*(1-p))), color= "tab:red",alpha=0.2)
    plt.vlines(x=p*k+np.sqrt(k*p*(1-p)), ymin=0, ymax=1.2*np.max(y), color="tab:red")
    
    plt.annotate(r'$w$', xy=(p*k,1.2*np.max(y)), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', size=1.3*fontsize, color="tab:red")
    
    plt.arrow(x=p*k-0.8*np.sqrt(k*p*(1-p)), y = 1.1*np.max(y), dx = 2*0.8*np.sqrt(k*p*(1-p)), dy =0, color="tab:red", lw=0.25, length_includes_head=True,head_width=0.01, head_length=0.2, overhang = 0.3) 
    plt.arrow(x=p*k+0.8*np.sqrt(k*p*(1-p)), y = 1.1*np.max(y), dx = -2*0.8*np.sqrt(k*p*(1-p)), dy =0, color="tab:red", lw=0.25, length_includes_head=True,head_width=0.01, head_length=0.2, overhang = 0.3) 
    

    plt.arrow(x=0.05*j_star, y = 0.4*np.max(y), dx = 0.9*j_star, dy =0, color="tab:blue", lw=0.25, length_includes_head=True,head_width=0.01, head_length=0.2, overhang = 0.3) 
    plt.arrow(x=0.95*j_star, y = 0.4*np.max(y), dx = -0.9*j_star, dy =0, color="tab:blue", lw=0.25, length_includes_head=True,head_width=0.01, head_length=0.2, overhang = 0.3) 

    plt.annotate(r'$j^*$', xy=(j_star/2,0.5*np.max(y)), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', size=1.3*fontsize, color="tab:blue")

    plt.xlim(0,k+1)
    fig = plt.gcf()
    fig.set_facecolor('white') 
    ax = plt.gca()
    plt.xlabel(r'$j$',fontsize=1.3*fontsize)
    plt.ylabel(r'Amplitude',fontsize=1.3*fontsize)
    arrowed_spines(fig, ax)

    plt.tight_layout()
    if save:
        plt.savefig(f"error_plot{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close() 

if gauss_plot:

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
    k = 20 # 300
    rho = 0.01 # 0.4 
    p = 0.5*(1+0.05) # 0.4, 0.1, 0.7

    j_star = k*0.5*(1+rho)

    x = np.linspace(0, k, 1000)
    y = norm.pdf(x,loc=p*k,scale=np.sqrt(k*p*(1-p)))
    

    plt.figure(figsize=figsize)
    plt.plot(x , y,color="black")
    plt.fill_between(x, y, where= (0 < x)&(x < j_star), color= "tab:blue",alpha=0.2)
    plt.vlines(x=j_star, ymin=0, ymax=1.3*np.max(y), color="tab:blue")

    plt.arrow(x=0.05*j_star, y = 1.1*np.max(y), dx = 0.9*j_star, dy =0, color="tab:blue", lw=0.25, length_includes_head=True,head_width=0.01*k/20, head_length=0.2*k/20, overhang = 0.3*k/20) 
    plt.arrow(x=0.95*j_star, y = 1.1*np.max(y), dx = -0.9*j_star, dy =0, color="tab:blue", lw=0.25, length_includes_head=True,head_width=0.01*k/20, head_length=0.2*k/20, overhang = 0.3*k/20) 

    plt.annotate(r'$j^*$', xy=(j_star/2,1.2*np.max(y)), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', size=1.3*fontsize, color="tab:blue")

    plt.xlim(0,k+1)
    fig = plt.gcf()
    fig.set_facecolor('white') 
    ax = plt.gca()
    plt.xlabel(r'$j$',fontsize=1.3*fontsize)
    plt.ylabel(r'Amplitude',fontsize=1.3*fontsize)
    arrowed_spines(fig, ax)

    plt.tight_layout()
    if save:
        plt.savefig(f"gauss_plot{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    plt.close()