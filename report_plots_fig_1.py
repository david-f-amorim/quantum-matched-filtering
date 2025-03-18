import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rcParams, patches

save = True 
pdf = True 
show= True 

rcParams['mathtext.fontset'] = 'stix' 
rcParams['font.family'] = 'STIXGeneral' 
rcParams["text.usetex"] = True

width=0.75 
fontsize=28 
titlesize=32
ticksize=22
figsize=(10,6)
pdf_str=".pdf" if pdf else ""

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

ticklabelpad_x = rcParams['xtick.major.pad']
ticklabelpad_y = rcParams['ytick.major.pad']

#### INITIAL #####

x = np.linspace(0,0.75,100)
plt.xlim(0,1)
plt.ylim(-0.5,1)
plt.plot(x,0.3*x, color="tab:blue")

fig = plt.gcf()
fig.set_facecolor('white') 
ax = plt.gca()

ax.add_patch(patches.Arc((0,0), 0.8,0.8,angle= 0, theta1=0, theta2=17.44, color="tab:blue"))

plt.annotate(r'$\theta$', xy=(0.35,0.05), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:blue")

plt.annotate(r'$\vert \Psi \rangle$', xy=(0.8,0.8*0.3), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:blue")

plt.annotate(r'$\vert u \rangle$', xy=(1.05,0.35), xytext=(0, -ticklabelpad_x), ha='center', va='center',
            xycoords='axes fraction', textcoords='offset points', fontsize="xx-large")
plt.annotate(r'$\vert m \rangle$', xy=(-0.1,1), xytext=(0, -ticklabelpad_y), ha='center', va='center',
            xycoords='axes fraction', textcoords='offset points', fontsize="xx-large")

arrowed_spines(fig, ax)

plt.tight_layout()
if save:
    plt.savefig(f"Grover_initial{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close() 

#### ORACLE #####

x = np.linspace(0,0.75,100)
plt.xlim(0,1)
plt.ylim(-0.5,1)
plt.plot(x,0.3*x, color="tab:blue", ls="--")
plt.plot(x,-0.3*x, color="tab:red")

fig = plt.gcf()
fig.set_facecolor('white') 
ax = plt.gca()

ax.add_patch(patches.Arc((0,0), 0.8,0.8,angle= 0, theta1=0, theta2=17.44, color="tab:blue"))

plt.annotate(r'$\theta$', xy=(0.35,0.05), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:blue")

plt.annotate(r'$\vert \Psi \rangle$', xy=(0.83,0.8*0.3), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:blue")

ax.add_patch(patches.Arc((0,0), 0.8,0.8,angle= 0, theta1=-17.44, theta2=0, color="tab:red"))

plt.annotate(r'$\theta$', xy=(0.35,-0.05), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:red")

plt.annotate(r'$\mathcal{U}_O \vert \Psi \rangle$', xy=(0.83,-0.8*0.3), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:red")

plt.annotate(r'$\vert u \rangle$', xy=(1.05,0.35), xytext=(0, -ticklabelpad_x), ha='center', va='center',
            xycoords='axes fraction', textcoords='offset points', fontsize="xx-large")
plt.annotate(r'$\vert m \rangle$', xy=(-0.1,1), xytext=(0, -ticklabelpad_y), ha='center', va='center',
            xycoords='axes fraction', textcoords='offset points', fontsize="xx-large")

arrowed_spines(fig, ax)

plt.tight_layout()
if save:
    plt.savefig(f"Grover_oracle{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close() 

#### FINAL #####

x = np.linspace(0,0.75,100)
plt.xlim(0,1)
plt.ylim(-0.35,1)
plt.plot(x,0.3*x, color="tab:blue", ls="--")
plt.plot(x,-0.3*x, color="tab:red",ls="--")
plt.plot(x,0.9*x, color="limegreen")

fig = plt.gcf()
fig.set_facecolor('white') 
fig.set_figheight(4)
fig.set_figwidth(5)
ax = plt.gca()

ax.add_patch(patches.Arc((0,0), 0.8,0.8,angle= 0, theta1=0, theta2=17.44, color="tab:blue"))

plt.annotate(r'$\theta$', xy=(0.35,0.05), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:blue")

plt.annotate(r'$\vert \Psi \rangle$', xy=(0.83,0.8*0.3), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:blue")

ax.add_patch(patches.Arc((0,0), 1,1,angle= 0, theta1=17.44, theta2=2.45*17.44, color="limegreen"))

plt.annotate(r'$2\theta$', xy=(0.35,0.2), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="limegreen")

plt.annotate(r'$ \mathcal{G} \vert \Psi \rangle$', xy=(0.83,0.8*0.9), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="limegreen")

ax.add_patch(patches.Arc((0,0), 0.8,0.8,angle= 0, theta1=-17.44, theta2=0, color="tab:red"))

plt.annotate(r'$\theta$', xy=(0.35,-0.05), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:red")

plt.annotate(r'$\mathcal{U}_O \vert \Psi \rangle$', xy=(0.83,-0.8*0.3), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:red")

plt.annotate(r'$\vert u \rangle$', xy=(1.05,0.35), xytext=(0, -ticklabelpad_x), ha='center', va='center',
            xycoords='axes fraction', textcoords='offset points', fontsize="xx-large")
plt.annotate(r'$\vert m \rangle$', xy=(-0.1,1), xytext=(0, -ticklabelpad_y), ha='center', va='center',
            xycoords='axes fraction', textcoords='offset points', fontsize="xx-large")

arrowed_spines(fig, ax)

plt.tight_layout()
if save:
    plt.savefig(f"Grover_final{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close() 


#### SECOND #####

x = np.linspace(0,0.75,100)
x2 = np.linspace(0,1/(1.5)-0.1,100)
plt.xlim(0,1)
plt.ylim(-0.5,1)
plt.plot(x,0.3*x, color="tab:blue", ls="--")
plt.plot(x,-0.3*x, color="tab:red", ls="--")
plt.plot(x,0.9*x, color="limegreen", ls="--")
plt.plot(x2,1.5*x2, color="tab:purple")

fig = plt.gcf()
fig.set_facecolor('white') 
ax = plt.gca()

ax.add_patch(patches.Arc((0,0), 0.8,0.8,angle= 0, theta1=0, theta2=17.44, color="tab:blue"))

plt.annotate(r'$\theta$', xy=(0.35,0.05), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:blue")

plt.annotate(r'$\vert \Psi \rangle$', xy=(0.83,0.8*0.3), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:blue")

ax.add_patch(patches.Arc((0,0), 1.2,1.2,angle= 0, theta1=2.445*17.44, theta2=3.245*17.44, color="tab:purple"))

plt.annotate(r'$2\theta$', xy=(0.35,0.42), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:purple")

plt.annotate(r'$ \mathcal{G}^2 \vert \Psi \rangle$', xy=(0.65,0.62*1.5), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:purple")


ax.add_patch(patches.Arc((0,0), 1,1,angle= 0, theta1=17.44, theta2=2.45*17.44, color="limegreen"))

plt.annotate(r'$2\theta$', xy=(0.35,0.2), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="limegreen")

plt.annotate(r'$ \mathcal{G} \vert \Psi \rangle$', xy=(0.83,0.8*0.9), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="limegreen")

ax.add_patch(patches.Arc((0,0), 0.8,0.8,angle= 0, theta1=-17.44, theta2=0, color="tab:red"))

plt.annotate(r'$\theta$', xy=(0.35,-0.05), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:red")

plt.annotate(r'$\mathcal{U}_O \vert \Psi \rangle$', xy=(0.83,-0.8*0.3), xytext=(0,0), ha='center', va='center',
            xycoords='data', textcoords='offset points', fontsize="xx-large", color="tab:red")

plt.annotate(r'$\vert u \rangle$', xy=(1.05,0.35), xytext=(0, -ticklabelpad_x), ha='center', va='center',
            xycoords='axes fraction', textcoords='offset points', fontsize="xx-large")
plt.annotate(r'$\vert m \rangle$', xy=(-0.1,1), xytext=(0, -ticklabelpad_y), ha='center', va='center',
            xycoords='axes fraction', textcoords='offset points', fontsize="xx-large")

arrowed_spines(fig, ax)

plt.tight_layout()
if save:
    plt.savefig(f"Grover_second{pdf_str}", bbox_inches='tight', dpi=500)
if show:
    plt.show()
plt.close() 