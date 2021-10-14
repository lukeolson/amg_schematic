import sys
import numpy as np
import matplotlib.pyplot as plt
import pyamg
import scipy.spatial
import scipy.sparse
plt.rcParams['image.cmap']='plasma_r'

savefig = False
if len(sys.argv) > 1:
    if sys.argv[1] == '--savefig':
        savefig = True

def showit(figname='tmp.pdf'):
    if savefig:
        plt.savefig(figname, bbox_inches='tight')
    else:
        plt.show()

data = pyamg.gallery.load_example('unit_square')

A = data['A'].tocsr()
V = 0.5 * (1.0 + (data['vertices'] /  (np.pi/2)))
E = data['elements']
n = A.shape[0]
b = np.zeros(n)

# set up a random guess (error)
np.random.rand(4848)
e0 = np.sin(np.pi*V[:,0])*np.sin(np.pi*V[:,1]) + 0.7 * np.abs(np.random.rand(n))
ml = pyamg.ruge_stuben_solver(A, max_levels=2, max_coarse=10, keep=True)

Ge = e0.copy()                  # relaxed error
pyamg.relaxation.relaxation.jacobi(A, Ge, b, iterations=2, omega=4/5)
r = b - A @ Ge                  # residual
rc = ml.levels[0].P.T @ r       # restricted residual
ec = scipy.sparse.linalg.spsolve(ml.levels[1].A.tocsr(), rc) # coarse error
e1 = Ge + ml.levels[0].P @ ec   # new, corrected solution (error)

oneplot = False

if oneplot:
    if savefig:
        figsize = (16,32)
    else:
        figsize = (6,12)
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=figsize)
else:
    figsize = (8,8)

if oneplot:
    ax = axs[0,0]
else:
    fig, ax = plt.subplots(figsize=figsize)
tc = ax.tripcolor(V[:,0], V[:,1], e0, vmin=0, vmax=1.5, edgecolor='w')
ax.axis('equal')
ax.axis('off')
cax = fig.add_axes([0.15, 0.9, 0.25, 0.01])
fig.colorbar(tc, cax=cax, orientation='horizontal')
if not oneplot:
    showit('amg_e0.pdf')

if oneplot:
    ax = axs[1,0]
else:
    fig, ax = plt.subplots(figsize=figsize)
ax.tripcolor(V[:,0], V[:,1], E, Ge, vmin=0, vmax=1.5, edgecolor='w')
ax.axis('equal')
ax.axis('off')
if not oneplot:
    showit('amg_Ge0.pdf')

# Create a (fake) view of the coarse problem
# The real graph of Ac is not planar here, making it hard to visualize
Ac = ml.levels[1].A.tocoo()
Imap = np.where(ml.levels[0].splitting==1)[0]
A = A.tocoo()
Vc = np.zeros((Ac.shape[0],2))
Vc[:,0] = V[Imap,0]
Vc[:,1] = V[Imap,1]
tric = scipy.spatial.Delaunay(Vc)
Ec = tric.simplices

if oneplot:
    ax = axs[2,0]
else:
    fig, ax = plt.subplots(figsize=figsize)
ax.tripcolor(Vc[:,0], Vc[:,1], Ec, -ec, vmin=0, vmax=1.5, edgecolor='w', lw=2)
ax.axis('equal')
ax.axis('off')
if not oneplot:
    showit('amg_ec.pdf')

if oneplot:
    ax = axs[3,0]
else:
    fig, ax = plt.subplots(figsize=figsize)
ax.tripcolor(V[:,0], V[:,1], E, e1, vmin=0, vmax=1.5, edgecolor='w')
ax.axis('equal')
ax.axis('off')
if not oneplot:
    showit('amg_e1.pdf')

if oneplot:
    ax = axs[0,1]
else:
    fig, ax = plt.subplots(figsize=figsize)
ax.spy(A, marker='s', ms=3, color='k', clip_on=False, markerfacecolor='w')
ax.axis('off')
if not oneplot:
    showit('amg_Anz.pdf')

if oneplot:
    ax = axs[1,1]
else:
    fig, ax = plt.subplots(figsize=figsize)
ax.tripcolor(V[:,0], V[:,1], E, Ge, vmin=0, vmax=1.5, edgecolor='w')
ax.plot(Vc[:,0], Vc[:,1], marker='s', lw=3, ms=10, linestyle='', markeredgewidth=3,
        color='r', markerfacecolor='w')
ax.axis('equal')
ax.axis('off')
if not oneplot:
    showit('amg_Ge_withCpts.pdf')

if oneplot:
    ax = axs[2,1]
else:
    fig, ax = plt.subplots(figsize=figsize)
ax.tripcolor(V[:,0], V[:,1], E, Ge, vmin=0, vmax=1.5)#, edgecolor='w')
ax.plot(Vc[:,0], Vc[:,1], marker='s', lw=3, ms=10, linestyle='', markeredgewidth=3,
        color='r', markerfacecolor='w')
ax.triplot(Vc[:,0], Vc[:,1], Ec, color='0.8')
#for i, j in zip(Ac.row, Ac.col):  
#    ic = Imap[i]
#    jc = Imap[j]
#    ax.plot([V[ic,0], V[jc,0]], [V[ic,1], V[jc,1]])
ax.axis('equal')
ax.axis('off')
if not oneplot:
    showit('amg_Ge_withgraph.pdf')

if oneplot:
    ax = axs[3,1]
else:
    fig, ax = plt.subplots(figsize=figsize)
ax.spy(Ac, marker='s', ms=3, color='k', clip_on=False, markerfacecolor='w')
ax.axis('off')
if not oneplot:
    showit('amg_Acnz.pdf')
else:
    showit('amg.pdf')
