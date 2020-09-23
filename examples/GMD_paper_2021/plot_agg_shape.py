from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
import numpy as np
from scipy import stats
from numpy import loadtxt
import glob
from aggregation import mcs
###########################################################################################
def setlabel(ax, label, loc=2, borderpad=0.6, **kwargs):
    legend = ax.get_legend()
    if legend:
        ax.add_artist(legend)
    line, = ax.plot(np.NaN,np.NaN,color='none',label=label)
    label_legend = ax.legend(handles=[line],loc=loc,handlelength=0,handleheight=0,handletextpad=0,borderaxespad=0,borderpad=borderpad,frameon=False,**kwargs)
    label_legend.remove()
    ax.add_artist(label_legend)
    line.remove()
############################################################################################

shapePath = '/work/lvonterz/SSRGA/shapefiles/'
monotypes = ['dori','leinSubs','lterzi']
dipole_res_OriLeinonen = 40e-6
fig,axes = plt.subplots(nrows=4,ncols=3,figsize=(14,18))
nrow=0; nlabel=0; ncol=0
figlabels = 'abcdefghijklmn'
for mono in monotypes:
    shapefiles = sorted(glob.glob(shapePath+mono+'/*.txt'))
    for s in shapefiles:
        print(s)
        print('ncol:',ncol)
        print('nrow:',nrow)
        if mono == 'dori':
            agg = np.loadtxt(s,skiprows=6)
            agg = agg[:,1:4]
        else:
            agg = np.loadtxt(s)
        ax = axes[nrow,ncol]
        ax.scatter(agg[:,2], agg[:,0], lw=(0,), s=1, c='k')
        ax.set_xlim((agg[:,2].min()*1.3, agg[:,2].max()*1.3))
        ax.set_ylim((agg[:,0].min()*1.3, agg[:,0].max()*1.3))
        #ax.set_xlabel('distance in Z')
        #ax.set_ylabel('distance in X')
        if (mono == 'dori' or mono == 'leinSubs'):
            agg = agg*dipole_res_OriLeinonen
        Dmax =  mcs.minimum_covering_sphere(agg)[1]*2*1e3
        ax.tick_params(axis='both',which='both', bottom=False, top=False, left = False, right = False, labelbottom=False,labelleft=False)
        setlabel(ax,'('+figlabels[nlabel]+')',prop={'size':18,'weight':'bold'})
        setlabel(ax,r'$D_{max}$=%2.1f mm'%float(Dmax),loc=3,prop={'size':18,'weight':'bold'})
        plt.tight_layout()
        ncol +=1
        nlabel+=1
        if ncol == 3:
            nrow +=1
            ncol = 0
outName = 'aggregate_flyer2.0'
plt.savefig(outName+'.png',dpi=600,bbox_inches='tight')
plt.savefig(outName+'.pdf',dpi=400,bbox_inches='tight')
plt.close()
