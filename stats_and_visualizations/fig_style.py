import matplotlib
"""
matplotlib setup
"""
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
matplotlib.rc('font', family = 'serif', size=11)
matplotlib.rc('axes', linewidth = 0.5)
matplotlib.rcParams['xtick.major.width']=0.5
matplotlib.rcParams['ytick.major.width']=0.5
matplotlib.rcParams['xtick.minor.width']=0.3
matplotlib.rcParams['ytick.minor.width']=0.3
matplotlib.rcParams['axes.labelsize']=12
matplotlib.rcParams['axes.titlesize']=14
matplotlib.rcParams['legend.fontsize']=11
matplotlib.rcParams['xtick.labelsize']=10
matplotlib.rcParams['ytick.labelsize']=10

def set_ticks():
    """
    Set inward ticks on all sides of the plot
    """
    matplotlib.pyplot.gca().tick_params('both',which='both',direction='in',right=True,top=True)