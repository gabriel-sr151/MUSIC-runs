#based on https://github.com/JETSCAPE/SummerSchool2020/blob/master/hydro_session/hydro_movie-TestRun.ipynb

from numpy import *
from os import path
home = path.expanduser("~")

from matplotlib import animation
import matplotlib.pyplot as plt
import sys # GSR

# define format for the plots
'''import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [6., 4.5]
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['xtick.minor.width'] = 0.8
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 0.8
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['ytick.direction'] = "in"
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['legend.numpoints'] = 1
mpl.rcParams['font.size'] = 15
mpl.rcParams['savefig.format'] = "pdf"
'''
#
#sys.exit() # INTERRUPT CODE for debugging

working_path = path.join(home, "MUSIC/final_data_files")

print(working_path)

# define the contour levels
#levels = linspace(0.13, 0.30, 50)

# define a customized color map
'''colors1 = array([[1, 1, 1, 1]])
colors2 = plt.cm.jet(linspace(0., 1, 10))
colors = vstack((colors1, colors2))
my_cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)

'''
# change the following line to your result folder
TestResultFolder = "acausality-stuff/test_run-bin-out-QCD-EoS-2"

# load hydrodynamic evolution data
data = fromfile(path.join(working_path, TestResultFolder,"evolution_for_movie_xyeta.dat"), dtype=float32)

#print(data.shape)


# read header about the grid information
header = data[0:12]

print(header) #ok 
#print(data)
#print(data[12:]) #the rest of the data seems to not be there
                 #do not set the T_cut in the input file to large values!!!! now it's ok

# read in data and reshape it to the correct form -- 
data = data[12:].reshape(-1, int(header[-1]))

#print(data.shape)

# get the list for tau frame
tau_list = unique(data[:, 0])
ntau = len(tau_list)
tau0 = header[0] 
dtau = header[1]
tau_list = array([tau0 + i*dtau for i in range(ntau)])

print(ntau, tau0, dtau)
#print(data[:, 0])

#sys.exit()

# define 3D grid in x, y, and eta_s (space-time rapidity)
neta = int(header[8])
eta_size = -2.*header[10]
deta = header[9]
eta = array([-eta_size/2.+i*deta for i in range(neta)])

nx = int(header[2])
x_size = 2.*abs(header[4])
dx = header[3]
x = array([-x_size/2.+i*dx for i in range(nx)])

ny = int(header[5])
y_size = 2.*abs(header[7])
dy = header[6]
y = array([-y_size/2.+i*dy for i in range(ny)])


# create 3D grids for energy density, temperature, and velocity
ed = zeros([ntau, neta, nx, ny])
pr = zeros([ntau, neta, nx, ny]) #GSR
T  = zeros([ntau, neta, nx, ny])
cs2 = zeros([ntau, neta, nx, ny]) #GSR
vx = zeros([ntau, neta, nx, ny])
vy = zeros([ntau, neta, nx, ny])
bulkPI = zeros([ntau, neta, nx, ny]) #GSR

for itau in range(ntau):
    idx = (abs(data[:, 0] - itau) < 0.1)
    data_cut = data[idx, :]
    for igrid in range(len(data_cut[:, 0])):
        x_idx   = int(data_cut[igrid, 1] + 0.1)
        y_idx   = int(data_cut[igrid, 2] + 0.1)
        eta_idx = int(data_cut[igrid, 3] + 0.1)
        u0 = sqrt(1. + data_cut[igrid, 9]**2.
                  + data_cut[igrid, 10]**2. + data_cut[igrid, 11]**2)
        ed[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 5]
        T[itau, eta_idx, x_idx, y_idx]  = data_cut[igrid, 7]
        vx[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 9]/u0
        vy[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 10]/u0

# print out some useful information about the evolution file
print("Read in data completed.")

print(nx, x[0], x[-1], dx)

print("nx = {0}, x_min = {1:.2f} fm, x_max = {2:.2f} fm, dx = {3:.2f} fm".format(nx, x[0], x[-1], dx))
print("ny = {0}, y_min = {1:.2f} fm, y_max = {2:.2f} fm, dy = {3:.2f} fm".format(ny, y[0], y[-1], dy))
print("neta = {0}, eta_min = {1:.2f} fm, eta_max = {2:.2f} fm, deta = {3:.2f}".format(neta, eta[0], eta[-1], deta))