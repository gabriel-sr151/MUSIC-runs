#based on https://github.com/JETSCAPE/SummerSchool2020/blob/master/hydro_session/hydro_movie-TestRun.ipynb
from numpy import *
from os import path
home = path.expanduser("~")

from matplotlib import animation
import matplotlib.pyplot as plt
import sys # GSR
import matplotlib.patches as mpatches

# define format for the plots
import matplotlib as mpl
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
#
#sys.exit() # INTERRUPT CODE for debugging

working_path = path.join(home, "MUSIC/final_data_files")
#print(working_path)

# define the contour levels
levelsT = linspace(0.13, 0.30, 50)
levelsbulk = linspace(-0.10, 0.40, 50)
levelscaus = linspace(-0.1, 1.20, 50)
levelsVW = linspace(-0.2, 0.2, 50)
levelscaus = linspace(-0.1, 1.20, 50)
levelsVW = linspace(-0.2, 0.2, 50)
levels2status = [-2, 0, 2]
levels3status = [-0.5, 0.5, 1.5, 2.5,3.5]


# define a custmized color map
colors1 = array([[1, 1, 1, 1]])
colors2 = plt.cm.jet(linspace(0., 1, 10))
colors = vstack((colors1, colors2))
my_cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)


# define the contour levels
#levels = linspace(0.13, 0.30, 50)

# define a customized color map
colors1 = array([[1, 1, 1, 1]])
colors2 = plt.cm.jet(linspace(0., 1, 10))
colors = vstack((colors1, colors2))
colors2stat = ['black','green','red']
colors3stat = ['green','yellow','red']
total_status_labels = ['causal & stable','acausal & stable', 'acausal & unstable']
my_cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)
my_cmap_2stat = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors2stat)
my_cmap_3stat = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['black'] + colors3stat)


# change the following line to your result folder
TestResultFolder = "acausality-stuff/run4-hard" 
                                           #run 1 -- pure bulk with bulk_relax_time_factor = 1/14.55 default bulk_relax_time_factor
                                           #run 2 -- pure bulk with bulk_relax_time_factor = 19.34 in input file    
                                           #run 3 (ERR) -- pure bulk with bulk_relax_time_factor = 1/19.36 in input file 
                                                    # ERR ---  never insert 1/14.55 in the input
                                                    # file. put the numerical value instead (0.0687).
                                           #run 4 (ERR) -- pure bulk with bulk_relax_time_factor = 1/15.0 in input file ERR
                                           #run 5 -- same as run1 for double checking -- something is weird when considering another 
                                                     #tau_bulk factor
                                           #run 1 finer -- run 1 input file with smaller delta_tau
                                           #run 4 - hard -- pure bulk with bulk_relax_time_factor = 1/15.0 changed in code
                                           #run 3 - hard -- pure bulk with bulk_relax_time_factor = 1/19.34 changed in code
                                               #>> for this run there was a energy density factor warning
                                           #run 4 - dcheck -- double check run -- input file implementation error found
                                           # pure bulk with bulk_relax_time_factor = 1/15.0 changed in code                                               
                                           #run 6 (ERR) - locally boosted IC vx -> relat_sum(vx,0.2vx) bulk_relax_time_factor = 1/15.0   
                                           # ERR implementation error: boost with wrong sign 
                                           #run 7 -- locally boosted IC vx -> relat_sum(vx,0.5vx) bulk_relax_time_factor = 1/15.0 
                                           #run7-vx+2vx -- locally boosted IC vx -> relat_sum(vx,2vx) bulk_relax_time_factor = 1/15.0 
                                           #run7-vx+1vx -- locally boosted IC vx -> relat_sum(vx,vx) bulk_relax_time_factor = 1/15.0
                                           #run8-global -- global boost with vx -> relat_sum(vx,-0.8) bulk_relax_time_factor = 1/15.0 
                                           #run8-global -- global boost with vx -> relat_sum(vx,-0.99) bulk_relax_time_factor = 1/15.0    
                                           # ------------------ all runs above this line contained an error postprocessing regarding Pi/(e+P)
                                           #                    because the bulk printed in Pi/(e+p) and not Pi  
                                           # run4XL -- pure bulk with bulk_relax_time_factor = 1/15.0 but with larger tau window  run 4 - hard is the reference          


bulk_relax_time_factor = 1./15. #MUSIC_default 1/14.55



# load hydrodynamic evolution data
data = fromfile(path.join(working_path, TestResultFolder,"evolution_all_xyeta.dat"), dtype=float32)

#print(data.shape)


# read header about the grid information
header = data[0:16]

print(header) #ok 
#print(data)
#print(data[12:]) #the rest of the data seems to not be there
                 #do not set the T_cut in the input file to large values!!!! now it's ok

# read in data and reshape it to the correct form -- 
data = data[16:].reshape(-1, int(header[-1]))

# get the list for tau frame
tau_list = unique(data[:, 0])
ntau = len(tau_list)
tau0 = header[0] 
dtau = header[1]
tau_list = array([tau0 + i*dtau for i in range(ntau)])

print(ntau, tau0, dtau)

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
vz = zeros([ntau, neta, nx, ny]) #GSR
bulkPI_norm = zeros([ntau, neta, nx, ny]) #GSR  -- Pi/(e+p)
pixx_norm = zeros([ntau, neta, nx, ny])  #GSR  -- SHEAR_TENSOR/(e+p)
pixy_norm = zeros([ntau, neta, nx, ny])
pixz_norm = zeros([ntau, neta, nx, ny])
piyy_norm = zeros([ntau, neta, nx, ny])
piyz_norm = zeros([ntau, neta, nx, ny]) 
pixx = zeros([ntau, neta, nx, ny])  #GSR  -- SHEAR_TENSOR
pixy = zeros([ntau, neta, nx, ny])
pixz = zeros([ntau, neta, nx, ny])
piyy = zeros([ntau, neta, nx, ny])
piyz = zeros([ntau, neta, nx, ny]) 



wchar2 = zeros([ntau, neta, nx, ny]) #GSR -- characteristic speed for pure bulk simulations
v2 = zeros([ntau, neta, nx, ny]) #GSR -- VW criterion
causality_status = zeros([ntau, neta, nx, ny]) #GSR
V2w2_status = zeros([ntau, neta, nx, ny]) #GSR                            
causal_AND_v2w2_status = zeros([ntau, neta, nx, ny]) #GSR


for itau in range(ntau):
    idx = (abs(data[:, 0] - itau) < 0.1)
    data_cut = data[idx, :]
    for igrid in range(len(data_cut[:, 0])):
        x_idx   = int(data_cut[igrid, 1] + 0.1)
        y_idx   = int(data_cut[igrid, 2] + 0.1)
        eta_idx = int(data_cut[igrid, 3] + 0.1)
        u0 = sqrt(1. + data_cut[igrid, 8]**2.
                  + data_cut[igrid, 9]**2. + data_cut[igrid, 10]**2)
        ed[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 4]
        pr[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 5]
        T[itau, eta_idx, x_idx, y_idx]  = data_cut[igrid, 6]
        cs2[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 7]
        vx[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 8]/u0
        vy[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 9]/u0
        vz[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 10]/u0
        #rhob[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, ]
        #muB[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, ]
        pixx_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 11]# shear/(e+p)
        pixy_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 12]
        pixz_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 13]
        piyy_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 14]
        piyz_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 15]


        bulkPI_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 16] # Pi/(e+p)

        v2[itau, eta_idx, x_idx, y_idx] = vx[itau, eta_idx, x_idx, y_idx]**2 + vy[itau, eta_idx, x_idx, y_idx]**2 \
                                       + vz[itau, eta_idx, x_idx, y_idx]**2
        
        ##################
        #   CHARACTERISTIC SPEEDS COMPUTATION BEGINS HERE ----------------------------------------
        #   below we consider tau_BULK = (zeta/(e+p))*(factor)*(1/(1/3-cs2)) change accordingly
        #   SOURCE -- 
        #   BRASIL!  
        ###################

        wchar2[itau, eta_idx, x_idx, y_idx] = cs2[itau, eta_idx, x_idx, y_idx] \
              + (1.0/bulk_relax_time_factor)*( (1.0/3.0 - cs2[itau, eta_idx, x_idx, y_idx])**(2.0) )\
                /(1.0 + bulkPI_norm[itau, eta_idx, x_idx, y_idx] )
                                                    
        
        ##################
        #   NECESSARY CAUSALITY CONDITIONS AND VW CRITERION BEGINS HERE ----------------------------------------
        #   below we consider tau_BULK = (zeta/(e+p))*(factor)*(1/(1/3-cs2)) change accordingly 
        ###################


        if (wchar2[itau, eta_idx, x_idx, y_idx] < 1.0):
            
            causality_status[itau, eta_idx, x_idx, y_idx] = 0

        else:

            causality_status[itau, eta_idx, x_idx, y_idx] = 1


        if wchar2[itau, eta_idx, x_idx, y_idx]*v2[itau, eta_idx, x_idx, y_idx] < 1.0:
            
            V2w2_status[itau, eta_idx, x_idx, y_idx] = 0

        else:

            V2w2_status[itau, eta_idx, x_idx, y_idx] = 1

        if (wchar2[itau, eta_idx, x_idx, y_idx] < 1.0 and (v2[itau, eta_idx, x_idx, y_idx] < 1.0)):
           
           causal_AND_v2w2_status[itau, eta_idx, x_idx, y_idx] = 1

        else: 

            if wchar2[itau, eta_idx, x_idx, y_idx]*v2[itau, eta_idx, x_idx, y_idx] < 1.0:
               
               causal_AND_v2w2_status[itau, eta_idx, x_idx, y_idx] = 2

            else:

                causal_AND_v2w2_status[itau, eta_idx, x_idx, y_idx] = 3   













#wchar2 = cs2 + 15.0*((1.0/3.0 - cs2)**(2.0))/(1 + bulkPI/(e+P))  




# print out some useful information about the evolution file
print("Read in data completed.")

#print(nx, x[0], x[-1], dx)

print("nx = {0}, x_min = {1:.2f} fm, x_max = {2:.2f} fm, dx = {3:.2f} fm".format(nx, x[0], x[-1], dx))
print("ny = {0}, y_min = {1:.2f} fm, y_max = {2:.2f} fm, dy = {3:.2f} fm".format(ny, y[0], y[-1], dy))
print("neta = {0}, eta_min = {1:.2f} fm, eta_max = {2:.2f} fm, deta = {3:.2f}".format(neta, eta[0], eta[-1], deta))


final_plots_folder = path.join(working_path, TestResultFolder)


######################################---PLOTS----########################################################




