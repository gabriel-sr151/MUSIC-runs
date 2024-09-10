#based on https://github.com/JETSCAPE/SummerSchool2020/blob/master/hydro_session/hydro_movie-TestRun.ipynb
from numpy import *
from os import path
home = path.expanduser("~")

from matplotlib import animation
import matplotlib.pyplot as plt
import sys # GSR

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
my_cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)


# change the following line to your result folder
TestResultFolder = "acausality-stuff/run8-global-boost-2" 
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
bulkPI = zeros([ntau, neta, nx, ny]) #GSR

wchar2 = zeros([ntau, neta, nx, ny]) #GSR -- characteristic speed for pure bulk simulations
v2 = zeros([ntau, neta, nx, ny]) #GSR -- VW criterion
causality_status = zeros([ntau, neta, nx, ny]) #GSR
V2w2_status = zeros([ntau, neta, nx, ny]) #GSR                            



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
        #pixx[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, ] #when shear is activated
        #pixy[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, ]
        #pixz[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, ]
        #piyy[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, ]
        #piyz[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, ]
        bulkPI[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 11]

        wchar2[itau, eta_idx, x_idx, y_idx] = cs2[itau, eta_idx, x_idx, y_idx] \
              + (1.0/bulk_relax_time_factor)*( (1.0/3.0 - cs2[itau, eta_idx, x_idx, y_idx])**(2.0) )\
                /(1.0 + bulkPI[itau, eta_idx, x_idx, y_idx]/(ed[itau, eta_idx, x_idx, y_idx]+pr[itau, eta_idx, x_idx, y_idx]+0.0000001) )
                                                     # i put a regulator otherwise things diverge
          
        v2[itau, eta_idx, x_idx, y_idx] = vx[itau, eta_idx, x_idx, y_idx]**2 + vy[itau, eta_idx, x_idx, y_idx]**2 \
                                       + vz[itau, eta_idx, x_idx, y_idx]**2

        if (wchar2[itau, eta_idx, x_idx, y_idx] < 1.0) or (T[itau, eta_idx, x_idx, y_idx] < 0.150):
            
            causality_status[itau, eta_idx, x_idx, y_idx] = 0.0

        else:

            causality_status[itau, eta_idx, x_idx, y_idx] = 1.0


        if wchar2[itau, eta_idx, x_idx, y_idx]*v2[itau, eta_idx, x_idx, y_idx] < 1.0:
            
            V2w2_status[itau, eta_idx, x_idx, y_idx] = 0.0

        else:

            V2w2_status[itau, eta_idx, x_idx, y_idx] = 1.0



#wchar2 = cs2 + 15.0*((1.0/3.0 - cs2)**(2.0))/(1 + bulkPI/(e+P))  




# print out some useful information about the evolution file
print("Read in data completed.")

#print(nx, x[0], x[-1], dx)

print("nx = {0}, x_min = {1:.2f} fm, x_max = {2:.2f} fm, dx = {3:.2f} fm".format(nx, x[0], x[-1], dx))
print("ny = {0}, y_min = {1:.2f} fm, y_max = {2:.2f} fm, dy = {3:.2f} fm".format(ny, y[0], y[-1], dy))
print("neta = {0}, eta_min = {1:.2f} fm, eta_max = {2:.2f} fm, deta = {3:.2f}".format(neta, eta[0], eta[-1], deta))


'''def zeta_ov_s(T_in_fm):

    #the simulation was ran with 'T_dependent_Bulk_to_S_ratio 1'

'''
    #zeta/s parametrization from src/transport_coeffs.cpp L194
    #T input in fm  
'''
    hbarc = 1### 
    T_in_GeV = hbarc*T_in_fm   
    Ttr = 0.18
    T_ov_Ttr = T_in_GeV/Ttr
    A1=-13.77, A2=27.55, A3=13.45
    lambda1=0.9, lambda2=0.25, lambda3=0.9, lambda4=0.22;
    sigma1=0.025, sigma2=0.13, sigma3=0.0025, sigma4=0.022;

    bulk = A1*T_ov_Ttr*T_ov_Ttr + A2*T_ov_Ttr - A3;

    if T_in_GeV < 0.995*Ttr:

        bulk = (lambda3*np.exp((T_ov_Ttr-1)/sigma3) + lambda4*np.exp((T_ov_Ttr-1)/sigma4) + 0.03)

    if T_in_GeV > 1.05*Ttr:

        bulk = (lambda1*exp(-(dummy-1)/sigma1) + lambda2*exp(-(dummy-1)/sigma2) + 0.001)

    return bulk    
'''


final_plots_folder = path.join(working_path, TestResultFolder)



# make a 2D meshgrid in the transverse plane
X, Y = meshgrid(x, y)



# make the contour plot
'''tau_idx = -1 #int(ntau*(2/3)) # 0 for the initial condition

fig = plt.figure()
cont = plt.contourf(X, Y, bulkPI[tau_idx, 0, :, :]/(ed[tau_idx, 0, :, :]+pr[tau_idx, 0, :, :]), levelsbulk, cmap=my_cmap, extend='both')
cbar = fig.colorbar(cont)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$y$ (fm)")
plt.text(1.0, 10.0, r'$\tau = {0:3.1f}$ fm'.format(tau_list[tau_idx]))
plt.tight_layout()
plt.savefig(f"{final_plots_folder}/Bulk_ov_e_p_Contour_XY-initial-tau_{tau_idx}-of-{ntau}")
'''
# make the contour plot
'''fig = plt.figure()
cont = plt.contourf(X, Y, cs2[0, 0, :, :], levelsbulk, cmap=my_cmap, extend='both')
cbar = fig.colorbar(cont)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$y$ (fm)")
plt.tight_layout()
plt.savefig(f"{final_plots_folder}/TestRun_cs2_Contour_XY")
'''



#tau_idx = 0*int(ntau*(2/3)) # 0 for the initial condition

# make the contour plot
'''fig = plt.figure()
cont = plt.contourf(X, Y, wchar2[tau_idx, 0, :, :], levelscaus, cmap=my_cmap, extend='both')
cbar = fig.colorbar(cont)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$y$ (fm)")
plt.text(1.0, 10.0, r'$\tau = {0:3.1f}$ fm'.format(tau_list[tau_idx]))
plt.tight_layout()
plt.savefig(f"{final_plots_folder}/TestRun_caus_Contour_XY-tau_{tau_idx}-of-{ntau}")
'''

'''Tau, X = meshgrid(tau_list, x)

y_idx = int(ny/2)  # pick the central point in the y direction

fig = plt.figure()
cont = plt.contourf(X, Tau, wchar2[:, 0, :, y_idx].transpose(), levelscaus,
                    cmap=my_cmap, extend='both')
cbar = fig.colorbar(cont)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$\tau$ (fm/c)")
plt.text(1.0, 10.0, r'$y = {0:3.1f}$ fm'.format(y[y_idx]))
#plt.tight_layout()
plt.savefig(f"{final_plots_folder}/TestRun_wchar2_Contour_TauX")
'''

'''tau_idx = -1 #*int(ntau*(2/3)) # 0 for the initial condition

fig = plt.figure()
cont = plt.contourf(X, Y, v2[tau_idx, 0, :, :]*wchar2[tau_idx, 0, :, :], levelsVW, cmap=my_cmap, extend='both')
cbar = fig.colorbar(cont)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$y$ (fm)")
plt.text(1.0, 10.0, r'$\tau = {0:3.1f}$ fm'.format(tau_list[tau_idx]))
plt.tight_layout()
plt.savefig(f"{final_plots_folder}/TestRun_V2W2_Contour_XY-tau_{tau_idx}-of-{ntau}")
'''


'''tau_idx = -1 #int(ntau*(2/3)) # 0 for the initial condition

# make the contour plot
fig = plt.figure()
cont = plt.contourf(X, Y, causality_status[tau_idx, 0, :, :], levelscaus, cmap=my_cmap, extend='both')
cbar = fig.colorbar(cont)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$y$ (fm)")
plt.text(1.0, 10.0, r'$\tau = {0:3.1f}$ fm'.format(tau_list[tau_idx]))
plt.tight_layout()
plt.savefig(f"{final_plots_folder}/TestRun_causality_status_Contour_XY-tau_{tau_idx}-of-{ntau}")
'''

######################################

X, Y = meshgrid(x, y)

# first plot the first frame as a contour plot
fig = plt.figure()
cont = plt.contourf(X, Y, v2[0, 0, :, :].transpose(), levelscaus,
                    cmap=my_cmap, extend='both')
time_text = plt.text(-6, 6, r"$\tau = {0:4.2f}$ fm/c".format(tau_list[0]))
cbar = fig.colorbar(cont)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$y$ (fm)")
plt.xlim([-8, 8])
plt.ylim([-8, 8])
plt.tight_layout()
    

# define animation function to update the contour at every time frame
def animate(i): 
    global cont, time_text
    for c in cont.collections: # collections WILL BE REMOVED SOON from matplotlib
        c.remove()  # removes only the contours, leaves the rest intact
    cont = plt.contourf(X, Y, causality_status[i, 0, :, :], levelscaus, cmap=my_cmap, extend='both')
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[i]))
    return cont, time_text

# create the animation
anim = animation.FuncAnimation(fig, animate, frames=ntau, repeat=False)

# save the animation to a file
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/animation_causality-status.gif", writer=writergif)