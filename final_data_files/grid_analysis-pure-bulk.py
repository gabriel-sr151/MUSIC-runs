#based on https://github.com/JETSCAPE/SummerSchool2020/blob/master/hydro_session/hydro_movie-TestRun.ipynb
import numpy as np
from numpy import *
from os import path
home = path.expanduser("~")

from matplotlib import animation
import matplotlib.pyplot as plt
import sys # GSR
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter  # For formatting the colorbar

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



# change the following line to your result folder
TestResultFolder = "acausality-stuff/run_paper-w_PI-th-zovs8" 
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
                                           # run4XL -- pure bulk with bulk_relax_time_factor = 1/15.0 but with larger tau window
                                           #   run 4 - hard is the reference          
                                           #run9-QRVoff -- quest revert regulator off
                                           #run4XL-echo+ -- quest revert activation test eps_scale 0.02
                                           #run4XL-echo+v2 -- quest revert activation test eps_scale 0.1
                                           #run_paper_min_IS -- minimal IS with zeta/s option 1 (default)
                                           #run_paper_zovs8 -- minimal IS with zeta/s option 8


#ATTENTION!!!!!!!!!!!!!!!!!!!!!!! When running the pure bulk case here, I mean pure bulk minimal Israel-Stewart
# then, make sure incl_delPIPI_GSR = 0 in src/dissipative.cpp

bulk_relax_time_factor = 1./15. #MUSIC_default 1/14.55
incl_delPIPI = 0



# load hydrodynamic evolution data
data = fromfile(path.join(working_path, TestResultFolder,"evolution_all_xyeta.dat"), dtype=float32)

#print(type(data))


# read header about the grid information
header = data[0:16]

#print(header) #ok 
#print(data.shape)

#sys.exit()
#
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
bulkPI_norm = zeros([ntau, neta, nx, ny]) #GSR  -- Pi/(e+p)


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
        #pixx[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, ] #when shear is activated
        #pixy[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, ]
        #pixz[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, ]
        #piyy[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, ]
        #piyz[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, ]
        bulkPI_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 11] # Pi/(e+p)

        '''bulkPI[itau, eta_idx, x_idx, y_idx] = bulkPI_norm[itau, eta_idx, x_idx, y_idx]\
             *(ed[itau, eta_idx, x_idx, y_idx] + pr[itau, eta_idx, x_idx, y_idx])'''
    
        
        ##################
        #   wchar2 = cs2 + (zeta/tau_BULK)*(1/(e+p+PI))
        #   below we consider tau_BULK = (zeta/(e+p))*(factor)*(1/(1/3-cs2)) change accordingly 
        ###################

        zeta_ov_tau_PI_e_pl_p = (1.0/bulk_relax_time_factor)*( (1.0/3.0 - cs2[itau, eta_idx, x_idx, y_idx])**(2.0) ) 
        #zeta/((e+p)tau_PI)
        
        delPIPI_OV_tauPI = incl_delPIPI*(2.0/3.0)

        wchar2[itau, eta_idx, x_idx, y_idx] = cs2[itau, eta_idx, x_idx, y_idx] \
              + (zeta_ov_tau_PI_e_pl_p + delPIPI_OV_tauPI*bulkPI_norm[itau, eta_idx, x_idx, y_idx])\
                /(1.0 + bulkPI_norm[itau, eta_idx, x_idx, y_idx])
              
              #+ (1.0/bulk_relax_time_factor)*( (1.0/3.0 - cs2[itau, eta_idx, x_idx, y_idx])**(2.0) )\
              #  /(1.0 + bulkPI_norm[itau, eta_idx, x_idx, y_idx] )
                                                    
        v2[itau, eta_idx, x_idx, y_idx] = vx[itau, eta_idx, x_idx, y_idx]**2 + vy[itau, eta_idx, x_idx, y_idx]**2 \
                                       + vz[itau, eta_idx, x_idx, y_idx]**2

        '''if (wchar2[itau, eta_idx, x_idx, y_idx] > 0.0 and wchar2[itau, eta_idx, x_idx, y_idx] < 1.0):
            
            causality_status[itau, eta_idx, x_idx, y_idx] = 0

        else:

            causality_status[itau, eta_idx, x_idx, y_idx] = 1

        #end causality test    


        if wchar2[itau, eta_idx, x_idx, y_idx]*v2[itau, eta_idx, x_idx, y_idx] < 1.0:
            
            V2w2_status[itau, eta_idx, x_idx, y_idx] = 0

        else:

            V2w2_status[itau, eta_idx, x_idx, y_idx] = 1

        #end vw test'''     

        if (wchar2[itau, eta_idx, x_idx, y_idx] > 0.0 and wchar2[itau, eta_idx, x_idx, y_idx] < 1.0):
           
           causal_AND_v2w2_status[itau, eta_idx, x_idx, y_idx] = 1

        else: 

            if wchar2[itau, eta_idx, x_idx, y_idx] < 0.0:

                causal_AND_v2w2_status[itau, eta_idx, x_idx, y_idx] = 4

            else:    

                if wchar2[itau, eta_idx, x_idx, y_idx]*v2[itau, eta_idx, x_idx, y_idx] < 1.0:
               
                    causal_AND_v2w2_status[itau, eta_idx, x_idx, y_idx] = 2

                else:

                    causal_AND_v2w2_status[itau, eta_idx, x_idx, y_idx] = 3 

        #end causality-vw test              

    








#wchar2 = cs2 + 15.0*((1.0/3.0 - cs2)**(2.0))/(1 + bulkPI/(e+P))  




# print out some useful information about the evolution file
print("Read in data completed.")

#print(nx, x[0], x[-1], dx)

print("nx = {0}, x_min = {1:.2f} fm, x_max = {2:.2f} fm, dx = {3:.2f} fm".format(nx, x[0], x[-1], dx))
print("ny = {0}, y_min = {1:.2f} fm, y_max = {2:.2f} fm, dy = {3:.2f} fm".format(ny, y[0], y[-1], dy))
print("neta = {0}, eta_min = {1:.2f} fm, eta_max = {2:.2f} fm, deta = {3:.2f}".format(neta, eta[0], eta[-1], deta))



final_plots_folder = path.join(working_path, TestResultFolder)


########################################  PLOTS SETTINGS  #########


# define the contour levels
levelsT = linspace(0.13, 0.30, 50)
levelsT_finer = linspace(0.13, 0.30, 256)
levelsbulk = linspace(-0.10, 0.40, 50)
levelsed = linspace(0.0, 0.40, 50)
levelscaus = linspace(-0.1, 1.20, 50)
levelsVW = linspace(-0.2, 0.2, 50)
levelscaus = linspace(-0.1, 1.20, 50)
levelsVW = linspace(-0.2, 0.2, 50)
levels2status = [-2, 0, 2]
levels3status = [-0.5, 0.5, 1.5, 2.5,3.5]


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




####################################################




###################################### contour plot for vw status pcolormesh

levels4status = [-0.5, 0.5, 1.5, 2.5,3.5,4.5]
colors4stat = ['green','yellow','red','purple']
total_status_labels4 = ['good','bad', r'ugly ($vw > 1$)', r'ugly ($w^{2}<0$)']
my_cmap_4stat = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['black'] + colors4stat)


# Choose which frames to plot (e.g., first frame)
frame_idx_list = [0,40,50] #[int(ntau*(0/82)),int(ntau*(40/82)), int(ntau*(50/82))]

for tau_idx in frame_idx_list:

    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create 2D meshgrid 
    X, Y = np.meshgrid(x, y)

    # Plot using pcolormesh
    plot = ax.pcolormesh(X, Y, causal_AND_v2w2_status[tau_idx, 0, :, :],
                        shading='nearest',  # Sharp edges between cells
                        cmap=my_cmap_4stat,
                        vmin=min(levels4status),
                        vmax=max(levels4status))

    # Add time annotation
    time_text = ax.text(-7.4, -7, r"$\tau = {0:4.2f}$ fm/c".format(tau_list[tau_idx]), 
                    color='white', fontsize=16)

    # Add colorbar (if desired)
    #cbar = fig.colorbar(plot, ax=ax, ticks=levels4status)
    #cbar.set_label('Status')  # Customize as needed

    # Create legend
    if tau_idx == 0:

        legend_patches = [mpatches.Patch(color=colors4stat[i], label=total_status_labels4[i])
                    for i in range(len(colors4stat))]
        ax.legend(handles=legend_patches,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.02), # Position above the plot
            ncol = len(colors4stat),# Number of columns (equal to number of items for horizontal layout)
            frameon=False,
            fontsize=16)
    #endif    

    # Axis formatting
    ax.set_xlabel(r"$x$ (fm)", fontsize=16)
    ax.set_ylabel(r"$y$ (fm)", fontsize=16)
    ax.tick_params(axis='both', labelsize = 16)
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    ax.set_aspect('equal')  # Keep aspect ratio square
    plt.tight_layout()

    # Save or show
    plt.savefig(f"{final_plots_folder}/full_status_frame_{tau_idx}-of-{ntau}-pm.png")

#######################################temperature contour plot, blue color code

    blue_transparent = LinearSegmentedColormap.from_list(
        'blue_alpha',
        [(0, 0, 1, 0),  # Transparent (alpha=0)
        (0, 0, 1, 1)]  # Opaque blue (alpha=1)
    )



    # 1. First plot the temperature contour
    fig = plt.figure(figsize=(8,8))
    ax = plt.gca() 

    cont = ax.contourf(X, Y, T[tau_idx, 0, :, :], levelsT_finer, cmap=blue_transparent, extend='both')

    if tau_idx == 0:

        cbar = fig.colorbar(cont, 
            ax = ax,                
            location='top',    # Places colorbar above the plot
            orientation='horizontal',  # Ensures horizontal layout
            pad=0.02, # Adds small spacing between plot and colorbar
            shrink = 0.8,
            aspect = 40, # Controls colorbar thickness
            )  
        cbar.set_label('T (GeV)', fontsize = 16 , labelpad=10)
        cbar.formatter = FormatStrFormatter('%.2g')  # 3 significant figures
        cbar.update_normal(cont)  # Ensure updates apply
    #endif    
    

    # 2. Then plot the status boundary where status == 3
    status_data = causal_AND_v2w2_status[tau_idx, 0, :, :]
    cont_status = plt.contour(X, Y, (status_data == 2).astype(float), 
                            levels=[0.5],  # This will draw the boundary between 0 and 1
                            colors='yellow',                           
                            linewidths=0.75)
    
    ax.set_xlabel(r"$x$ (fm)", fontsize = 18)
    ax.set_ylabel(r"$y$ (fm)", fontsize = 18)
    ax.tick_params(axis='both', labelsize = 16)
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    ax.set_aspect('equal')
    ax.text(-7.4, -7, r'$\tau = {0:3.1f}$ fm/c'.format(tau_list[tau_idx]), fontsize = 18) 
    
    plt.tight_layout()
    plt.savefig(f"{final_plots_folder}/temperature_XY-tau_{tau_idx}-of-{ntau}-with-bad")

# end for in tau_idx

################ causal and v2w2 status animation ---2 with pcolormesh

levels4status = [-0.5, 0.5, 1.5, 2.5,3.5,4.5]
colors4stat = ['green','yellow','red','purple']
total_status_labels4 = ['good','bad', r'ugly ($vw > 1$)', r'ugly ($w^{2}<0$)']
my_cmap_4stat = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['black'] + colors4stat)


# make a 2D meshgrid in the transverse plane
X, Y = np.meshgrid(x, y)

# Set up the figure
fig = plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot first frame
img = ax.pcolormesh(X, Y, causal_AND_v2w2_status[0, 0, :, :].transpose(),
                    shading='nearest',  # No interpolation between cells
                    cmap=my_cmap_4stat,
                    vmin=min(levels4status),  # Minimum value for colormap
                    vmax=max(levels4status))  # Maximum value for colormap

time_text = ax.text(-7.4, -7, r"$\tau = {0:4.2f}$ fm/c".format(tau_list[0]), color='white')

# Create legend
legend_patches = [mpatches.Patch(color=colors4stat[i], label=total_status_labels4[i])
                 for i in range(len(colors4stat))]
ax.legend(handles=legend_patches,
          loc='center left',
          bbox_to_anchor=(1.05, 0.5),
          frameon=False)

# Axis labels and limits
ax.set_xlabel(r"$x$ (fm)")
ax.set_ylabel(r"$y$ (fm)")
ax.set_xlim([-8, 8])
ax.set_ylim([-8, 8])
plt.tight_layout(rect=[0, 0, 1, 1])

# Animation update function
def animate(i):
    # Update the image data
    img.set_array(causal_AND_v2w2_status[i, 0, :, :].ravel())
    
    # Update time text
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[i]))
    
    return img, time_text

# Create the animation
anim = animation.FuncAnimation(fig, animate, frames=ntau, repeat=False)

# Save the animation
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/animation_full-status-w-elli--pm.gif", writer=writergif)

plt.close()  # Close the figure to prevent display in notebooks



#####################################----temperature animation, blue color code

'''blue_transparent = LinearSegmentedColormap.from_list(
    'blue_alpha',
    [(0, 0, 1, 0),  # Transparent (alpha=0)
     (0, 0, 1, 1)]  # Opaque blue (alpha=1)
)

X, Y = meshgrid(x, y)

# first plot the first frame as a contour plot
fig = plt.figure()
cont = plt.contourf(X, Y, T[0, 0, :, :], levelsT_finer ,
                    cmap=blue_transparent, extend='both')
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
    cont = plt.contourf(X, Y, T[i, 0, :, :], levelsT, cmap=blue_transparent, extend='both')
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[i]))
    return cont, time_text

# create the animation
anim = animation.FuncAnimation(fig, animate, frames=ntau, repeat=False)

# save the animation to a file
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/temperature-XY.gif", writer=writergif)'''


'''#######################################temperature contour plot, rainbow color code

tau_idx = int(ntau*(40/82)) # 0 for the initial condition

fig = plt.figure()
cont = plt.contourf(X, Y, T[tau_idx, 0, :, :], levelsT, cmap=my_cmap, extend='both')
cbar = fig.colorbar(cont)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$y$ (fm)")
plt.xlim([-8, 8])
plt.ylim([-8, 8])
plt.tight_layout()
plt.savefig(f"{final_plots_folder}/temperature_XY-tau_{tau_idx}-of-{ntau}")
'''
######################################animation with velocity field

'''nskip = 2  # only plot every other point to speed up the live animation

X, Y = meshgrid(x, y)

v_mag = sqrt(vx[-1, 0, :, :]**2 + vy[-1, 0, :, :]**2.)

# first plot the first frame as the contour plot
levels2 = (linspace(0.10**0.25, 0.3**0.25, 30))**(4.)
fig = plt.figure()
cont = plt.contourf(X[::nskip, ::nskip], Y[::nskip, ::nskip],
                    T[0, 0, ::nskip, ::nskip],
                    levels2, cmap='Reds', extend='both')
Q = plt.quiver(X[::nskip, ::nskip], Y[::nskip, ::nskip],
              vy[0, 0, ::nskip, ::nskip],
              vx[0, 0, ::nskip, ::nskip],
              units='xy', scale_units='xy', scale=0.5, color='b')
time_text = plt.text(-7.5, 6.5, r"$\tau = {0:4.2f}$ fm/c".format(tau_list[0]))
cbar = fig.colorbar(cont)
plt.tight_layout()
plt.xlim(-8, 8)
plt.ylim(-8, 8)

# update the temperature contour and velocity vector field 
def update_quiver(num, Q, X, Y):
    global cont, time_text
    for c in cont.collections:
        c.remove()  # removes only the contours, leaves the rest intact
    cont = plt.contourf(X[::nskip, ::nskip], Y[::nskip, ::nskip],
                        T[num, 0, ::nskip, ::nskip],
                        levels2, cmap='Reds', extend='both')
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[num])) 
    
    U = vy[num, 0, ::nskip, ::nskip]
    V = vx[num, 0, ::nskip, ::nskip]
    
    Q = plt.quiver(X[::nskip, ::nskip], Y[::nskip, ::nskip],
                   U, V, units='xy', scale_units='xy', scale=0.5, color='b')
    return Q, cont, time_text  

# create the animation
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y),
                               frames=ntau, blit=False, repeat=False)

# save the animation
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/animation_Tandflow.gif", writer=writergif)
'''


'''
#####################################----temperature animation, Rainbow color code

X, Y = meshgrid(x, y)

# first plot the first frame as a contour plot
fig = plt.figure()
cont = plt.contourf(X, Y, T[0, 0, :, :], levelsT ,
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
    cont = plt.contourf(X, Y, T[i, 0, :, :], levelsT, cmap=my_cmap, extend='both')
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[i]))
    return cont, time_text

# create the animation
anim = animation.FuncAnimation(fig, animate, frames=ntau, repeat=False)

# save the animation to a file
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/temperature.gif", writer=writergif)


'''


# make a 2D meshgrid in the transverse plane
#X, Y = meshgrid(x, y)



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

'''tau_idx = int(ntau*(1/3)) # 0 for the initial condition

fig = plt.figure()
cont = plt.contourf(X, Y, v2[tau_idx, 0, :, :]*wchar2[tau_idx, 0, :, :], levelsVW, cmap=my_cmap, extend='both')
cbar = fig.colorbar(cont)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$y$ (fm)")
plt.text(1.0, 10.0, r'$\tau = {0:3.1f}$ fm'.format(tau_list[tau_idx]))
plt.tight_layout()
plt.savefig(f"{final_plots_folder}/V2W2_Contour_XY-tau_{tau_idx}-of-{ntau}")

'''
'''tau_idx = int(ntau*(18/51)) # 0 for the initial condition

fig = plt.figure()
cont = plt.contourf(X, Y, V2w2_status[tau_idx, 0, :, :], levelscaus, cmap=my_cmap, extend='both')
cbar = fig.colorbar(cont)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$y$ (fm)")
plt.text(1.0, 10.0, r'$\tau = {0:3.1f}$ fm'.format(tau_list[tau_idx]))
plt.xlim([-8, 8])
plt.ylim([-8, 8])
plt.tight_layout()
plt.savefig(f"{final_plots_folder}/V2W2_status-countour-tau_{tau_idx}-of-{ntau}")
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


###################################### contour plot for vw status with contours

'''levels4status = [-0.5, 0.5, 1.5, 2.5,3.5,4.5]
colors4stat = ['green','yellow','red','pink']
total_status_labels4 = ['causal & stable','acausal & stable', 'acausal & unstable','elliptic']
my_cmap_4stat = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['black'] + colors4stat)


X, Y = meshgrid(x, y)

tau_idx = int(ntau*(40/82)) # 0 for the initial condition

fig = plt.figure(figsize=(10,6))
cont = plt.contourf(X, Y, causal_AND_v2w2_status[tau_idx, 0, :, :], 
                    levels = levels4status, 
                    cmap=my_cmap_4stat,
                    extend='both')
#cbar = fig.colorbar(cont)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$y$ (fm)")
plt.text(-7.4, -7, r'$\tau = {0:3.1f}$ fm'.format(tau_list[tau_idx]), color ='white')
legend_patches = [mpatches.Patch(color=colors4stat[i], label = total_status_labels4[i])
                  for i in range(len(colors4stat))]
plt.legend(handles = legend_patches,
            loc='center left', 
            bbox_to_anchor=(1.05,0.5), 
            frameon=False)
plt.xlim([-8, 8])
plt.ylim([-8, 8])
plt.tight_layout()
plt.savefig(f"{final_plots_folder}/full_status-countour-pure-bulk-tau_{tau_idx}-of-{ntau}-GYR-elli")'''


###################################### -- causal and vw animation with contours

'''X, Y = meshgrid(x, y)

# first plot the first frame as a contour plot
fig = plt.figure(figsize=(10,6))
cont = plt.contourf(X, Y, causal_AND_v2w2_status[0, 0, :, :].transpose(), 
                    levels = levels4status, 
                    cmap=my_cmap_4stat, 
                    extend='both')
time_text = plt.text(-7.4, -7, r"$\tau = {0:4.2f}$ fm/c".format(tau_list[0]), color ='white')
#cbar = fig.colorbar(cont, ticks = [0,1,2,3])
#cbar.ax.set_yticklabels(total_status_labels)
legend_patches = [mpatches.Patch(color=colors4stat[i], label = total_status_labels4[i])
                  for i in range(len(colors4stat))]
plt.legend(handles = legend_patches,
            loc='center left', 
            bbox_to_anchor=(1.05,0.5), 
            frameon=False)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$y$ (fm)")
plt.xlim([-8, 8])
plt.ylim([-8, 8])
plt.tight_layout(rect=[0, 0, 1, 1])   

# define animation function to update the contour at every time frame
def animate(i): 
    global cont, time_text
    for c in cont.collections: # collections WILL BE REMOVED SOON from matplotlib
        c.remove()  # removes only the contours, leaves the rest intact
    cont = plt.contourf(X, Y, causal_AND_v2w2_status[i, 0, :, :], levels = levels4status, cmap=my_cmap_4stat, extend='both')
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[i]))
    return cont, time_text

# create the animation
anim = animation.FuncAnimation(fig, animate, frames=ntau, repeat=False)

# save the animation to a file
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/animation_full-status-elli.gif", writer=writergif)'''