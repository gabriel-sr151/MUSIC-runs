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

working_path = path.join(home, "MUSIC/acausal-instab-2p1-pure_bulk_public")
# if you have a different path, please change it accordingly 
        



# change the following line to your result folder
TestResultFolder = "run_paper-w_PI-th-zovs8-QRVoff"#"run_paper_zovs8_QRVoff" 
                   #"run_paper-w_PI-th-zovs8-QRVoff"


bulk_relax_time_factor = 1./15.
incl_delPIPI = 1.0 # 1.0 for including delPIPI (run_paper-w_PI-th-zovs8-QRVoff), 
                   # 0.0 for not including it (run_paper_zovs8_QRVoff)




# load hydrodynamic evolution data
data = fromfile(path.join(working_path, TestResultFolder,"evolution_all_xyeta.dat"), dtype=float32)



# read header about the grid information
header = data[0:16]


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


wchar2 = zeros([ntau, neta, nx, ny]) #GSR -- characteristic speed for pure bulk simulations
v2 = zeros([ntau, neta, nx, ny]) #GSR -- VW criterion
causality_status = zeros([ntau, neta, nx, ny]) #GSR
V2w2_status = zeros([ntau, neta, nx, ny]) #GSR                            
causal_AND_v2w2_status = zeros([ntau, neta, nx, ny]) #GSR
active_cells = zeros([ntau, neta, nx, ny]) #GSR


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
        
        if ed[itau, eta_idx, x_idx, y_idx] > 0.15: #GeV/fm3 -- see chun's output file

            active_cells[itau, eta_idx, x_idx, y_idx] = 10.0
        
        #end if    

          

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







###################################### contour plot for vw status pcolormesh

levels4status = [-0.5, 0.5, 1.5, 2.5,3.5,4.5]
colors4stat = ['green','yellow','red','purple']
total_status_labels4 = ['good','bad', r'ugly ($vw > 1$)', r'ugly ($w^{2}<0$)']
my_cmap_4stat = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['black'] + colors4stat)

colors4stat_lp = ['green','yellow','red'] #['green','red','yellow','purple']#to change columns of legend_patches
total_status_labels4_lp = ['good','bad',r'ugly ($vw > 1$)']
 #['good',r'ugly ($vw > 1$)','bad', r'ugly ($w^{2}<0$)'] #to change columns of legend_patches

# Choose which frames to plot (e.g., first frame)
#frame_idx_list = [int(ntau*(0/82)),int(ntau*(40/82)), int(ntau*(50/82))]
#frame_idx_list = [int(ntau*(0)),int(ntau*(1/2)), int(ntau*(9/10))]
frame_idx_list = [0, 20, 40]

#fontsize data
GBU_time_text_fs = 22
GBU_legend_fs = 22
GBU_axes_fs = 24
GBU_ticks_fs = 22

Tmap_time_text_fs = 22
Tmap_cbar_fs = 22
Tmap_axes_fs = 24
Tmap_ticks_fs = 22
Tmap_cbar_num_fs = 18
Tmap_status_subreg = 3 # 3 for ugly vw>1
Tmap_color_subreg = 'red'

for tau_idx in frame_idx_list:

    if tau_idx == 0:

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 9))

    else:

        fig, ax = plt.subplots(figsize=(8, 8))    
    #endif    

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
                    color='white', fontsize=GBU_time_text_fs)
    

    # Add colorbar (if desired)
    #cbar = fig.colorbar(plot, ax=ax, ticks=levels4status)
    #cbar.set_label('Status')  # Customize as needed

    # Create legend
    if tau_idx == 0:

        legend_patches = [mpatches.Patch(color=colors4stat_lp[i], label=total_status_labels4_lp[i])
                    for i in range(len(colors4stat_lp))]       
        ax.legend(handles=legend_patches,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.02), # Position above the plot
            ncol = len(colors4stat_lp) ,#len(colors4stat),# Number of columns (equal to number of items for horizontal layout)
            frameon=False,
            fontsize=GBU_legend_fs,
            handletextpad=0.5,# Space between patch and text
            borderaxespad=0.02,# Padding between legend and axes
            columnspacing= 0.3 # Space between columns
            )
    #endif    

    # Axis formatting
    ax.set_xlabel(r"$x$ (fm)", fontsize= GBU_axes_fs)
    ax.set_ylabel(r"$y$ (fm)", fontsize= GBU_axes_fs)
    ax.tick_params(axis='both', labelsize = GBU_ticks_fs)
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    ax.set_aspect('equal')  # Keep aspect ratio square
    if tau_idx == 0:
        
        ax.text(1, 7, 'Initial conditions', color = 'white', fontsize = Tmap_time_text_fs)
        ax.text(6, -7, '(a)', color = 'white', fontsize = Tmap_time_text_fs)

    else:

        if incl_delPIPI < 0.001:

            ax.text(1, 7, 'Minimal IS', color = 'white', fontsize = Tmap_time_text_fs)
            ax.text(6, -7, '(c)', color = 'white', fontsize = Tmap_time_text_fs)

        else:

            ax.text(1, 7, r'IS with $\delta_{\Pi \Pi}$', color = 'white', fontsize = Tmap_time_text_fs)
            ax.text(6, -7, '(e)', color = 'white', fontsize = Tmap_time_text_fs)

   
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
    if tau_idx == 0:

        fig = plt.figure(figsize=(8,9))

    else:

        fig = plt.figure(figsize=(8,8))    
    
    ax = plt.gca() 

    cont = ax.contourf(X, Y, T[tau_idx, 0, :, :], levelsT_finer, cmap=blue_transparent, extend='both')

    if tau_idx == 0:

        cbar = fig.colorbar(cont, 
            ax = ax,                
            location='top',    # Places colorbar above the plot
            orientation='horizontal',  # Ensures horizontal layout
            pad=0.02, # Adds small spacing between plot and colorbar
            shrink = 1.0,
            aspect = 40, # Controls colorbar thickness
            )  
        cbar.set_label('T (GeV)', fontsize = Tmap_cbar_fs, labelpad = Tmap_cbar_num_fs)
        cbar.formatter = FormatStrFormatter('%4.2g')  # 3 significant figures
        cbar.update_normal(cont)  # Ensure updates apply
    #endif    
    

    # 2. Then plot the status boundary where status == Tmap_status_subreg
    status_data = causal_AND_v2w2_status[tau_idx, 0, :, :]
    cont_status = plt.contour(X, Y, (status_data == Tmap_status_subreg).astype(float), 
                            levels=[0.5],  # This will draw the boundary between 0 and 1
                            colors=Tmap_color_subreg,                           
                            linewidths=0.75)
    
    ax.set_xlabel(r"$x$ (fm)", fontsize = Tmap_axes_fs)
    ax.set_ylabel(r"$y$ (fm)", fontsize = Tmap_axes_fs)
    ax.tick_params(axis='both', labelsize = Tmap_ticks_fs)
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    ax.set_aspect('equal')
    ax.text(-7.4, -7, r'$\tau = {0:4.2f}$ fm/c'.format(tau_list[tau_idx]), fontsize = Tmap_time_text_fs)

    if tau_idx == 0:
        
        ax.text(1, 7, 'Initial conditions', fontsize = Tmap_time_text_fs)
        ax.text(6, -7, '(b)', fontsize = Tmap_time_text_fs)

    else:

        if incl_delPIPI < 0.001:

            ax.text(1, 7, 'Minimal IS', fontsize = Tmap_time_text_fs)
            ax.text(6, -7, '(d)', fontsize = Tmap_time_text_fs)

        else:

            ax.text(1, 7, r'IS with $\delta_{\Pi \Pi}$', fontsize = Tmap_time_text_fs)
            ax.text(6, -7, '(f)', fontsize = Tmap_time_text_fs)

    
    plt.tight_layout()
    plt.savefig(f"{final_plots_folder}/temperature_XY-tau_{tau_idx}-of-{ntau}-with-ugly", dpi = 300)

# end for in tau_idx

################ causal and v2w2 status animation ---2 with pcolormesh

levels4status = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
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
anim.save(f"{final_plots_folder}/animation_full-status.gif", writer=writergif, dpi=300)

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
anim.save(f"{final_plots_folder}/temperature-XY.gif", writer=writergif)
'''





