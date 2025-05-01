#based on https://github.com/JETSCAPE/SummerSchool2020/blob/master/hydro_session/hydro_movie-TestRun.ipynb
import numpy as np
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
levelsV = linspace(0.0, 1.0, 50)
levels2status = [-2, 0, 2]
levels3status = [-15,-5, 5, 15, 25, 35]
levels4status = [-5, 5, 15, 25, 35]


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
colors3stat = ['pink','green','yellow','red']
#colors3stat = ['green','yellow','red']  # for QM talk
colors4stat = ['green','blue','red']
total_status_labels = ['elliptical','causal & stable','acausal & stable', 'acausal & unstable']
which_status_labels = ['sound mode',r'shear $\mathfrak{g}$-mode','shear w-mode']
my_cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)
my_cmap_2stat = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors2stat)
my_cmap_3stat = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', 
                                                             colors3stat[:1] + ['black'] + colors3stat[1:])
#my_cmap_3stat = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', 
#                                                              ['pink','black'] + colors3stat) # for QM talk
my_cmap_4stat = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', 
                                                              ['black'] + colors4stat)


# change the following line to your result folder
TestResultFolder = "acausality-w-shear/run5-QRVoff" 
                                           # >run 1 -- no second order terms, i excluded even the ones that music
                                           #                            doesn't by default -- energy 2x error
                                           # >run 2 -- i reincluded the terms excluded by me in run 1 
                                           # incl_sec_mus = 1.0
                                           # incl_sec = 0.0
                                           # >run 3 -- i included second order terms
                                           # incl_sec_mus = 1.0
                                           # incl_sec = 1.0 -- Reynolds square terms ON
                                           # >run4-both-coups included only delta_pipi, delta_PIPI, lambda_PIpi, 
                                           # lambda_piPI terms all other coeffs excluded
                                           # >run4-no-PI-coup included only delta_pipi, delta_PIPI, (lambda_PIpi = 0), 
                                           # lambda_piPI terms all other coeffs excluded
                                           # >run4-no-sh-coup included only delta_pipi, delta_PIPI, lambda_PIpi, 
                                           # (lambda_piPI = 0) terms all other coeffs excluded
                                           # >run4-taupipi-on-no-lambs included only delta_pipi, delta_PIPI, 
                                           # and tau_pipi all other coeffs excluded
                                           # >run4-taupipi-on-lambs-on included only delta_pipi, delta_PIPI, 
                                           #  lambda_PIpi, lambda_piPI, tau_pipi all other coeffs excluded 
                                           # NO Reynolds square terms
                                           # >run5-QRVoff used music Include_second_order_terms 0 questrevert 
                                           # regulator off

                                           
        
                                           
print('RESULTS BEING ANALYZED:',TestResultFolder)




# load hydrodynamic evolution data
data = fromfile(path.join(working_path, TestResultFolder,"evolution_all_xyeta.dat"), dtype=float32)

#print(data.shape)


# read header about the grid information
header = data[0:16]

#print(header) #ok 
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

#print(ntau, tau0, dtau)

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



v2 = zeros([ntau, neta, nx, ny]) #GSR -- VW criterion
causality_status = zeros([ntau, neta, nx, ny]) #GSR
V2w2_status = zeros([ntau, neta, nx, ny]) #GSR                            
causal_AND_v2w2_status = zeros([ntau, neta, nx, ny]) #GSR
wchar2_min = zeros([ntau, neta, nx, ny]) #GSR
wchar2_max = zeros([ntau, neta, nx, ny]) #GSR
wchar2_min_which = zeros([ntau, neta, nx, ny]) #GSR
wchar2_max_which = zeros([ntau, neta, nx, ny]) #GSR


frac_ugly = zeros(ntau)
frac_bad = zeros(ntau)
frac_good = zeros(ntau)
frac_elli = zeros(ntau)
N_active = zeros(ntau) # number of nonzero v2w2 status

frac_elli_from_sound = zeros(ntau)
frac_elli_from_g = zeros(ntau)
frac_elli_from_shear = zeros(ntau)

frac_acaus_from_sound = zeros(ntau)
frac_acaus_from_g = zeros(ntau)
frac_acaus_from_shear = zeros(ntau)
frac_acaus = zeros(ntau)

for itau in range(ntau):

    idx = (abs(data[:, 0] - itau) < 0.1)
    data_cut = data[idx, :]
    frac_ugly[itau] = 0.0
    frac_bad[itau] = 0.0
    frac_good[itau] = 0.0
    frac_elli[itau] = 0.0
    N_active[itau] = 0.0
    frac_acaus[itau] = 0.0

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
        pixx_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 11]# shear_tensor/(e+p) in the LRF
        pixy_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 12]
        pixz_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 13]
        piyy_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 14]
        piyz_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 15]


        bulkPI_norm[itau, eta_idx, x_idx, y_idx] = data_cut[igrid, 16] # Pi/(e+p)

        v2[itau, eta_idx, x_idx, y_idx] = vx[itau, eta_idx, x_idx, y_idx]**2 + vy[itau, eta_idx, x_idx, y_idx]**2 \
                                       + vz[itau, eta_idx, x_idx, y_idx]**2
        
        ##################
        #   CHARACTERISTIC SPEEDS COMPUTATION BEGINS HERE----------------------------------------
        #   SOURCE -- arxiv: 2005.11632  
        ###################

        #pimunu-eigenvalues

        pizz_norm = - pixx_norm[itau, eta_idx, x_idx, y_idx] - piyy_norm[itau, eta_idx, x_idx, y_idx]

        pi_norm_matrix = np.array([[pixx_norm[itau, eta_idx, x_idx, y_idx], pixy_norm[itau, eta_idx, x_idx, y_idx], pixz_norm[itau, eta_idx, x_idx, y_idx]],
                                   [pixy_norm[itau, eta_idx, x_idx, y_idx], piyy_norm[itau, eta_idx, x_idx, y_idx], piyz_norm[itau, eta_idx, x_idx, y_idx]],
                                   [pixz_norm[itau, eta_idx, x_idx, y_idx], piyz_norm[itau, eta_idx, x_idx, y_idx], pizz_norm]])
       
        eigv_pi_norm = np.linalg.eigh(pi_norm_matrix)[0] # since the matrix is symmmetric numpy has a more efficient method

        #first order to rlx time ratios

        bulk_relax_time_factor = 1.0/15.0 #MUSIC_default 1/14.55
        eta_OV_tauPI_ed_PL_pr = 1.0/5.0 # (eta/[tau_BULK*(e+p)])
        zeta_OV_tauPI_ed_PL_pr = (1.0/bulk_relax_time_factor)\
                               *( (1.0/3.0 - cs2[itau, eta_idx, x_idx, y_idx])**(2.0) ) # (zeta/[tau_BULK*(e+p)])

        
        #second order terms
        incl_sec_mus = 1.0 #include second order terms; excluded by 'Include_second_order_terms = 0' by default
        incl_sec = 0.0 #include second order terms; included by 'Include_second_order_terms = 0' by default
        incl_lamb_PI_shear = 0.0 #include shear coupling in bulk eom; make sure incl_sec = 1.0
        incl_lamb_pi_PI = 0.0 #include bulk_PI coupling in shear eom; make sure incl_sec = 1.0
        incl_tau_pipi = 0.0 #include tau_pipi

        #second order terms in bulk eom

        delPIPI_OV_tauPI = incl_sec_mus*(2.0/3.0)
        lamb_PI_pi_OV_tau_PI = (8.0/5.0)*(1.0/3.0 - cs2[itau, eta_idx, x_idx, y_idx])
        lamb_PI_pi_OV_tau_PI = incl_lamb_PI_shear*incl_sec*lamb_PI_pi_OV_tau_PI

        #second order terms in shear eoms

        lamb_pi_PI_OV_tau_pi = incl_lamb_pi_PI*incl_sec*(6.0/5.0)
        delpipi_OV_taupi = incl_sec_mus*(4.0/3.0)
        taupipi_OV_taupi = incl_tau_pipi*incl_sec*(10.0/7.0)
 
        #characteristic speeds begin

        num_char_w = 3 #"number" of characteristic velocities
        wchar2_sound = zeros([num_char_w]) #GSR -- characteristic speed
        wchar2_shear_g = zeros([num_char_w]) #GSR -- characteristic speed
        wchar2_shear_w = zeros([num_char_w, num_char_w]) #GSR -- characteristic speed

        wchar2_list = []

        for a in range(num_char_w):

            wchar2_sound[a] = cs2[itau, eta_idx, x_idx, y_idx] \
              +  1.0/(1.0 + bulkPI_norm[itau, eta_idx, x_idx, y_idx] + eigv_pi_norm[a])\
              *(zeta_OV_tauPI_ed_PL_pr + delPIPI_OV_tauPI*bulkPI_norm[itau, eta_idx, x_idx, y_idx]
                + lamb_PI_pi_OV_tau_PI*eigv_pi_norm[a]
                + (1.0/3.0)*eta_OV_tauPI_ed_PL_pr + (1.0/6.0)*lamb_pi_PI_OV_tau_pi + delpipi_OV_taupi*eigv_pi_norm[a]\
                - (1.0/6.0)*taupipi_OV_taupi*eigv_pi_norm[a]\
                + eta_OV_tauPI_ed_PL_pr + (1.0/2.0)*lamb_pi_PI_OV_tau_pi*bulkPI_norm[itau, eta_idx, x_idx, y_idx]
                + (1.0/2.0)*taupipi_OV_taupi*eigv_pi_norm[a]  )
            
            wchar2_list.append(wchar2_sound[a])
            
            wchar2_shear_g[a] = (4*eta_OV_tauPI_ed_PL_pr \
                + 2*lamb_pi_PI_OV_tau_pi*bulkPI_norm[itau, eta_idx, x_idx, y_idx] + taupipi_OV_taupi*eigv_pi_norm[a])\
                /( 4*(1.0 + bulkPI_norm[itau, eta_idx, x_idx, y_idx]) )
            
            wchar2_list.append(wchar2_shear_g[a])         


            for b in range(num_char_w):

                if b != a:

                     wchar2_shear_w[a, b] = ( eta_OV_tauPI_ed_PL_pr \
                        + (1/2)*lamb_pi_PI_OV_tau_pi*bulkPI_norm[itau, eta_idx, x_idx, y_idx]\
                        + (1/4)*taupipi_OV_taupi*(eigv_pi_norm[a] + eigv_pi_norm[b])) \
                        /(1.0 + bulkPI_norm[itau, eta_idx, x_idx, y_idx] + eigv_pi_norm[a])  

                     wchar2_list.append(wchar2_shear_w[a, b]) 

            #end for

        #end for


        #characteristic speeds end

        ###################  EXTREMA of CHARACTERISTIC SPEEDS

        wchar2_min[itau, eta_idx, x_idx, y_idx] = min(wchar2_list)
        wchar2_max[itau, eta_idx, x_idx, y_idx] = max(wchar2_list)

       # print(wchar2_min[itau, eta_idx, x_idx, y_idx], wchar2_max[itau, eta_idx, x_idx, y_idx])

       ###################  WHICH OF THE MODES ARE DO THE EXTREMUM CHARACTERISTIC SPEEDS BELONG TO?

       #MAX

        if (wchar2_list.index(wchar2_max[itau, eta_idx, x_idx, y_idx]) % 4) == 0:

            wchar2_max_which[itau, eta_idx, x_idx, y_idx] = 10 #sound mode is min if index is 0,4,8  

        else:

            if (wchar2_list.index(wchar2_max[itau, eta_idx, x_idx, y_idx]) % 4) == 1:  

                wchar2_max_which[itau, eta_idx, x_idx, y_idx] = 20 #g mode is min if index is 1,5,9

            else: 

                wchar2_max_which[itau, eta_idx, x_idx, y_idx] = 30 # shear mode is min if index is 2,3,6,7,10,11

        '''if wchar2_list.index(wchar2_max[itau, eta_idx, x_idx, y_idx]) <= 2: 

            wchar2_max_which[itau, eta_idx, x_idx, y_idx] = 10 #sound mode is max if index is 0,1,2           

        else:

            if wchar2_list.index(wchar2_max[itau, eta_idx, x_idx, y_idx])<=5: 

                wchar2_max_which[itau, eta_idx, x_idx, y_idx] = 20 #g mode is max if index is 3,4,5

            else: 

                 wchar2_max_which[itau, eta_idx, x_idx, y_idx] = 30 # shear mode is max if index is >5'''

       #MIN
        
        if (wchar2_list.index(wchar2_min[itau, eta_idx, x_idx, y_idx]) % 4) == 0:

            wchar2_min_which[itau, eta_idx, x_idx, y_idx] = 10 #sound mode is min if index is 0,4,8  

        else:

            if (wchar2_list.index(wchar2_min[itau, eta_idx, x_idx, y_idx]) % 4) == 1:  

                wchar2_min_which[itau, eta_idx, x_idx, y_idx] = 20 #g mode is min if index is 1,5,9

            else: 

                wchar2_min_which[itau, eta_idx, x_idx, y_idx] = 30 # shear mode is min if index is 2,3,6,7,10,11                                                    


        '''if wchar2_list.index(wchar2_min[itau, eta_idx, x_idx, y_idx]) <= 2: 

            wchar2_min_which[itau, eta_idx, x_idx, y_idx] = 10 #sound mode is min if index is 0,1,2           

        else:

            if wchar2_list.index(wchar2_min[itau, eta_idx, x_idx, y_idx])<=5: #g mode is min if index is 3,4,5 

                wchar2_min_which[itau, eta_idx, x_idx, y_idx] = 20

            else: 

                 wchar2_min_which[itau, eta_idx, x_idx, y_idx] = 30 # shear mode is min if index is >5''' 
        
        '''print(wchar2_min[itau, eta_idx, x_idx, y_idx], wchar2_list.index(wchar2_min[itau, eta_idx, x_idx, y_idx]),\
               wchar2_min_which[itau, eta_idx, x_idx, y_idx])
        print(wchar2_max[itau, eta_idx, x_idx, y_idx], wchar2_list.index(wchar2_max[itau, eta_idx, x_idx, y_idx]),\
               wchar2_max_which[itau, eta_idx, x_idx, y_idx])
        '''
        #sys.exit()

        ##################
        #   NECESSARY CAUSALITY CONDITIONS AND VW CRITERION BEGINS HERE ---------------------------------------- 
        ###################

        if ( (wchar2_max[itau, eta_idx, x_idx, y_idx] < 1.0) and (wchar2_min[itau, eta_idx, x_idx, y_idx] > 0.0) ):
           
           causal_AND_v2w2_status[itau, eta_idx, x_idx, y_idx] = 10.0

           frac_good[itau] += 1.0
           N_active[itau] += 1.0

        else: 

            #elliptical cells begin
            if (wchar2_min[itau, eta_idx, x_idx, y_idx] <= 0.0):

                causal_AND_v2w2_status[itau, eta_idx, x_idx, y_idx] = -10.0 # CHANGE TO -10 WHEN SEPARATING 
                                                                         # ELLIPTICAL CELLS to 30 when
                                                                         # merging with the ugly cells
                frac_elli[itau] += 1.0
                N_active[itau]+= 1.0

                #### WHAT MODE IS ELLIPTIC? -BEGIN

                if (wchar2_list.index(wchar2_min[itau, eta_idx, x_idx, y_idx]) % 4) == 0:

                    frac_elli_from_sound[itau] += 1.0 #sound mode is min if index is 0,4,8  

                else:

                    if (wchar2_list.index(wchar2_min[itau, eta_idx, x_idx, y_idx]) % 4) == 1:  

                        frac_elli_from_g[itau] += 1.0 #g mode is min if index is 1,5,9

                    else: 

                        frac_elli_from_shear[itau] += 1.0 # shear mode is min if index is 2,3,6,7,10,11


                 #### WHAT MODE IS ELLIPTIC? - END   

            #elliptical cells end
            else:

                frac_acaus[itau] += 1.0

                #vw criterion -begin
                if wchar2_max[itau, eta_idx, x_idx, y_idx]*v2[itau, eta_idx, x_idx, y_idx] < 1.0:
                    
                    causal_AND_v2w2_status[itau, eta_idx, x_idx, y_idx] = 20.0

                    frac_bad[itau] += 1.0
                    N_active[itau] += 1.0

                else:

                    causal_AND_v2w2_status[itau, eta_idx, x_idx, y_idx] = 30.0 

                    frac_ugly[itau] += 1.0 
                    N_active[itau] += 1.0 
                
                #vw criterion -end
                #### WHAT MODE IS BAD OR UGLY? -BEGIN

                if (wchar2_list.index(wchar2_max[itau, eta_idx, x_idx, y_idx]) % 4) == 0:

                    frac_acaus_from_sound[itau] += 1.0 #sound mode is max if index is 0,4,8  

                else:

                    if (wchar2_list.index(wchar2_max[itau, eta_idx, x_idx, y_idx]) % 4) == 1:  

                        frac_acaus_from_g[itau] += 1.0 #g mode is max if index is 1,5,9

                    else: 

                        frac_acaus_from_shear[itau] += 1.0 # shear mode is max if index is 2,3,6,7,10,11


                 #### WHAT MODE IS BAD OR UGLY? - END                    
                
                #end if
            #end if
        #end if
    #end for
    frac_elli_from_shear[itau] = frac_elli_from_shear[itau]/frac_elli[itau]
    frac_elli_from_g[itau] = frac_elli_from_g[itau]/frac_elli[itau]
    frac_elli_from_sound[itau] = frac_elli_from_sound[itau]/frac_elli[itau]

    frac_acaus_from_shear[itau] = frac_acaus_from_shear[itau]/frac_acaus[itau]
    frac_acaus_from_g[itau] = frac_acaus_from_g[itau]/frac_acaus[itau]
    frac_acaus_from_sound[itau] = frac_acaus_from_sound[itau]/frac_acaus[itau]

    frac_ugly[itau] = frac_ugly[itau]/N_active[itau]
    frac_bad[itau] = frac_bad[itau]/N_active[itau]
    frac_good[itau] = frac_good[itau]/N_active[itau]
    frac_elli[itau] = frac_elli[itau]/N_active[itau]


#end for    






#sys.exit()



# print out some useful information about the evolution file
print("Read in data completed.")


print("nx = {0}, x_min = {1:.2f} fm, x_max = {2:.2f} fm, dx = {3:.2f} fm".format(nx, x[0], x[-1], dx))
print("ny = {0}, y_min = {1:.2f} fm, y_max = {2:.2f} fm, dy = {3:.2f} fm".format(ny, y[0], y[-1], dy))
print("neta = {0}, eta_min = {1:.2f} fm, eta_max = {2:.2f} fm, deta = {3:.2f}".format(neta, eta[0], eta[-1], deta))


final_plots_folder = path.join(working_path, TestResultFolder)

######################################---PLOTS----########################################################

############## fractions of each mode among the acausal cells

fig = plt.figure()

frac_acaus_sum = frac_acaus_from_sound + frac_acaus_from_shear + frac_acaus_from_g

plt.plot(tau_list, frac_acaus_from_sound, label = 'sound mode', color = 'green')
plt.plot(tau_list, frac_acaus_from_shear, label = 'shear-w mode', color = 'blue')
plt.plot(tau_list, frac_acaus_from_g, label = r'shear $\mathfrak{g}$ mode', color = 'red')
plt.plot(tau_list, frac_acaus_sum, ':' , label = 'sum', color = 'black')
plt.xlabel(r"$\tau (fm)$")
plt.ylabel("fraction among acausal")
plt.tight_layout()
plt.legend()
plt.savefig(f"{final_plots_folder}/fracs-acausal-modes")

############## fractions of each mode among the elliptical cells

fig = plt.figure()

frac_elli_sum = frac_elli_from_sound + frac_elli_from_shear + frac_elli_from_g

plt.plot(tau_list, frac_elli_from_sound, label = 'sound mode', color = 'green')
plt.plot(tau_list, frac_elli_from_shear, label = 'shear-w mode', color = 'blue')
plt.plot(tau_list, frac_elli_from_g, label = r'shear $\mathfrak{g}$ mode', color = 'red')
plt.plot(tau_list, frac_elli_sum, '--' , label = 'sum', color = 'black')
plt.xlabel(r"$\tau (fm)$")
plt.ylabel("fraction among elliptical")
plt.tight_layout()
plt.legend()
plt.savefig(f"{final_plots_folder}/fracs-elli-modes")


############# animation for which mode is wchar2min

'''X, Y = meshgrid(x, y)

# first plot the first frame as a contour plot
fig = plt.figure(figsize=(10,6))
cont = plt.contourf(X, Y, wchar2_min_which[0, 0, :, :].transpose(), 
                    levels = levels4status, 
                    cmap=my_cmap_4stat, 
                    extend='both')
time_text = plt.text(-7.4, -7, r"$\tau = {0:4.2f}$ fm/c".format(tau_list[0]), color ='white')
legend_patches = [mpatches.Patch(color=colors4stat[i], label = which_status_labels[i])
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
    cont = plt.contourf(X, Y, wchar2_min_which[i, 0, :, :],\
                         levels = levels4status, cmap=my_cmap_4stat, extend='both')
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[i]))
    return cont, time_text

# create the animation
anim = animation.FuncAnimation(fig, animate, frames=ntau, repeat=False)

# save the animation to a file
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/animation_which-min-status.gif", writer=writergif)

'''




############# animation for which mode is wchar2max

'''X, Y = meshgrid(x, y)

# first plot the first frame as a contour plot
fig = plt.figure(figsize=(10,6))
cont = plt.contourf(X, Y, wchar2_max_which[0, 0, :, :].transpose(), 
                    levels = levels4status, 
                    cmap=my_cmap_4stat, 
                    extend='both')
time_text = plt.text(-7.4, -7, r"$\tau = {0:4.2f}$ fm/c".format(tau_list[0]), color ='white')
legend_patches = [mpatches.Patch(color=colors4stat[i], label = which_status_labels[i])
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
    cont = plt.contourf(X, Y, wchar2_max_which[i, 0, :, :],\
                         levels = levels4status, cmap=my_cmap_4stat, extend='both')
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[i]))
    return cont, time_text

# create the animation
anim = animation.FuncAnimation(fig, animate, frames=ntau, repeat=False)

# save the animation to a file
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/animation_which-max-status.gif", writer=writergif)

'''
############## fractions good, bad, ugly

fig = plt.figure()

frac_sum = frac_good + frac_bad + frac_ugly + frac_elli

plt.plot(tau_list, frac_good, label = 'good', color = 'green')
plt.plot(tau_list, frac_bad, label = 'bad', color = 'yellow')
plt.plot(tau_list, frac_ugly, label = 'ugly', color = 'red')
plt.plot(tau_list, frac_elli, label = 'elliptical', color = 'pink')
plt.plot(tau_list, frac_sum, '--' , label = 'sum', color = 'black')
plt.xlabel(r"$\tau (fm)$")
plt.ylabel("fractions")
plt.tight_layout()
plt.legend()
plt.savefig(f"{final_plots_folder}/fracs-GoodBadUglyElli")

#sys.exit()

################ contour plot of a frame of full_status

'''X, Y = meshgrid(x, y)

tau_idx = 0 # 0 for the initial condition

fig = plt.figure(figsize=(10,6))
cont = plt.contourf(X, Y, causal_AND_v2w2_status[tau_idx, 0, :, :], 
                    levels = levels3status, 
                    cmap=my_cmap_3stat,
                    extend='both')
#cbar = fig.colorbar(cont)
plt.xlabel(r"$x$ (fm)")
plt.ylabel(r"$y$ (fm)")
plt.text(-7.4, -7, r'$\tau = {0:3.1f}$ fm'.format(tau_list[tau_idx]), color ='white')
legend_patches = [mpatches.Patch(color=colors3stat[i], label = total_status_labels[i])
                  for i in range(len(colors3stat))]
plt.legend(handles = legend_patches,
            loc='center left', 
            bbox_to_anchor=(1.05,0.5), 
            frameon=False)
plt.xlim([-8, 8])
plt.ylim([-8, 8])
plt.tight_layout()
plt.savefig(f"{final_plots_folder}/full_status-countour-w-shear-all-second-order-terms-tau_{tau_idx}-of-{ntau}-GYR")
'''


################ causal and v2w2 status

# make a 2D meshgrid in the transverse plane
X, Y = meshgrid(x, y)

# first plot the first frame as a contour plot
fig = plt.figure(figsize=(10,6))
cont = plt.contourf(X, Y, causal_AND_v2w2_status[0, 0, :, :].transpose(), 
                    levels = levels3status, 
                    cmap=my_cmap_3stat, 
                    extend='both')
time_text = plt.text(-7.4, -7, r"$\tau = {0:4.2f}$ fm/c".format(tau_list[0]), color ='white')
legend_patches = [mpatches.Patch(color=colors3stat[i], label = total_status_labels[i])
                  for i in range(len(colors3stat))]
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
    cont = plt.contourf(X, Y, causal_AND_v2w2_status[i, 0, :, :],\
                         levels = levels3status, cmap=my_cmap_3stat, extend='both')
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[i]))
    return cont, time_text

# create the animation
anim = animation.FuncAnimation(fig, animate, frames=ntau, repeat=False)

# save the animation to a file
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/animation_full-status-w-elli.gif", writer=writergif)



############### v2w2 animation

'''X, Y = meshgrid(x, y)

# first plot the first frame as a contour plot
fig = plt.figure()
cont = plt.contourf(X, Y, v2[0, 0, :, :]*wchar2_max[0, 0, :, :], levelsV,
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
    cont = plt.contourf(X, Y, v2[i, 0, :, :]*wchar2_max[i, 0, :, :], levelsV, cmap=my_cmap, extend='both')
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[i]))
    return cont, time_text

# create the animation
anim = animation.FuncAnimation(fig, animate, frames=ntau, repeat=False)

# save the animation to a file
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/animation-v2w2.gif", writer=writergif)
'''

############### fluid velocity animation

'''X, Y = meshgrid(x, y)

# first plot the first frame as a contour plot
fig = plt.figure()
cont = plt.contourf(X, Y, v2[0, 0, :, :], levelsV,
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
    cont = plt.contourf(X, Y, v2[i, 0, :, :], levelsV, cmap=my_cmap, extend='both')
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[i]))
    return cont, time_text

# create the animation
anim = animation.FuncAnimation(fig, animate, frames=ntau, repeat=False)

# save the animation to a file
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/animation-v2.gif", writer=writergif)
'''


############### minimum propagation speeds

'''X, Y = meshgrid(x, y)

# first plot the first frame as a contour plot
fig = plt.figure()
cont = plt.contourf(X, Y, wchar2_min[0, 0, :, :], levelsV,
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
    cont = plt.contourf(X, Y, wchar2_min[i, 0, :, :], levelsV, cmap=my_cmap, extend='both')
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[i]))
    return cont, time_text

# create the animation
anim = animation.FuncAnimation(fig, animate, frames=ntau, repeat=False)

# save the animation to a file
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/animation-w2char-min.gif", writer=writergif)
'''


################ maximum propagation speeds

'''X, Y = meshgrid(x, y)

# first plot the first frame as a contour plot
fig = plt.figure()
cont = plt.contourf(X, Y, wchar2_max[0, 0, :, :], levelsV,
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
    cont = plt.contourf(X, Y, wchar2_max[i, 0, :, :], levelsV, cmap=my_cmap, extend='both')
    time_text.set_text(r"$\tau = {0:4.2f}$ fm/c".format(tau_list[i]))
    return cont, time_text

# create the animation
anim = animation.FuncAnimation(fig, animate, frames=ntau, repeat=False)

# save the animation to a file
writergif = animation.PillowWriter(fps=10)
anim.save(f"{final_plots_folder}/animation-w2char-max.gif", writer=writergif)

'''
