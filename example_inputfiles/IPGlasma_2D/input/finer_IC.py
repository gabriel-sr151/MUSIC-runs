import numpy as np
from numpy import *
import sys
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

os.chdir(script_dir)

orig_IC = "./epsilon-u-Hydro-t0.6-0.dat" 
altered_IC = open(f"./coarse_grain-GSR/epsilon-u-Hydro-t0.6-0-coarse-grained-to-256x256.dat","w")
print('# dummy 1 etamax= 1 xmax= 1024 ymax= 1024 deta= 0 dx= 0.0332031 dy= 0.033203125', file = altered_IC)


orig_data = []

with open(orig_IC, 'r') as file_d:
   
   file_d.readline() #skipping first line (header)

   for line in file_d:
       
       columns = list(map(float, line.split()))

       if columns != []:

            orig_data.append(columns) 

#print(shape(orig_data))

orig_data1 = np.array(orig_data)
orig_data_grid = orig_data1.reshape((512,512,18))

new_data_grid = zeros([1024,1024,18])
new_data_grid[::2,::2,:] = orig_data_grid #the subsampled grid is enforced to be the original grid


#print(new_data_grid[-2,0,:])
print(type(new_data_grid))

# glossary 
''' 
       tau-tau0 = columns[0]
       x = columns[1]
       y = columns[2]
       e_orig = columns[3]
       u_orig = columns[4:8] #(u^tau, u^x, u^y, u^eta)
       pi^(tautau)_orig, pi^(taux)_orig, pi^(tauy)_orig, pi^(taueta)_orig = columns[8:12]
       pi^(xx)_orig, pi^(xy)_orig, pi^(xeta)_orig = columns[12:15]
       pi^(yy)_orig, pi^(yeta)_orig = columns[15:17]
       pi^(etaeta)_orig = columns[18]
'''

'''
trying to create intermediate points in the new as averages of the original grid is not trivial because
u_mu*u^mu =-1 and pi_mu_nu is transversal and symmetric
'''

tau0 = 0.6
for i in range(512):

    for j in range(512):
         
        v_thres = 0.05
        utau_thres = sqrt(1/(1-v_thres**2))

        if orig_data[i,j,4] < utau_thres: # confirm if this is true in Milne coordinates

            new_data_grid[2*i+1,2*j,:] = orig_data[i,j,:]
            new_data_grid[2*i-1,2*j,:] = orig_data[i,j,:]
            new_data_grid[2*i,2*j-1,:] = orig_data[i,j,:]
            new_data_grid[2*i,2*j-1,:] = orig_data[i,j,:] 

            

         
      
