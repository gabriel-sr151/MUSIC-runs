import numpy as np
from numpy import *
import sys
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

os.chdir(script_dir)

orig_IC = "./epsilon-u-Hydro-t0.6-0.dat" 
altered_IC = open(f"./coarse_grain-GSR/epsilon-u-Hydro-t0.6-0-coarse-grained-to-256x256.dat","w")
print('# dummy 1 etamax= 1 xmax= 256 ymax= 256 deta= 0 dx= 0.1328125 dy= 0.1328125', file = altered_IC)

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
#print(shape(orig_data_grid))
print(orig_data_grid[0,:,:])

new_data_grid = orig_data_grid[::2,::2,:]
#print(shape(new_data_grid))
#print('altered data \n',new_data_grid[0,:,:])

new_data1 = new_data_grid.reshape(-1,18)

for line in new_data1:

    result = ' '.join(map(str, line))

    print(result, file = altered_IC)
        




sys.exit()

