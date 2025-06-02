import pandas as pd
import numpy as np
import sys
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

os.chdir(script_dir)

orig_IC = "./epsilon-u-Hydro-t0.6-0.dat" 


#fact = 2.0 # v_final= relativistic_sum(vx, fact*vx) for local boost
vx = 0.0

dvx = 0.99 # change in vx for global boost
gam_b = 1.0/( np.sqrt(1.0-dvx**(2.0)) ) 


#ic_new = open(f"./epsilon-u-Hydro-t0.6-0-increased-in-x-by-{fact}.dat","w")
ic_new = open(f"./epsilon-u-Hydro-t0.6-0-boosted-global-by-{dvx}.dat","w")
print('# dummy 1 etamax= 1 xmax= 512 ymax= 512 deta= 0 dx= 0.0664062 dy= 0.066406', file = ic_new)


with open(orig_IC, 'r') as file_d:
   
   file_d.readline() #skipping first line

   for line in file_d:
       
       columns = line.split()
      
       u = columns[4:8]
       ui = [float(comp) for comp in u]
       #vx = (ui[1]/ui[0]) for local boost
       
       #dvx = -fact*vx # change in vx for local boost          
       #gam_b = 1.0/( np.sqrt(1.0-dvx**(2.0)) ) #boost Lorentz factor for local boost  

       uf_t = (gam_b)*ui[0] - (gam_b*dvx)*ui[1]
       uf_x = -(gam_b*dvx)*ui[0] + (gam_b)*ui[1]


       line_output = columns
       line_output[4] = round(uf_t,4)
       line_output[5] = round(uf_x,4)
           
        #printing the list as a line in the output file
       result = ' '.join(map(str, line_output))
       print(result, file = ic_new)        




