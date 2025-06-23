#neighboor + 1 with periodical boundary conditions. L+1 maps to 1
def x_plus_1(x, L):

    return (x % L) + 1

#neighboor -1  with periodical boundary conditions. 1-1 maps to L
def x_minus_1(x,L):

    return L-((L-(x-1)) % L)


ic_new = open(f"./coarse_grain-GSR/epsilon-u-Hydro-t0.6-0-coarse-grained-512x512-to-256x256.dat","w")
print('# dummy 1 etamax= 1 xmax= 256 ymax= 256 deta= 0 dx= 0.0664062 dy= 0.066406', file = ic_new)

# glossary 
'''
       e_orig = columns[3]
       u_orig = columns[4:8] #(utau, ux, uy, ueta)
       pi_tautau_orig, pi_taux_orig, pi_tauy_orig, pi_taueta_orig = columns[9:13]
       pi_xx_orig, pi_xy_orig, pi_xeta_orig = columns[14:17]
       pi_yy_orig, pi_yeta_orig = columns[17:19]
       pi_etaeta_orig = columns[20]
'''


xmin = -17   
ymin = -17
dx = float(34/512)
dy = float(34/512)






'''def process_ic_file(orig_file):

    # Split the data into lines
    lines = orig_file.strip().split('\n')
    
    # Convert each line to a list of floats
    data = [list(map(float, line.split())) for line in lines[1:]]

    return data[0]
    
    # Find unique x and y coordinates (sorted)
    x_coords = sorted(set(row[1] for row in data))
    y_coords = sorted(set(row[2] for row in data))
    
    # Calculate spacing between points (assuming uniform grid)
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]

    for row in data:

        x = float(row[1])
        y = float(row[1])
'''

'''process_ic_file(orig_IC)
'''      
        

       

    
       
       

           
     