import numpy as np
import mpmath as mp



def smooth_theta(x,width,hside_lth):
    
   '''
   box initial conditions for MUSIC
  x -- spatial coordinate in fm
  width -- smoothening width
  hside_lth -- half side length of the box  
   '''


   return (1/2)*( (1 + mp.tanh( (x + hside_lth)/width )) - (1 + mp.tanh( (x - hside_lth)/width )))



'''
tst_range = np.linspace(-3.0,3.0,20)

for x in tst_range:

    print(smooth_theta(x,0.1,1.0))
'''

L = 2.0# box side
w = 0.1# width
e0 = 0.02 #maximum energy (GeV)

file_path = open(f'box-IC-data-side-{L}fm-max-ener-{e0}GeV-width-{w}fm.dat',"w")

print('# dummy 1 etamax= 1 xmax= 512 ymax= 512 deta= 0 dx= 0.0664062 dy= 0.0664062', file = file_path)

x_range = np.linspace(-17,17,512)
y_range = np.linspace(-17,17,512)

for x in x_range:

    for y in y_range:

        x = round(x,4)
        y = round(y,4)
        data = e0*smooth_theta(x,w,L/2)*smooth_theta(y,w,L/2)
        data = round(data,4)         
        print('0', x, y, data, '1 0 0 0 0 0 0 0 0 0 0 0 0 0', file=file_path)




