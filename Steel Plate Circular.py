import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as colors
import math

#Input plate dimensions


#Input bolt number


d_hole = 0.75*25.4 # hole diameter in milimeters

edge_x = 2*25.4 # edge distance in milimeters
edge_y = 2*25.4 # edge distance in milimeters

mesh_size = 2 # Finite element size in milimiters

n_bolts = 6

x_bolt = np.zeros(n_bolts)
y_bolt = np.zeros(n_bolts)


x_bolt = np.zeros(n_bolts)
y_bolt = np.zeros(n_bolts)


s_hor = 3*25.4
s_ver = 5*25.4

b = 15*25.4
h = 15*25.4


# for j in range(n_row):
#     for i in range(n_col):
#         x_bolt[i+j*n_col] = edge_x + i*s_hor
        
# for j in range(n_row):        
#     for i in range(n_col):
#         y_bolt[i+j*n_col] = edge_y + j*s_ver


plate_radius = 200
bolt_radius = 200-51

for i in range(n_bolts):
    angle = 2*math.pi/n_bolts
    x_bolt[i] = bolt_radius*math.cos(angle*i)
    y_bolt[i] = bolt_radius*math.sin(angle*i)


fy = 620
E = 200000
Nk = 0 # Axial load in N
Myk = 54.23*10**6 # Mx in N*mm
Mxk = 101.68*10**6 # My in N*mm

fig, ax = plt.subplots()


ax.plot()

circle0 = plt.Circle((0,0),radius=plate_radius,fill=False)
ax.add_patch(circle0)

for i in range(n_bolts):
    circle = plt.Circle((x_bolt[i],y_bolt[i]),radius=d_hole*0.5,fill=False)
    ax.add_patch(circle)
    plt.text(x_bolt[i]+d_hole*0.5,y_bolt[i]+d_hole*0.5,i+1)
    
plt.xlim([-2*plate_radius,2*plate_radius])
plt.ylim([-2*plate_radius,2*plate_radius])
plt.axis('equal')
plt.show()




x_size = int(2*plate_radius/mesh_size)
y_size = int(2*plate_radius/mesh_size)



x = np.linspace(-plate_radius,plate_radius,x_size)
y = np.linspace(-plate_radius,plate_radius,y_size)

Ui = np.zeros((x_size,y_size))
Vi = np.zeros((x_size,y_size))
Xi = np.zeros((x_size,y_size))
Yi = np.zeros((x_size,y_size))

for i in range(x_size):
    for j in range(y_size):
        Ui[i,j] = -mesh_size/2 - plate_radius/2 + i*mesh_size
        Vi[i,j] = -mesh_size/2 - plate_radius/2 + j*mesh_size

ST = np.zeros((x_size,y_size))





x_size = int(2*plate_radius/mesh_size)
y_size = int(2*plate_radius/mesh_size)




        
xg = 0
yg = 0
a = 0

for i in range(x_size):
    for j in range(y_size):
        xg += mesh_size**2*Ui[i,j]
        yg += mesh_size**2*Vi[i,j]
        a += mesh_size**2

xg = xg/a
yg = yg/a

for i in range(x_size):
    for j in range(y_size):
        Xi[i,j] = Ui[i,j] - xg
        Yi[i,j] = Vi[i,j] - yg

for i in range(x_size):
    for j in range(y_size):
        distance = ((Xi[i,j])**2+(Yi[i,j])**2)**0.5
        if(distance <= plate_radius):
            ST[i,j] = 1
for k in range(n_bolts):
    for i in range(x_size):
        for j in range(y_size):
            distance = ((Xi[i,j]-x_bolt[k])**2+(Yi[i,j]-y_bolt[k])**2)**0.5
            if(distance <= 0.5*d_hole):
                ST[i,j] = 2
EPS0 = 0
k1 = 0
k2 = 0

EPSi = np.zeros((x_size,y_size))
Di = np.zeros((x_size,y_size))
SIGi = np.zeros((x_size,y_size))

for i in range(x_size):
    for j in range(y_size):
        EPSi[i,j] = EPS0 + k1*Yi[i,j] - k2*Xi[i,j]
        
        
for i in range(x_size):
    for j in range(y_size):
        
        if(ST[i,j] == 1):
            if(EPSi[i,j] < -fy/E):
                Di[i,j] = 0
            if(EPSi[i,j] <= 0):
                Di[i,j] = E
            
            if(EPSi[i,j] > 0):
                Di[i,j] = 0
            
        if(ST[i,j] == 2):
            if(EPSi[i,j] < 0):
                Di[i,j] = 0
            if(EPSi[i,j] <= fy/E):
                Di[i,j] = E
            
            if(EPSi[i,j] > fy/E):
                Di[i,j] = 0
        

        if(ST[i,j] == 0):
            Di[i,j] = 0
                
        SIGi[i,j] = Di[i,j]*EPSi[i,j]
        

K = np.zeros((3,3))

for i in range(x_size):
    for j in range(y_size):
        K[0,0] += Di[i,j]*mesh_size**2
        K[0,1] += Di[i,j]*mesh_size**2*Yi[i,j]
        K[0,2] += Di[i,j]*mesh_size**2*Xi[i,j]
        K[1,1] += Di[i,j]*mesh_size**2*Yi[i,j]*Yi[i,j]
        K[1,2] += Di[i,j]*mesh_size**2*Xi[i,j]*Yi[i,j]
        K[2,2] += Di[i,j]*mesh_size**2*Xi[i,j]*Xi[i,j]
        
K[2,0] = - K[0,2]
K[2,1] = - K[2,1]
K[1,0] = - K[0,1]
K[2,0] = - K[0,2]
K[2,1] = - K[1,2]
        
F = np.zeros((3,1))

for i in range(x_size):
    for j in range(y_size):
        F[0,0] += SIGi[i,j]*mesh_size**2
        F[1,0] += SIGi[i,j]*mesh_size**2*Yi[i,j]
        F[2,0] += SIGi[i,j]*mesh_size**2*Xi[i,j]
        
F[0,0] += - Nk
F[1,0] += - Mxk
F[2,0] = -F[2,0] - Myk



delta = np.zeros((3,1))

delta[0,0] = EPS0
delta[1,0] = k1
delta[2,0] = k2     

deltaNew = np.zeros((3,1))

deltaNew = delta - np.matmul(np.linalg.inv(K),F)

EPS0 = deltaNew[0,0]
k1 = deltaNew[1,0]
k2 = deltaNew[2,0]

X, Y = np.meshgrid(x, y)
plt.pcolor(X,Y,SIGi.transpose(),cmap='RdBu');
cb = plt.colorbar(extend='neither');
cb.set_label('Stress Contour (MPa)', rotation=270)
plt.axis('square');
plt.axis('off');
plt.title('Stress Contour');
plt.show()


# EPSi = np.zeros((600,600))
# Di = np.zeros((600,600))
# SIGi = np.zeros((600,600))

#%%
k = 0

for k in range(100):
        
    Fold = np.zeros((3,3))
    Fold = F
    
    for i in range(x_size):
        for j in range(y_size):
            EPSi[i,j] = EPS0 + k1*Yi[i,j] - k2*Xi[i,j]
       

            if(ST[i,j] == 1):
                if(EPSi[i,j] < -fy/E):
                    Di[i,j] = 0
                if(EPSi[i,j] <= 0):
                    Di[i,j] = E
                
                if(EPSi[i,j] > 0):
                    Di[i,j] = 0
                
            if(ST[i,j] == 2):
                if(EPSi[i,j] < 0):
                    Di[i,j] = 0
                if(EPSi[i,j] <= fy/E):
                    Di[i,j] = E
                
                if(EPSi[i,j] > fy/E):
                    Di[i,j] = 0

            if(ST[i,j] == 0):
                Di[i,j] = 0
                
            SIGi[i,j] = Di[i,j]*EPSi[i,j]
            
    # plt.imshow(SIGi,interpolation='bilinear')
    # plt.axis('equal')
    # plt.show()
    
    K = np.zeros((3,3))
    
    for i in range(x_size):
        for j in range(y_size):
            K[0,0] += Di[i,j]*mesh_size**2
            K[0,1] += Di[i,j]*mesh_size**2*Yi[i,j]
            K[0,2] += Di[i,j]*mesh_size**2*Xi[i,j]
            K[1,1] += Di[i,j]*mesh_size**2*Yi[i,j]*Yi[i,j]
            K[1,2] += Di[i,j]*mesh_size**2*Xi[i,j]*Yi[i,j]
            K[2,2] += Di[i,j]*mesh_size**2*Xi[i,j]*Xi[i,j]
            
    K[2,0] = - K[0,2]
    K[2,1] = - K[2,1]
    K[1,0] = - K[0,1]
    K[2,0] = - K[0,2]
    K[2,1] = - K[1,2]
            

    F = np.zeros((3,1))
    
    for i in range(x_size):
        for j in range(y_size):
            F[0,0] += SIGi[i,j]*mesh_size**2
            F[1,0] += SIGi[i,j]*mesh_size**2*Yi[i,j]
            F[2,0] += SIGi[i,j]*mesh_size**2*Xi[i,j]
            
    F[0,0] += - Nk
    F[1,0] += - Mxk
    F[2,0] = -F[2,0] - Myk
    
    
    delta = np.zeros((3,1))
    
    delta[0,0] = EPS0
    delta[1,0] = k1
    delta[2,0] = k2     
    
    deltaNew = np.zeros((3,1))
    
    if(np.linalg.det(K)==0):
        print("Equilibrium not satisfied for geometry/load")
        break
    
    deltaNew = delta - np.matmul(np.linalg.inv(K),F)
    
    EPS0 = deltaNew[0,0]
    k1 = deltaNew[1,0]
    k2 = deltaNew[2,0]
    
    if (np.abs(F[1,0])<0.1*10**7) and (np.abs(F[2,0])<0.1*10**7):
        break
        
    else:
        if k == 99:
            print("Equilibrium not satisfied for geometry/load")

    print(k)
        
#%%

# plt.imshow(SIGi,interpolation='bilinear')
# plt.axis('equal')
# plt.show()


vmin = np.min(SIGi)
vmax = np.max(SIGi)
X, Y = np.meshgrid(x, y)

maxabs = int(np.max([np.abs(vmin),np.abs(vmax)]))
plt.figure();
if vmin<0:
    norm = TwoSlopeNorm(vmin = vmin, vcenter = 0,  vmax = vmax)
plt.pcolor(X,Y,SIGi.transpose(),cmap='RdBu',norm=norm);

for i in range(n_bolts):
    x_value = int((plate_radius + x_bolt[i])/mesh_size)
    y_value = int((plate_radius + y_bolt[i])/mesh_size)
    value = np.round(SIGi[x_value,y_value]*math.pi*d_hole**2/4*10**-3,1)
    valuetext = "%3.1f kN" % value
    plt.text(x_bolt[i]+d_hole*0.5,y_bolt[i]+d_hole*0.5,valuetext)
    

cb = plt.colorbar(extend='neither');
cb.set_label('Stress Contour (MPa)', rotation=270)
plt.axis('square');
plt.axis('off');
plt.title('Stress Contour');
plt.show()

# for i in range(n_bolts):
#     value = np.round(SIGi[int(x_bolt[i]/mesh_size),int(y_bolt[i]/mesh_size)]*math.pi*d_hole**2/4*10**-3/4.448,1)
#     print("Bolt %d force (%3.01f,%3.1f): %3.2f kN" % (i+1,x_bolt[i],y_bolt[i],value))