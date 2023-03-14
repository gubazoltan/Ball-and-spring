import numpy as np
import systema_code as sac
import time

#%%

#set parameters for the calculation
r1 = 2.6
r2 = 2
k1 = 1
k2 = 1
l1 = 6
l2 = 6
l3 = 10
d = 20
R = 10

k3 = 0.86

#%%

#define the grid
numpoints = 61
a1s = np.linspace( -np.pi, np.pi, numpoints)
a2s = np.linspace( -np.pi, np.pi, numpoints)

#%%

#define the name of the file into which the data is to be saved
filename = "combinedspectrum_r1_2.6_k3_0.86" + "_" + str(numpoints)

#create the potential
U = sac.potprep(r1 = r1, r2 = r2, k1 = k1, k2 = k2, k3 = k3, l1 = l1, l2 = l2, l3 = l3, d = d, R = R)

start = time.time()

#calculate the d vectors on the grid
dvec_container = sac.dvector_field(pot = U, a1s = a1s, a2s = a2s, filename = filename)

end = time.time()

#print elapsed time
print((end-start)/60)