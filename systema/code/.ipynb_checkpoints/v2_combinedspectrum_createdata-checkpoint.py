import numpy as np
import code_s3b1r2_v2 as sprc
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

gamma = np.pi/2

#%%
numpoints = 61

a1s = np.linspace( -np.pi, np.pi, numpoints)
a2s = np.linspace( -np.pi, np.pi, numpoints)

#%%
filename = "v2_combinedspectrum_r1_2.6_k3_0.86" + "_" + str(numpoints)

U = sprc.potprep(r1 = r1, r2 = r2, k1 = k1, k2 = k2, k3 = k3, l1 = l1, l2 = l2, l3 = l3, d = d, R = R, gamma = gamma)

start = time.time()

dvec_container = sprc.dvector_field(pot = U, a1s = a1s, a2s = a2s, filename = filename)

end = time.time()
print((end-start)/60)