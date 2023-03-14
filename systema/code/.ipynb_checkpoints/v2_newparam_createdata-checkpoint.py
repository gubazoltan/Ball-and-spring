import numpy as np
import code_s3b1r2_improving as sprc
import time
import multiprocessing

#%%
def calculator(k3):
    #set parameters for the calculation
    r1 = 2
    r2 = 2
    k1 = 0.93
    k2 = 1
    l1 = 6
    l2 = 6
    d = 20
    
    l3 = 10
    R = 10
    gamma = np.pi/2
    
    numpoints = 41

    a1s = np.linspace( -np.pi, np.pi, numpoints)
    a2s = np.linspace( -np.pi, np.pi, numpoints)

    U = sprc.potprep(r1 = r1, r2 = r2, k1 = k1, k2 = k2, k3 = k3, l1 = l1, l2 = l2, l3 = l3, d = d, R = R, gamma = gamma)
    
    dvec_container = sprc.dvector_field(pot = U, a1s = a1s, a2s = a2s, filename = None)
    
    phases = sprc.phase_func(dvec_container = dvec_container)
    vorticities = sprc.vort_func(phases = phases)
    
    
    filename = "version2_nonrestricted_k1_0.93_k3_" + f"{k3:.3f}" + "_" + str(numpoints)
    
    np.savetxt(fname = filename, X = vorticities)
    
    return vorticities

#%%
start = time.time()

k3s = [0.78, 0.80, 0.82, 0.84, 0.86]

pool   = multiprocessing.Pool() #create pool instance
datas = pool.map(calculator,k3s)

end = time.time()

print((end-start)/60)

#%%

reshaped_data = np.array(datas).reshape( (len(k3s) * 41 * 41))
fname = "version2_vorts_k1_0.93"
np.savetxt(fname = fname, X = reshaped_data)
