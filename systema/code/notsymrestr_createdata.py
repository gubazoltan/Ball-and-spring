import numpy as np
import systema_code as sac
import time
import multiprocessing

#%%

#define auxiliary function to obtain the Weyl points for different k3 values
def vort_calc(k3):
    
    #set parameters for the calculation
    r1 = 2.6
    r2 = 2
    k1 = 1
    k2 = 1
    l1 = 6
    l2 = 6
    d = 20
    l3 = 10
    R = 10
    
    #define the grid
    numpoints = 61

    a1s = np.linspace( -np.pi, np.pi, numpoints)
    a2s = np.linspace( -np.pi, np.pi, numpoints)
    
    #create the potential
    U = sac.potprep(r1 = r1, r2 = r2, k1 = k1, k2 = k2, k3 = k3, l1 = l1, l2 = l2, l3 = l3, d = d, R = R)
    
    #calculate the d vectors on the grid
    dvec_container = sac.dvector_field(pot = U, a1s = a1s, a2s = a2s, filename = None)
    
    #calculate the phases from the dvectors
    phases = sac.phase_func(dvec_container = dvec_container)
    
    #calculate the vorticities from the phases
    vorticities = sac.vort_func(phases = phases)
        
    #return the vorticities
    return vorticities

#%%

start = time.time()

#define the possible values of k3
k3s = [0.76, 0.78, 0.80, 0.82, 0.84, 0.86]

#create pool instance for multi-threading
pool = multiprocessing.Pool()

#carry out the calculation
datas = pool.map(vort_calc, k3s)

end = time.time()

#print elapsed time
print((end-start)/60)

#%%

#save the data
numpoints = 61

reshaped_data = np.array(datas).reshape( (len(k3s) * numpoints * numpoints))

fname = "../datas/notsymrestr_r1_2.6_" + str(numpoints)

np.savetxt(fname = fname, X = reshaped_data)
