import numpy as np
import matplotlib.pyplot as plt
import sympy as sp    
import matplotlib.colors

def exact_spectrum(k0s, angles, m0s, dm1, dm2, diff = True):
    """
    Calculate the exact eigenfrequencies for the system characterised by k0s, angles, m0s, dm1 and dm2. 
    Instead of using both m0s value and dm1, dm2 values, one can use simple m0s and set dm1 = dm2 = 0. The two are equivalent. 
    In the case of this function, calculation of the eigenfrequencies is exact, meaning that the 6x6 dynamical matrix is diagonalized numerically.

    Parameters
    ----------
    k0s : float list of 2 entries. Contains the spring constants for which the spectrum is to be calculated.
    
    angles : float list of 2 entries. Contains the angles for which the spectrum is to be calculated.
    
    m0s : float list of 2 entries. Contains the masses for which the spectrum is to be calculated.
    
    dm1 : float. The mass detuning of the first mass.
    
    dm2 : float. The mass detuning of the second mass.
    
    diff : boolean. For True, only the difference between the originally degenerate eigenfrequencies is returned.
        For False, both eigenfrequencies are returned. The default is True.

    Returns
    -------
    for diff = True : returns the absolute values of the difference of the initially degenerate normal modes. 
    for diff = False : returns both the initially degenerate eigenfrequencies. 
    
    """
    
    #these values are always the same
    m3 = 1.
    k3 = 1.
    
    #unpack the input values
    k1, k2 = k0s
    alpha, beta = angles
    m1, m2 = m0s
    
    #update the masses in the case of additional mass detunings
    m1 += dm1
    m2 += dm2
    
    #create auxiliary matrices
    Rmat = np.array([[0., 0., np.cos(beta), -np.sin(beta), -np.cos(beta), np.sin(beta)],
                      [-np.cos(alpha), -np.sin(alpha), 0., 0., np.cos(alpha), np.sin(alpha)],
                      [-1., 0., 1., 0., 0., 0.]])

    Kmat = np.array([[k1, 0., 0.],
                      [0., k2, 0.],
                      [0., 0., k3]])

    Mmat = np.array([[1. / np.sqrt(m1), 0., 0., 0., 0., 0.],
                        [0., 1. / np.sqrt(m1), 0., 0., 0., 0.],
                        [0., 0., 1. / np.sqrt(m2), 0., 0., 0.],
                        [0., 0., 0., 1. / np.sqrt(m2), 0., 0.],
                        [0., 0., 0., 0., 1. / np.sqrt(m3), 0.],
                        [0., 0., 0., 0., 0., 1. / np.sqrt(m3)]]) 
    
    #define the dynamical matrix of the system
    DinM = Mmat @  ( Rmat.T @  ( Kmat @ ( Rmat @ Mmat ) ) )
    
    #find the eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(DinM)
    
    #sort the eigenvalues
    np.sort(vals)
    
    #make the small values zero
    vals[ np.abs(vals) < 1e-10 ] = 0
    
    #take the square root of the values
    vals = np.sqrt(vals)
    
    #if only the difference is needed
    if diff:
        #get the difference between the relevant eigenvalues
        val_diff = np.abs(vals[4] - vals[3])

        #return the absolute value difference of the relevant eigenvalues
        return val_diff
    
    #otherwise return both frequencies
    else:
        return vals[3],vals[4]
    
def spectrum_grid(k0s, angles, m0s, dm1s, dm2s, plot = False, cbar_visible = False, save_file = None):
    """
    Create the spectrum as a function of the mass detunings dm1 and dm2. 
    The spectrum is defined as the absolute value difference between the initially degenerate eigenfrequencies.

    Parameters
    ----------
    k0s : float list of 2 entries. Contains the spring constants for which the spectrum is to be calculated.
    
    angles : float list of 2 entries. Contains the angles for which the spectrum is to be calculated.
    
    m0s : float list of 2 entries. Contains the masses for which the spectrum is to be calculated.
    
    dm1s : list of floats. The possible detunings of the first mass.
    
    dm2s : list of floats. The possible detunings of the second mass.
    
    plot : boolean. if True then create a heatmap containing the spliting. The default is False.
    
    cbar_visible : boolean, if True then create a colorbar for the plot. The default is False.
    
    save_file: string. The name of the into which the plot is to be saved. Optional: the default values is None.

    Returns
    -------
    diffs : 2-dimensional list of floats. Contains the absolute value difference between the initially degenerate eigenfrequencies. 

    """
    #define the grid for the spectrum
    M1, M2 = np.meshgrid(dm1s, dm2s)
    
    #create a container for the diff values
    diffs = np.zeros( len(dm1s) * len(dm2s) ).reshape((len(dm2s), len(dm1s)))
    
    #iterate through the grid and calculate the spectrum
    for i, dm2 in enumerate(dm2s):
        for j, dm1 in enumerate(dm1s):
            
            #calculate the absolute value difference of the initially degenerate eigenfrequencies
            diff = exact_spectrum(k0s = k0s, angles = angles, m0s = m0s, dm1 = dm1, dm2 = dm2)
    
            #add the value to the container
            diffs[i,j] = diff
            
    #plot the diff values if necessary
    if plot == False:
        pass
    
    else:
        #create the figure
        fig = plt.figure()
        
        #define the extent of the values on the axis based on the dm1s and dm2s values
        extent = [dm1s[0], dm1s[-1], dm2s[0], dm2s[-1]]
        
        #create the heatmap plot 
        #NOTE: need to iterate in opposite direction in the case of the first index
        ims = plt.imshow(diffs[::-1,:], extent = extent)
        
        #add labels to the axis
        plt.xlabel(r"$\Delta m_1$", fontsize = 16)
        plt.ylabel(r"$\Delta m_2$", fontsize = 16)

        #add colorbar whenever needed
        if cbar_visible == True:
            
            #add colorbar
            cbar = plt.colorbar(ims)
            
            #add labels to the colorbar
            cbar.set_label(r"$|\omega_1-\omega_2|$", fontsize = 16)
            
        else:
            pass
        
        #save the plot if the save_file optional parameter is not None
        if save_file == None:
            pass
        
        else: 
            #save the figure in the figure folder
            save_file = "../figures/" + save_file
            plt.savefig(save_file, dpi = 1200)
            
    #return the diff values
    return diffs

def spectrum_single_surfaces(k0s, angles, m0s, dm1s, dm2s, plot = False, save_file = None, elev = 30, azim = -50):
    """
    Function that calculates the exact splitting on a grid which is defined with respect to the m0s point.

    Parameters
    ----------
    k0s : float list of 2 entries. Contains the spring constants for which the spectrum is to be calculated.
    
    angles : float list of 2 entries. Contains the angles for which the spectrum is to be calculated.
    
    m0s : float list of 2 entries. Contains the masses for which the spectrum is to be calculated.
    
    dm1s : list of floats. The possible detunings of the first mass.
    
    dm2s : list of floats. The possible detunings of the second mass.
    
    plot : boolean. if True then create a heatmap containing the spliting. The default is False.
    
    save_file: string. The name of the into which the plot is to be saved. Optional: the default values is None.
    
    elev : float. The elevation for the 3D plot. The default value is 30.
    
    azim : float. The azimuth angle for the 3D plot. The default value is -50.

    Returns
    -------
    diffs : 2-dimensional list of floats. Contains the absolute value difference between the initially degenerate eigenfrequencies.

    """
    
    #define the grid for the spectrum
    M1, M2 = np.meshgrid(dm1s, dm2s)
    
    #create a container for the diff values
    diffs = np.zeros( len(dm1s) * len(dm2s) ).reshape((len(dm2s), len(dm1s)))

    #iterate through the grid and calculate the spectrum
    for i, dm2 in enumerate(dm2s):
        for j, dm1 in enumerate(dm1s):
            
            #calculate the absolute value difference of the initially degenerate eigenfrequencies
            diff = exact_spectrum(k0s = k0s, angles = angles, m0s = m0s, dm1 = dm1, dm2 = dm2)
    
            #add the value to the container
            diffs[i,j] = diff
    
    #scale up the diff values
    diffs_scaled = diffs * 1000
    
    #plot the diff values if necessary
    if plot == False:
        pass
    
    else:
        #create the 3D figure
        fig = plt.figure(figsize = (5,5))
        ax = fig.add_subplot(111, projection='3d')
        
        #add labels to the z-axis
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r"$1000 \cdot \Delta \omega$", fontsize = 16, rotation = 90)
        
        #normalize the diff values for the colormap
        vmin = np.min( diffs_scaled[:][:] )
        vmax = np.max( diffs_scaled[:][:] )
        norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
        
        #add surface plot
        ax.plot_surface(M1, M2, diffs_scaled, cmap = matplotlib.cm.viridis,
                               linewidth = 1, antialiased = False, alpha = 0.8, facecolors = plt.cm.viridis(norm(diffs_scaled)) )

        #add labels to the axis
        plt.xlabel(r"$\Delta m_1$", fontsize = 16)
        plt.ylabel(r"$\Delta m_2$", fontsize = 16)
             
        #change the point of view for the 3D figure
        ax.view_init(elev=elev, azim=azim)
  
        #save the plot if the save_file optional parameter is not None
        if save_file == None:
            pass
        
        else: 
            #save the figure in the figure folder
            save_file = "../figures/" + save_file
            plt.savefig(save_file, dpi = 1200, bbox_inches = "tight")
            
    #return the diff values
    return diffs

def symbolic_matrices(k0s, angles):
    """
    Create a symbolic dynamical matrix and its derivatives where the masses m1 and m2 are symbols.

    Parameters
    ----------
    k0s : float list of 2 entries. Contains the spring constants.
    
    angles : float list of 2 entries. Contains the angles.
    
    Returns
    -------
    symbDyn : sympy Matrix. The symbolic dynamical matrix of the system when the spring constants are fixed.
    
    d1_symbDyn : sympy Matrix. The symbolic derivative of the dynamical matrix of the system. 
                The derivative is taken with respect to the mass m1.
        
    d2_symbDyn : sympy Matrix. The symbolic derivative of the dynamical matrix of the system. 
                The derivative is taken with respect to the mass m2.
    
    m1 : sympy Symbol. The symbol representing the mass of the first body.
    
    m2 : sympy Symbol. The symbol representing the mass of the second body.

    """

    #these values are always the same
    m3 = 1.
    k3 = 1.
    
    #unpack the input values
    k1, k2 = k0s
    alpha, beta = angles

    #define the symbolic masses
    m1 = sp.Symbol('m_1', Real = True, Positive = True)
    m2 = sp.Symbol('m_2', Real = True, Positive = True)
    
    #create matrices
    Rmat = sp.Matrix([[0., 0., sp.cos(beta), -sp.sin(beta), -sp.cos(beta), sp.sin(beta)],
                      [-sp.cos(alpha), -sp.sin(alpha), 0., 0., sp.cos(alpha), sp.sin(alpha)],
                      [-1., 0., 1., 0., 0., 0.]])

    Kmat = sp.Matrix([[k1, 0., 0.],
                      [0., k2, 0.],
                      [0., 0., k3]])

    M0_pow = sp.Matrix([[1. / sp.sqrt(m1), 0., 0., 0., 0., 0.],
                        [0., 1. / sp.sqrt(m1), 0., 0., 0., 0.],
                        [0., 0., 1. / sp.sqrt(m2), 0., 0., 0.],
                        [0., 0., 0., 1. / sp.sqrt(m2), 0., 0.],
                        [0., 0., 0., 0., 1. / sp.sqrt(m3), 0.],
                        [0., 0., 0., 0., 0., 1. / sp.sqrt(m3)]])
    
    #define the symbolic dynamical matrix
    symbDyn = M0_pow *  ( Rmat.T *  ( Kmat * ( Rmat * M0_pow ) ) )
    
    #find the derivatives of the symbolic dynamical matrix as a function of the masses m1 and m2
    d1_symbDyn = sp.diff(symbDyn, m1)
    d2_symbDyn = sp.diff(symbDyn, m2)
    
    #return the matrices and the symbols
    return symbDyn, d1_symbDyn, d2_symbDyn, m1, m2

def projector_func(Mat, v1, v2):
    """
    Function that is used to project down a 6x6 Matrix to the subspace spanned by the vectors v1 and v2.
    
    Parameters
    ----------
    Mat : sympy Matrix. The 6x6 matrix which needs to be projected down to the subspace spanned by v1 and v2.
    
    v1 : sympy array. The column vector representing the first vector in the subspace.
    
    v2 : sympy array. The column vector representing the second vector in the subspace.
    
    Returns
    -------
    proj_Mat : sympy Matrix. The 2x2 projected matrix.
    
    """
   
    #define a container for the projected matrix
    proj_Mat = np.zeros(4).reshape((2,2))
    
    #define an array from the eigenvectors
    vs = [v1, v2]
    
    #iterate though the array for each entry of the matrix
    for i, vi in enumerate(vs):
        for j, vj in enumerate(vs):
            
            #obtain the projected matrix element
            proj_Mat[i,j] = vi.T @ ( Mat @ vj) 
    
    #make the matrix symmetric
    proj_Mat = (proj_Mat + proj_Mat.T) / 2
 
    #return the projected matrix
    return proj_Mat

def pauli_components(Mat):
    """
    Function used to decompose a 2x2 real symmetric matrix into a sum of pauli x and z matrices.    

    Parameters
    ----------
    Mat : numpy or sympy array. The 2x2 matrix to be decomposed. 
    
    Returns
    -------
    dx, dz : floats. The Pauli x and z components of the Mat matrix

    """

    #define the Pauli matrices
    sx = np.array([ [0, 1], [1, 0]] )
    sz = np.array([ [1, 0], [0, -1]] )
    
    #get the corresponding components
    dx = np.trace(sx @ Mat) / 2
    dz = np.trace(sz @ Mat) / 2
    
    #return the components
    return dx, dz

def it_func(m0s, symbDyn, d1_symbDyn, d2_symbDyn, m1, m2):
    """
    Calculate a single-step of the Newton method that is used during the search of the Weyl points. 

    Parameters
    ----------
    m0s : float list of 2 entries. Contains the masses for which the dynamical matrix is to be calculated.
    
    symbDyn : sympy Matrix. The symbolic dynamical matrix of the system when the spring constants are fixed.
    
    d1_symbDyn : sympy Matrix. The symbolic derivative of the dynamical matrix of the system. 
                The derivative is taken with respect to the mass m1.
        
    d2_symbDyn : sympy Matrix. The symbolic derivative of the dynamical matrix of the system. 
                The derivative is taken with respect to the mass m2.
    
    m1 : sympy Symbol. The symbol representing the mass of the first body.
    
    m2 : sympy Symbol. The symbol representing the mass of the second body.
    
    Returns
    -------
    deltamass : numpy array of floats, a step towards a degeneracy point in the configuration space.

    """
    
    #unpack the input values
    m1val, m2val = m0s
    
    #evaluate the dynamical matrix and its derivatives at the m0 points
    Dyn_0_evaled = symbDyn.evalf(subs = {m1 : m1val, m2 : m2val})
    Dyn_d1_evaled = d1_symbDyn.evalf(subs = {m1 : m1val, m2 : m2val})
    Dyn_d2_evaled = d2_symbDyn.evalf(subs = {m1 : m1val, m2 : m2val})
    
    #find the eigevectors of the dynamical matrix so that it can be projected to the quasi-degenerate subspace

    #make a numpy array out of the sympy matrix
    D0vecs = np.array(Dyn_0_evaled).astype(np.float64)
    
    #get the eigenvalues and the eigenvectors
    vals, vecs = np.linalg.eigh(D0vecs)

    #now sort the eigenvectors based on the eigenvalues
    vals_indices = np.argsort(vals)

    #sort the eigenvectors 
    sorted_vecs = vecs[:,vals_indices[::]]
    
    #get the relevant eigenvectors corresponding the quasi-degenerate subspace
    v1 = sorted_vecs[:,3]
    v2 = sorted_vecs[:,4]
    
    #project the evaluated dynamical matrix and its derivatives to the quasi-degenerate subspace
    Deff_0 = projector_func(Mat = Dyn_0_evaled, v1 = v1, v2 = v2)
    Deff_d1 = projector_func(Mat = Dyn_d1_evaled, v1 = v1, v2 = v2)
    Deff_d2 = projector_func(Mat = Dyn_d2_evaled, v1 = v1, v2 = v2)
    
    #obtain the components of the delta vector and the g matrix
    dx, dz = pauli_components(Mat = Deff_0) 
    g_x1, g_z1 = pauli_components(Mat = Deff_d1)
    g_x2, g_z2 = pauli_components(Mat = Deff_d2)

    #define the delta vector and the g matrix
    dvec = np.array([[dx],[dz]])
    g_mat = np.array([[g_x1, g_x2],
                      [g_z1, g_z2]])
    
    #get relative position of the approximate degeneracy 
    deltamass = - (np.linalg.inv(g_mat)) @ dvec 
    
    #return the delta mass vector
    return deltamass

def weyl_search(k0s, angles, m0s, max_it = 30, prec_it = 1e-12):
    """
    Find Weyl points in the configuration space using Newton's method.

    Parameters
    ----------
    k0s : float list of 2 entries. Contains the spring constants.
    
    angles : float list of 2 entries. Contains the angles.
    
    m0s : float list of 2 entries. Contains the masses for which the dynamical matrix is to be calculated.
    
    max_it : int. The maximum number of iterations done in the search. The default is 30.
    
    prec_it : float. The precision of the Weyl point search. The default is 1e-12.

    Returns
    -------
    it_m0s : list of two floats. The final approximate position of the degeneracy point. 
    
    itnum : int. The number of iterations done in the search.

    """

    #create the symbolic matrices
    symbDyn, d1_symbDyn, d2_symbDyn, m1, m2 = symbolic_matrices(k0s = k0s, angles = angles)
    
    #initial point in the Weyl point search
    it_m0s = m0s.copy()
    
    #set the iteration number to one
    itnum = 1
    
    #calculate the first correction to it_m0s
    deltamass = it_func(m0s = it_m0s, symbDyn = symbDyn,
                        d1_symbDyn = d1_symbDyn, d2_symbDyn = d2_symbDyn, m1 = m1, m2 = m2 )
    
    #repeat this procedure if the norm of deltamass is larger than prec_it
    while np.linalg.norm(deltamass) > prec_it and itnum < max_it : 
        
        #update the approximate location of the degeneracy
        it_m0s[0] += deltamass[0,0]
        it_m0s[1] += deltamass[1,0]
        
        #calculate the new correction to it_m0s
        deltamass = it_func(m0s = it_m0s, symbDyn = symbDyn,
                            d1_symbDyn = d1_symbDyn, d2_symbDyn = d2_symbDyn, m1 = m1, m2 = m2 )
        
        #increase the iteration number
        itnum += 1
        
    #return the approximate location of the degeneracy point and the number of iterations
    return it_m0s, itnum

def isin(locs, newloc, splitting, loc_threshold = 1e-12, freq_threshold = 1e-13):
    """
    Function to evaluate whether the new degeneracy point is indeed a new degeneracy point.

    Parameters
    ----------
    locs : list of 2-component lists. The list that contains those [m1,m2] coordinates which are degeneracy points.
    
    newloc : 2-component list. The new [m1,m2] coordinate that is returned from the Newton's method.
    
    splitting : float. The absolute value difference of the eigenfrequencies of the initially degenerate normal modes.
    
    loc_threshold : float. The minimum distance between two degeneracy points after which they are considered to be different. The default is 1e-12.
    
    freq_threshold : float. The maximum splitting of the degenerate frequencies at the degeneracy point. The default is 1e-13.

    Returns
    -------
    boolean
    True: if the newloc is a degeneracy point which was already found.
    False: if the newloc is a degeneracyp point which was not found yet.

    """
    #iterate through all the already found degeneracy points
    for loc in locs:
        
        #calculate the distance between the degeneracy point and the newly found degeneracy point
        dx = np.abs(loc[0] - newloc[0])
        dy = np.abs(loc[1] - newloc[1])
        
        dist = np.sqrt(dx**2 + dy**2)
        
        #if the distance is smaller than the threshhold then return True
        if dist < loc_threshold:
            return True
        
        else:
            pass
    
    #if the splitting is larger than the threshhold then the degeneracy is "already found" = it is not a degeneracy point
    if splitting > freq_threshold:
        return True
    
    else:
        pass
    
    #return False meaning that the newly found degeneracy point is indeed a new degeneracy point. 
    return False

def DinM_eff_creator(k0s, angles, m0s):
    """
    Fucntion that creates the effective dynamical matrix for the system characterised by k0s, angles and m0s.
    This effective dynamical matrix is used to find the charge of the degeneracy point.

    Parameters
    ----------
    k0s : float list of 2 entries. Contains the spring constants for which the effective dynamical matrix is to be calculated.
    
    angles : float list of 2 entries. Contains the angles for which the effective dynamical matrix is to be calculated.
    
    m0s : float list of 2 entries. Contains the masses for which the effective dynamical matrix is to be calculated.

    Returns
    -------
    DinM_eff : 2x2 sympy matrix. The effective dynamical matrix which is a symbolic object with symbols dm1 and dm2. 
    
    dm1 : sympy Symbol. The symbol representing the mass detuning of the first body.
    
    dm2 : sympy Symbol. The symbol representing the mass detuning of the second body.

    """
    #these values are always the same
    m3 = 1.
    k3 = 1.
    
    #unpack the input values
    k1, k2 = k0s
    alpha, beta = angles
    m1, m2 = m0s
    
    #define the mass detunings as sympy symbols
    dm1 = sp.Symbol('m_1', real = True, positive = True)
    dm2 = sp.Symbol('m_2', real = True, positive = True)
    
    #define the auxiliary matrices
    Rmat = sp.Matrix([[0., 0., np.cos(beta), -np.sin(beta), -np.cos(beta), np.sin(beta)],
                      [-np.cos(alpha), -np.sin(alpha), 0., 0., np.cos(alpha), np.sin(alpha)],
                      [-1., 0., 1., 0., 0., 0.]])
    
    Kmat = sp.Matrix([[k1, 0., 0.],
                      [0., k2, 0.],
                      [0., 0., k3]])
    
    M0_pow = sp.Matrix([[1. / sp.sqrt(m1), 0, 0, 0, 0, 0],
                        [0, 1. / sp.sqrt(m1), 0, 0, 0, 0],
                        [0, 0, 1. / sp.sqrt(m2), 0, 0, 0],
                        [0, 0, 0, 1. / sp.sqrt(m2), 0, 0],
                        [0, 0, 0, 0, 1. / sp.sqrt(m3), 0],
                        [0, 0, 0, 0, 0, 1. / sp.sqrt(m3)]])    
    
    M1_pow = - 1 / 2 * sp.Matrix([[dm1/(m1**(3/2)), 0, 0, 0, 0, 0],
                                  [0, dm1/(m1**(3/2)), 0, 0, 0, 0],
                                  [0, 0, dm2/(m2**(3/2)), 0, 0, 0],
                                  [0, 0, 0, dm2/(m2**(3/2)), 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]])
    
    M2_pow = 3 / 8 * sp.Matrix([[ dm1**2 / ( m1**(5/2) ), 0, 0, 0, 0, 0],
                                [0, dm1**2 / ( m1**(5/2) ), 0, 0, 0, 0],
                                [0, 0, dm2**2 / ( m2**(5/2) ), 0, 0, 0],
                                [0, 0, 0, dm2**2 / ( m2**(5/2) ), 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]])
    
    #define the dynamical matrix and evaluate it at the origin of the configuration space
    D0 = M0_pow *  ( Rmat.T *  ( Kmat * ( Rmat * M0_pow ) ) )
    D0vecs = D0.evalf(subs = {dm1 : 0, dm2 : 0})
    
    D0vecs = np.array(D0vecs).astype(np.float64)
    vals, vecs = np.linalg.eigh(D0vecs)
    
    #now sort the eigenvectors based on the eigenvalues
    vals_indices = np.argsort(vals)

    #sort the eigenvectors 
    sorted_vecs = vecs[:,vals_indices[::]]
    
    #set the first three eigenvalues to zero because they should be zero!
    sorted_vals = vals[vals_indices]
    sorted_vals[:3] = 0
    
    #now we have the eigenstates and the eigenvalues
    v0 = sp.Matrix(sorted_vecs[:,0])
    v1 = sp.Matrix(sorted_vecs[:,1])
    v2 = sp.Matrix(sorted_vecs[:,2])
    v3 = sp.Matrix(sorted_vecs[:,3])
    v4 = sp.Matrix(sorted_vecs[:,4])
    v5 = sp.Matrix(sorted_vecs[:,5])
    
    #OPTIONAL: set +- gauge for all the eigenvectors
    v0 *= np.sign(v0[0,0])
    v1 *= np.sign(v1[0,0])
    v2 *= np.sign(v2[0,0])
    v3 *= np.sign(v3[0,0])
    v4 *= np.sign(v4[0,0])
    v5 *= np.sign(v5[0,0])
    
    #create the vectors of the originally degenerate states
    v03 = np.array([[np.sqrt(3.)/2.],[-1./2.],[-np.sqrt(3.)/2.],[-1./2.],[0.],[1.]]) / np.sqrt(3.)
    v04 = np.array([[1./2.],[np.sqrt(3.)/2.],[1./2.],[-np.sqrt(3.)/2.],[-1.],[0.]]) / np.sqrt(3.)
    
    #check the overlap of these states 
    ol33 = v03.T.dot(v3)
    ol43 = v04.T.dot(v3)
    ol34 = v03.T.dot(v4)
    ol44 = v04.T.dot(v4)
    
    #if the overlap of v3 and v03 is higher than the overlap of v3 and v04 then set v3 as the state that corresponds to v03
    if np.abs(ol33) > np.abs(ol43) :
        
        #assign the states this way
        v3_star = v3.copy()
        v4_star = v4.copy()
        
        #if the overlap is negative then v3_star is not correctly aligned with v03 and the overall orientation must be flipped
        if ol33 < 0: 
            v3_star *= -1
            
        else:
            pass
        
        #do the same thing for v4_star too
        if ol44 < 0: 
            v4_star *= -1
            
        else:
            pass
    
    #otherwise the overlap between v3 and v04 is higher than the overlap between v3 and v03 meaning that v3 corresponds to v04
    else: 
        
        #assign the states this way
        v3_star = v4.copy()
        v4_star = v3.copy()   
        
        #if the overlap is negative then v3_star is not correctly aligned with v04 and the overall orientation must be flipped
        if ol43 < 0:
            v3_star *= -1
            
        else:
            pass
        
        #do the same thing for v4_star too
        if ol34 < 0:
            v4_star *= -1
        else:
            pass
    
    #now assign the correctly oriented eigenvectors 
    v3 = v3_star.copy()
    v4 = v4_star.copy()
    
    #now carry out a second order Schrieffer-Wolff transformation
    D0 = M0_pow *  ( Rmat.T *  ( Kmat * ( Rmat * M0_pow ) ) )
    
    #calculate the first order perturbation of the dyanmical matrix
    D1_prime = M0_pow *  (Rmat.T *  (Kmat * (Rmat * M1_pow))) +  M1_pow *  (Rmat.T *  (Kmat * (Rmat * M0_pow)))
    
    #define an array for the eigenvectors
    vs = np.array([v0, v1, v2, v3, v4, v5])
    
    #define the matrix of the first order perturbation
    Mfirst = sp.zeros(6,6)

    #obtain the matrix elements of the first order perturbation
    for ii, vi in enumerate(vs):
        for jj, vj in enumerate(vs):
            entry = vi.T @ D1_prime @ vj
            Mfirst[ii,jj] = entry
            
    #do a symmetrization
    Mfirst = ( Mfirst + Mfirst.T ) / 2 
    
    #calculate the second order perturbation of the dynamical matrix
    D2_prime =  M0_pow *  (Rmat.T *  (Kmat * (Rmat * M2_pow)))
    D2_prime += M1_pow *  (Rmat.T *  (Kmat * (Rmat * M1_pow)))
    D2_prime += M2_pow *  (Rmat.T *  (Kmat * (Rmat * M0_pow)))
    
    #define the matrix of the second order perturbation
    Msecond = sp.zeros(6,6)
    
    #obtian the matrix elements
    for ii, vi in enumerate(vs):
        for jj, vj in enumerate(vs):
            entry = vi.T @ D2_prime @ vj
            Msecond[ii,jj] = entry
        
    #do a symmetrization
    Msecond = ( Msecond + Msecond.T ) / 2
    
    #define the effective hamiltonian
    DinM_eff = sp.zeros(2,2)
    
    #add the unperturbed terms
    DinM_eff[0,0] = v3.T @ D0 @ v3 
    DinM_eff[1,1] = v4.T @ D0 @ v4
    DinM_eff[0,1] = v3.T @ D0 @ v4
    DinM_eff[1,0] = v4.T @ D0 @ v3
        
    #add the first order Schrieffer-Wolff terms
    DinM_eff[0,0] += Mfirst[3,3]
    DinM_eff[0,1] += Mfirst[3,4]
    DinM_eff[1,0] += Mfirst[4,3]
    DinM_eff[1,1] += Mfirst[4,4]
    
    DinM_eff[0,0] += Msecond[3,3]
    DinM_eff[0,1] += Msecond[3,4]
    DinM_eff[1,0] += Msecond[4,3]
    DinM_eff[1,1] += Msecond[4,4]
    
    #define the intermediate states
    intermediate_states = np.array([0,1,2,5])
    
    #get the eigenvalues
    w3_square = sorted_vals[3]
    w4_square = sorted_vals[4]
    
    #now add the second order Schrieffer-Wolff term
    for i in intermediate_states:
        
        #obtain the frequency square of the given intermediate state
        wi_square = sorted_vals[i]
        
        #add the corresponding term to all the matrix entries
        DinM_eff[0,0] += 1 / 2 * Mfirst[3,i] * Mfirst[i, 3] * ( 1 / (w3_square - wi_square) + 1 / (w3_square - wi_square) ) 
        DinM_eff[0,1] += 1 / 2 * Mfirst[3,i] * Mfirst[i, 4] * ( 1 / (w3_square - wi_square) + 1 / (w4_square - wi_square) ) 
        DinM_eff[1,0] += 1 / 2 * Mfirst[4,i] * Mfirst[i, 3] * ( 1 / (w4_square - wi_square) + 1 / (w3_square - wi_square) ) 
        DinM_eff[1,1] += 1 / 2 * Mfirst[4,i] * Mfirst[i, 4] * ( 1 / (w4_square - wi_square) + 1 / (w4_square - wi_square) ) 
    
    #return the dynamical matrix and the symbols needed to evaluate the effective matrix
    return DinM_eff, dm1, dm2


def normalize(v):
    """
    Used to normalize vectors

    Parameters
    ----------
    v : list of float. The vector to be normalized.

    Returns
    -------
    normalized vector : list of floats. 

    """
    #calcuate the norm
    norm = np.linalg.norm(v)
    
    #if the norm vanishes then simply return the vector
    if norm == 0:
        return v
    
    #otherwise return the normalized vector
    else:
        return v / norm
    
def dvector(DinM_eff, dm1, dm2, dm1_val, dm2_val):
    """
    Decompose the effective dynamical matrix into Pauli matrices

    Parameters
    ----------
    DinM_eff : sympy matrix, the effective dynamical matrix of the system at the given degeneracy point
    
    dm1 : sympy Symbol, the symbol representing the detuning of the mass of the first body.
    
    dm2 : sympy Symbol, the symbol representing the detuning of the mass of the second body.
    
    dm1_val : float. The detuning of the mass of the first body.
    
    dm2_val : float. The detuning of the mass of the second body.
    
    Returns
    -------
    dvec : 2-entry list of floats. The normalized decomposition of the effective dynamical matrix in terms of Pauli matrices.
        This function is used during the evaluation of the winding. 

    """

    #define the pauli matrices
    sx = np.array([[0,1],[1,0]])
    sz = np.array([[1,0],[0,-1]])    
    
    #evaluate the effective dynamical matrix
    Deff_evaled = DinM_eff.evalf(subs = {dm1 : dm1_val, dm2 : dm2_val})
    
    #make a numpy array out of the sympy matrix
    Deff_evaled = np.array(Deff_evaled).astype(np.float64)
    
    #get the components 
    dx = np.trace(sx @ Deff_evaled) / 2
    dz = np.trace(sz @ Deff_evaled) / 2
    
    #define the d vector
    dvec = np.array([dx, dz])

    #get the normalized d vector
    dvec = normalize(dvec)
    
    #return the normalized d vector
    return dvec

def signdet_method(Deff, dm1, dm2):
    """
    Method used to obtain the charge of a Weyl-point. The function evaluates the first order derivative of the effective dynamical matrix at the location of the 
    degeneracy point and calculates the sign of the determinant of the matrix that is composed out of the derivatives. 
    The sign of the determinant is equivalent to the chirality. 

    Parameters
    ----------
    Deff : sympy matrix. The 2x2 effective dynamical matrix of the system. 
    
    dm1 : sympy Symbol. The symbol representing the detuning of the mass of the first body.
    
    dm2 : sympy Symbol. The symbol representing the detuning of the mass of the second body.

    Returns
    -------
    q : int. The charge of the Weyl point. 

    """
    
    #calculate and evaluate the derivatives of the effective dynamical matrix at zero detunings (at the location of the degeneracy)
    d1_Deff = np.array((sp.diff(Deff,dm1)).evalf(subs = {dm1 : 0, dm2 : 0})).astype(np.float64)
    d2_Deff = np.array((sp.diff(Deff,dm2)).evalf(subs = {dm1 : 0, dm2 : 0})).astype(np.float64)
    
    #get the components of the derivatives in terms of Pauli matrices
    gx1, gz1 = pauli_components(d1_Deff)
    gx2, gz2 = pauli_components(d2_Deff)
    
    #construct the matrix that couples the mass detunings to the Pauli matrices 
    gmat = np.array([[gx1, gx2],
                     [gz1, gz2]])
    
    #calculate the charge (a.k.a chirality) of the Weyl point as the sign of the determinant of the g-matrix.
    q = np.sign(np.linalg.det(gmat))
    
    #return the charge of the Weyl point
    return q

def winding_method(Deff, dm1, dm2, radius = 1e-6, n = 100):  
    """
    Method used to obtain the charge of degeneracy points (both Charge-1 and Charge-2 Weyl points). 

    Parameters
    ----------
    Deff : sympy matrix. The 2x2 effective dynamical matrix of the system. 
    
    dm1 : sympy Symbol. The symbol representing the detuning of the mass of the first body.
    
    dm2 : sympy Symbol. The symbol representing the detuning of the mass of the second body.
    
    radius : float. The radius of the loop around the degeneracy point. The default is 1e-8.
    
    n : int. The number of lines into which the loop is decomposed. The default is 100.

    Returns
    -------
    q : int. The charge of the degeneracy point. 

    """
    #create a list of angles that is used to parametrise the loop that encloses the degeneracy point
    phis = np.linspace(0, 2 * np.pi, n, endpoint = False)
    
    #using the angle-parametrization, define a parametrization of the loop using the mass detunings.
    dm1_vals = radius * np.cos(phis)
    dm2_vals = radius * np.sin(phis)
    
    #define the container for the winding (chirality, charge)
    q = 0

    #define container for the phases
    phases = np.zeros(n)
    
    #iterate through the loop
    for idx, dm1_val in enumerate(dm1_vals):
        
        #obtain the dm2 value
        dm2_val = dm2_vals[idx]
        
        #obtain the corresponding normalized d vector
        dvec = dvector(DinM_eff = Deff, dm1 = dm1, dm2 = dm2, dm1_val = dm1_val, dm2_val = dm2_val)
        
        #calculate the phase of the dvector and add the phase to the container
        phases[idx] = np.arctan2( dvec[1], dvec[0] )
        
    #then calculate the winding of the dvectors
    
    #iterate through the angles
    for i in range(n):
        
        #get the angle difference between the phases and add it to the winding 
        q += np.angle( np.exp( 1.j * ( phases[i] - phases[ i - 1] ) ) )
    
    #divide the winding container by 2pi
    q = q / (2 * np.pi)
    
    #return the winding
    return q

def charge(k0s, angles, loc, method = "winding"):
    """
    Function used to calculate the charge of the degeneracy point either with the "winding" or with the "signdet" method.

    Parameters
    ----------
    k0s : float list of 2 entries. Contains the spring constants for which the effective dynamical matrix is to be calculated.
    
    angles : float list of 2 entries. Contains the angles for which the effective dynamical matrix is to be calculated.
    
    m0s : float list of 2 entries. Contains the masses for which the effective dynamical matrix is to be calculated.
    
    method : string. The method that we use to calculate the charge associated with the degeneracy point. The default is "winding".

    Returns
    -------
    q : int. The charge of the degeneracy point. 


    """
    
    #get the effective dynamical matrix
    Deff, dm1, dm2 = DinM_eff_creator(k0s = k0s, angles = angles, m0s = loc)
    
    #calculate the charge depending on the value of the method input variable
    if method == "signdet":
        q = signdet_method(Deff = Deff, dm1 = dm1, dm2 = dm2)
        
    elif method == "winding":
        q = winding_method(Deff = Deff, dm1 = dm1, dm2 = dm2)
    
    else:
        pass
    
    #return the charge of the degeneracy point
    return q

def all_point_finder(k0s, angles, search_it = 20, max_it = 20, prec_it = 1e-12, loc_threshold = 1e-11, freq_threshold = 1e-13):
    """
    Function that is used to find all the degeneracy points in the configuration space using Newton's method.

    Parameters
    ----------
    k0s : float list of 2 entries. Contains the spring constants that characterise the system.
    
    angles : float list of 2 entries. Contains the angles that characterise the system.
    
    search_it : int. The number of random Newton's method search to be carried out. The default is 20.
    
    max_it : int. The maximum number of iterations to be carried out during a single Newton's method. The default is 20.
    
    prec_it : float. The termination argument of the Newton's method. The default is 1e-12.
    
    loc_threshold : float. The minimum distance between two degeneracy points after which they are considered to be different. The default is 1e-12.
    
    freq_threshold : float. The maximum splitting of the degenerate frequencies at the degeneracy point. The default is 1e-13.

    Returns
    -------
    wplocs : list of 2-component lists. The list that contains those [m1,m2] coordinates which are degeneracy points.

    """
    
    #create a container for the location of the degeneracy points
    wplocs = []
    
    #if the parameters are symmetric then there is only a single degeneracy point and it is located at the origin
    if k0s == [1., 1.] and angles == [np.pi/3, np.pi/3]:
        wplocs.append([1., 1.])
        return wplocs
    
    #otherwise if the parameters are not symmetric then need to use Newton's method to find degeneracy points
    else:
        
        #do search_it number of Newton's method
        for i in range(search_it):
            
            #create random initial point for the search algorithm
            m0s = (np.random.rand(2)-0.5)/10+1
            
            #find the Weyl point closest to this initial point
            loc, itnum = weyl_search(k0s = k0s, angles = angles, m0s = m0s, max_it = max_it, prec_it = prec_it)
            
            #calculate the splitting at the given degeneracy point
            splitting = exact_spectrum(k0s = k0s, angles = angles, m0s = loc, dm1 = 0., dm2 = 0., diff = True)
            
            #check whether the newly found degeneracy point is indeed a new point
            if not isin(locs = wplocs, newloc = loc, splitting = splitting, loc_threshold = loc_threshold, freq_threshold = freq_threshold):
    
                #if it is a new degeneracy point then add it to the container
                wplocs.append(loc)
                
        #make numpy array out of container
        wplocs = np.array(wplocs)
        
        #return the degeneracy points
        return wplocs  

def charge_calc(k0s, angles, wplocs, method = "winding"):
    """
    Function that is used to calculate the charge of degeneracy point found in the configuration space for the system characterised by the 
    parameters k0s and angles.

    Parameters
    ----------
    k0s : float list of 2 entries. Contains the spring constants that characterise the system.
    
    angles : float list of 2 entries. Contains the angles that characterise the system.
    
    wplocs : list of 2-component lists. The list that contains those [m1,m2] coordinates which are degeneracy points.
    
    method : string. The method that we use to calculate the charge associated with the degeneracy point. The default is "winding". 

    Returns
    -------
    qs : list of integers. The list that contains the charges of the degeneracy points.
    
    """
    
    #create container for the charge of the degeneracy points
    qs = []
    
    #iterate through the degeneracy points
    for loc in wplocs:
        
        #obtain the charge of the degeneracy point
        q = charge(k0s = k0s, angles = angles, loc = loc, method = method)
        
        #add the charge to the container
        qs.append(q)
    
    #make numpy array out of the charge list
    qs = np.array(qs)
    
    #return the charges
    return qs