import numpy as np
import sympy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors

def potprep(r1, r2, k1, k2, k3, l1, l2, l3, d, R):
    """
    Function that is used to create the elastic potential. The elastic potential is a symbolic function of the coordinates x and y. 

    Parameters
    ----------
    r1 : float. The radius of the first ring.
    
    r2 : float. The radius of the second ring.
    
    k1 : float. The spring constant of the first srping.
    
    k2 : float. The spring constant of the second srping.
    
    k3 : float. The spring constant of the third srping.
    
    l1 : float. The rest length of the first spring. 
    
    l2 : float. The rest length of the second spring. 
    
    l3 : float. The rest length of the third spring. 
    
    d : float. The distance between the two rings.
    
    R : float. The y coordinate of the suspension point of the third spring.

    Returns
    -------
    U : sympy expression. The elastic potential as a symbolic function of the coordinates (x,y) and also the angles (alpha1,alpha2)

    """
    
    #define the coordinates
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    
    #define the angles
    a1 = sp.Symbol('alpha_1')
    a2 = sp.Symbol('alpha_2')
    
    #define the coordinates of the srpings' suspension points
    #first spring
    r1x = - d / 2 - r1 + r1 * sp.cos( a1 )
    r1y = r1 * sp.sin( a1 )  
    
    #second spring
    r2x = d / 2 + r2 - r2 * sp.cos( a2 ) 
    r2y = r2 * sp.sin( a2 ) 
    
    #third spring
    r3x = 0
    r3y = R
    
    #define the elongation of the springs
    dl1 = sp.sqrt( (r1x - x)**2 + (r1y - y)**2 ) - l1
    dl2 = sp.sqrt( (r2x - x)**2 + (r2y - y)**2 ) - l2
    dl3 = sp.sqrt( (r3x - x)**2 + (r3y - y)**2 ) - l3
    
    #define the total elastic potential
    U = 0.5 * k1 * (dl1**2) + 0.5 * k2 * (dl2**2) + 0.5 * k3 * (dl3**2)
    
    #return the potential
    return U

def myfunc(cords, l):
    """
    Function that we use to minimize the potential

    Parameters
    ----------
    cords : list with 2 floats. The (x,y) coordinates at which the elastic potential is evaluated
    
    l : lambda function. The lambda function that represents the elastic potential

    Returns
    -------
    val : float. The value of the elastic potential at the coordinates (x,y) = cords

    """
    #get the coordinates
    x, y = cords[0], cords[1]
    
    #evaluate the function
    val = l(x, y)
    
    #return the value
    return val

def mincords(pot, alpha1, alpha2, x0, y0):
    """
    Function that is used to find the equilibrium position of the body. The equilibrium position is found by minimizing the elastic potential of the body.

    Parameters
    ----------
    pot : sympy expression. The elastic potential as a symbolic function of the coordinates (x,y) and also the angles (alpha1,alpha2)
    
    alpha1 : float. The angle that characterises the suspension point of the first spring. 
    
    alpha2 : float. The angle that characterises the suspension point of the second spring. 
    
    x0 : float. The initial x0 coordinate for the minimization of the elastic potential.
    
    y0 : float. The initial y0 coordinate for the minimization of the elastic potential.

    Returns
    -------
    minx : float. The x coordinate of the equilibrium position of the body.
    
    miny : float. The y coordinate of the equilibrium position of the body.
    
    pot_at_a1a2 : sympy expression. The leastic potential as a function of the coordinates (x,y) evaluated at the angles (alpha1,alpha2)

    """
    #evaluate the potential for the two angles
    pot_at_a1a2 = pot.evalf(subs = {'alpha_1' : alpha1, 'alpha_2' : alpha2})

    #make a lambda function out of the expression which takes the x and y values and evaluate the potential
    evaled_pot = sp.utilities.lambdify(['x', 'y'], pot_at_a1a2)
    
    #define the initial guess for the coordinates that minimize the potential
    guess = [x0, y0]
    
    #find the coordinates that minimize the potential energy
    min_cords = minimize(myfunc, guess, args = (evaled_pot))
    minx, miny = min_cords.x
    
    #return the coordinates of the minima
    return minx, miny, pot_at_a1a2

def dvector(pot, alpha1, alpha2):
    """
    Decompose the dynamical matrix into Pauli matrices

    Parameters
    ----------
    pot : sympy expression. The elastic potential as a symbolic function of the coordinates (x,y) and also the angles (alpha1,alpha2)
    
    alpha1 : float. The angle that characterises the suspension point of the first spring. 
    
    alpha2 : float. The angle that characterises the suspension point of the second spring. 

    Returns
    -------
    dvec : 3-entry list of floats. The decomposition of the dynamical matrix in terms of Pauli matrices.

    """
    
    #obtain the equilibrium position of the body for angles alpha1 and alpha2
    minx, miny, pot_at_a1a2 = mincords(pot = pot, alpha1 = alpha1, alpha2 = alpha2, x0 = 0, y0 = 0)
    
    #get the first order derivatives of the potential
    dx = sp.diff(pot_at_a1a2, 'x')
    dy = sp.diff(pot_at_a1a2, 'y')
    
    #get the second order derivates
    dxdx = sp.diff(dx, 'x')
    dxdy = sp.diff(dx, 'y')
    dydy = sp.diff(dy, 'y')
    
    #evaluate the second order derivatives at the minima of the potential
    dxdx_evaled = dxdx.evalf(subs = {'x' : minx, 'y' : miny})
    dxdy_evaled = dxdy.evalf(subs = {'x' : minx, 'y' : miny})
    dydy_evaled = dydy.evalf(subs = {'x' : minx, 'y' : miny})
    
    #obtain the components of the dvector
    d0 = (dxdx_evaled + dydy_evaled) / 2
    d1 = dxdy_evaled 
    d3 = (dxdx_evaled - dydy_evaled) / 2
    
    #define the d vector
    dvec = np.array([d0, d1, d3], dtype = "float64")
    
    #return the d vector
    return dvec

def dvector_field(pot, a1s, a2s, filename = None):
    """
    Function that is used to calculate the dvector on a finite grid defined by the angles a1s and a2s.

    Parameters
    ----------
    pot : sympy expression. The elastic potential as a symbolic function of the coordinates (x,y) and also the angles (alpha1,alpha2)
    
    a1s : list of floats. The possible values of the angle alpha1.
    
    a2s : list of floats. The possible values of the angle alpha2.
    
    filename : string. The name of the file into which the calculated data is to be saved. The default is None.

    Returns
    -------
    dvec_container : matrix of size N x M x 3 of floats. The vector stored at dvec_container[i,j,:] corresponds to the dvector at angles alpha1=a1s[j] and alpha2=a2s[i].  

    """
    
    #create a container for the dvector components
    dvec_container = np.zeros( len(a1s) * len(a2s) * 3 ).reshape( ( len(a2s), len(a1s), 3 ) )
    
    #iterate through all the angles 
    for idx_y, alpha2 in enumerate(a2s):
        for idx_x, alpha1 in enumerate(a1s): 
            
            #obtain the dvector that corresponds to the angles alpha1 and alpha2
            dvec = dvector(pot = pot, alpha1 = alpha1, alpha2 = alpha2)
            
            #add the dvector to the container
            dvec_container[idx_y, idx_x, :] = dvec[:]
        
    #if the filename is none then the data will not be saved
    if filename == None:
        pass
    
    #otherwise save the components of the dvectors
    else:
        #save each component in a different file 
        save_file1 = "../datas/" + filename + "_1"
        save_file2 = "../datas/" + filename + "_2"
        save_file3 = "../datas/" + filename + "_3"
        
        np.savetxt(fname = save_file1, X = dvec_container[:,:,0])
        np.savetxt(fname = save_file2, X = dvec_container[:,:,1])
        np.savetxt(fname = save_file3, X = dvec_container[:,:,2])
        
    #return the dvector container
    return dvec_container
    
    
def eigenfreqs(dvec, what = "vals"):
    """
    Function that calculate the eigenfrequencies of the system by using the corresponding dvector.

    Parameters
    ----------
    dvec : 3-entry list of floats. The decomposition of the dynamical matrix in terms of Pauli matrices.
    
    what : string, either vals of diffs. Determines the returned objects. In the case of "vals", both eigenfrequencies are returned, while in the case 
        of "diff", only the absolute value difference of the two eigenfrequencies is returned. The default is "vals".

    Returns
    -------
    if what = "vals": list of 2 floats. The eigenfrequencies of the system.
    
    if what = "diff": float. The absolute value difference of the two eigenfrequencies of the system.

    """
    
    #calculate the eigenvalues using the dvector 
    val1 = np.sqrt(dvec[0] - np.sqrt(dvec[1]**2 + dvec[2]**2))
    val2 = np.sqrt(dvec[0] + np.sqrt(dvec[1]**2 + dvec[2]**2))
 
    #return the corresponding value
    if what == "vals":
        return np.array([val1, val2])
    
    elif what == "diff":
        return np.abs(val2-val1)

def freq_spectrum(dvec_container, style = "heatmap"):
    """
    Function that is used to calculate the eigenfrequency spectrum of the system by using the dvectors. 

    Parameters
    ----------
    dvec_container : matrix of size N x M x 3 of floats.
    
    style : string, either heatmap or surface. Determines the style of the plot to be created. The default is "heatmap".

    Returns
    -------
    if style = "heatmap": matrix of floats. The matrix that contains the eigenfrequencies of the system. 
    
    if style = "surface": 2 matrices of floats. The matrices that contain the two eigenfrequencies of the system. The first matrix contains the lower eigenvalues,
        while the second one contains the higher eigenfrequencies. 

    """
    
    #obtain the size of the dvector container
    len_y = len(dvec_container)
    len_x = len(dvec_container[0])
    
    #calculate the important quantity depending on the style parameter
    if style == "heatmap":
        
        #create container for the absolute value difference of the eigenfrequencies and reshape the container
        freqdiffs = np.zeros( len_y * len_x ).reshape(( len_y, len_x ))
        
        #iterate through the angles
        for idx_y in range(len_y):
            for idx_x in range(len_x):
                
                #get the dvector
                dvec = dvec_container[idx_y, idx_x, :]
                
                #obtain the difference of the frequency values
                diff = eigenfreqs(dvec = dvec, what = "diff")
                
                #add the frequency difference to the container
                freqdiffs[idx_y, idx_x] = diff
        
        #return absolute value difference of the eigenfrequencies
        return freqdiffs
    
    elif style == "surface":
        
        #create container for the frequencies and reshape them
        freq1s = np.zeros( len_y * len_x ).reshape(( len_y, len_x ))
        freq2s = np.zeros( len_y * len_x ).reshape(( len_y, len_x ))
        
        #iterate through the angles
        for idx_y in range(len_y):
            for idx_x in range(len_x):
                
                #get the dvector
                dvec = dvec_container[idx_y, idx_x, :]
                
                #get both frequencies
                ws = eigenfreqs(dvec = dvec, what = "vals")
                
                #sort the frequency values
                ws.sort()
                
                #add the values to the corresponding container
                freq1s[idx_y, idx_x] = ws[0]
                freq2s[idx_y, idx_x] = ws[1]
                
        #return the eigenfrequency matrices
        return freq1s, freq2s
    
def freq_spectrum_plotter(dvec_container, a1s, a2s, style = "heatmap", elev = 10, azim = 40, plot_filename = None, cbar_visible = False):
    """
    

    Parameters
    ----------
    dvec_container : matrix of size N x M x 3 of floats. The vector stored at dvec_container[i,j,:] corresponds to the dvector at angles alpha1=a1s[j] and alpha2=a2s[i].  
    
    a1s : list of floats. The possible values of the angle alpha1.
    
    a2s : list of floats. The possible values of the angle alpha2.
    
    style : string, either heatmap or surface. Determines the style of the plot to be created. The default is "heatmap".

    elev : float. The elevation for the 3D plot. The default value is 10.
    
    azim : float. The azimuth angle for the 3D plot. The default value is 40.
    
    plot_filename : string. The name with which the figure is to be saved.
    
    cbar_visible : bool. If True, then show the colorbar for the figure, otherwise do not show the colorbar.

    Returns
    -------
    fig : matplotlib figure. The figure that was created from the data.

    """
    
    #create the meshgrid for the parameter space
    A1, A2 = np.meshgrid(a1s, a2s)
    
    #make the plot depending on the style parameter
    if style == "heatmap":
        
        #obtain the spectrum (the difference between the eigenfrequencies)
        freqdiffs = freq_spectrum(dvec_container = dvec_container, style = "heatmap")
        
        #create the figure
        fig = plt.figure(figsize = (5,5))
        
        #add labels to the axis
        plt.xlabel(r"$\alpha$", fontsize = 16)
        plt.ylabel(r"$\beta$", fontsize = 16)
        
        #plot the data 
        #NOTE: the first index needs to be reversed
        ims = plt.imshow(freqdiffs[::-1,:], extent = [-np.pi, np.pi, -np.pi, np.pi])
    
        #add nice tick labels to the axis
        tickvals = [-np.pi, -np.pi/2,0, np.pi/2, np.pi]
        ticklabels = [r"$-\pi$", r"$- \pi /2$", r"$0$", r"$\pi/2$", r"$\pi$"]
        
        plt.xticks(tickvals, ticklabels, fontsize = 10)
        plt.yticks(tickvals, ticklabels, fontsize = 10)
            
        #create colorbar for the figure if needed
        if cbar_visible == True:
            #add a colorbar with labels
            cbar = plt.colorbar(ims)
            cbar.set_label(r"$|\omega_1-\omega_2|$", fontsize = 16)
        else:
            pass
    
    elif style == "surface":
        
        #create the spectrum (both eigenfrequencies)
        freq1s, freq2s = freq_spectrum(dvec_container = dvec_container, style = "surface")
        
        #create a 3d plot       
        fig = plt.figure(figsize = (5,5))
        ax = fig.add_subplot(projection = '3d')
        
        #add labels to the axis
        plt.xlabel(r"$\alpha$", fontsize = 16)
        plt.ylabel(r"$\beta$", fontsize = 16)
        
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r"$\omega$", fontsize = 16, rotation = 90)

        #normalize the data for the colormap
        vmin = np.min( [ np.min( freq1s[:][:] ) ,np.min( np.min( freq2s[:][:] ) ) ] )
        vmax = np.max( [ np.max( freq1s[:][:] ) ,np.max( np.max( freq2s[:][:] ) ) ] )
        norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
    
        #create the surface plots
        surf1 = ax.plot_surface( X = A1, Y = A2, Z = freq1s[:,:], cmap = cm.viridis,
                               linewidth = 1, antialiased = False, alpha = 0.8, facecolors = plt.cm.viridis(norm(freq1s)) )
        
        surf2 = ax.plot_surface( X = A1, Y = A2, Z = freq2s[:,:], cmap = cm.viridis,
                               linewidth = 1, antialiased = False, alpha = 0.8, facecolors = plt.cm.viridis(norm(freq2s)) )
        
        #create colorbar if needed
        if cbar_visible == True:
            m = cm.ScalarMappable(cmap = plt.cm.viridis, norm = norm)
            m.set_array([])
            plt.colorbar(m, shrink = 0.6, aspect = 20)
        
        else:
            pass
        
        #add nice tick labels to the axis
        tickvals = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        ticklabels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
        
        plt.xticks(tickvals, ticklabels)
        plt.yticks(tickvals, ticklabels)
        
        #set an extent for the z axis
        ax.set_zlim([1.2,1.5])
        
        #change point of view angle
        ax.view_init(elev = elev, azim = azim)
    
    #save if needed
    if plot_filename == None:
        pass
    else: 
        save_file = "../figures/" + plot_filename
        plt.savefig(save_file, dpi = 1200, bbox_inches = "tight")
    
    #return the figure
    return fig

def phase_func(dvec_container):
    """
    Function that is used to calculate the phases that correspond to the dvector data.

    Parameters
    ----------
    dvec_container : matrix of size N x M x 3 of floats.

    Returns
    -------
    phases : matrix of size N x M of floats. The matrix that contains the phases associated to the d vectors. 

    """
    
    #find the size of the container
    ylen = len(dvec_container)
    xlen = len(dvec_container[0])
    
    #define a container for the phases
    phases = np.zeros( xlen * ylen ).reshape((ylen, xlen))

    #iterate through the whole discrete parameter space
    for i in range(ylen):
        for j in range(xlen):
            
            #define the phase based on the angle of the d vector 
            phases[i, j] = np.arctan2( dvec_container[i, j, 2], dvec_container[i, j, 1] )
            
    #return the phases
    return phases

def plaquette_corners(i, j):
    """
    Function that is used to define the corners of the plaquette characterised by the indices i and j.
    NOTE: the convetion here is that the loop runs in anti-clockwise direction due to the representation of the entries in python arrays. 

    Parameters
    ----------
    i : integer. The column index of entry.
    
    j : integer. The row index of entry.

    Returns
    -------
    b : list of 2 integers. The coordinates of the neighbour of (i,j) on the right.
    
    c : list of 2 integers. The coordinates of the neighbour of (i,j+1) on the top.
    
    d : list of 2 integers. The coordinates of the neighbour of (i+1,j+1) on the left.

    """
    b = [i, (j + 1)] #on the right
    c = [(i + 1), (j + 1)] #up
    d = [(i + 1), j] #left
    
    #return the corners of the plaquette
    return b, c, d

def vort_func(phases):
    """
    Function that is used to calculate the vorticity of the d vector field. 

    Parameters
    ----------
    phases : matrix of size N x M of floats. The matrix that contains the phases associated to the d vectors. 

    Returns
    -------
    vorticities : matrix of size (N-1) x (M-1) of floats. The matrix contains the winding of the d vector field in each plaquette.

    """
    
    #find the size of the phases array
    ylen = len(phases)
    xlen = len(phases[0])
    
    #define container for the vorticities
    vorticities =  np.zeros( xlen * ylen ).reshape(( ylen, xlen ))
    
    #iterate through the whole discrete parameter space
    for ia in range(ylen - 1):
        for ja in range(xlen - 1):
            
            #find the corners of the plaquette 
            b, c, d = plaquette_corners(i = ia, j = ja)
            
            #get the phases of the corners
            pa = phases[ia, ja]
            pb = phases[b[0], b[1]]
            pc = phases[c[0], c[1]]
            pd = phases[d[0], d[1]]
            
            #calculate the quantities associated with the edges
            phi_ab = np.angle( np.exp( 1.j * ( pb - pa )) )
            phi_bc = np.angle( np.exp( 1.j * ( pc - pb )) )
            phi_cd = np.angle( np.exp( 1.j * ( pd - pc )) )
            phi_da = np.angle( np.exp( 1.j * ( pa - pd )) )
            
            #find the vorticity of the plaquette defined by the corner (i,j)
            Qa = (phi_ab + phi_bc + phi_cd + phi_da )  / ( 2 * np.pi ) 
            
            #add the vorticitiy to the corresponding point in the discrete parameter space
            vorticities[ia, ja] = Qa
            
    #return the vorticities
    return vorticities

def weyl_points_plot(dvec_container, a1s, a2s, plot_filename = None):
    """
    Function that is used to plot the location of Weyl points in the configuration space. 

    Parameters
    ----------
    dvec_container : matrix of size N x M x 3 of floats. The vector stored at dvec_container[i,j,:] corresponds to the dvector at angles alpha1=a1s[j] and alpha2=a2s[i].  

    a1s : list of floats. The possible values of the angle alpha1.
    
    a2s : list of floats. The possible values of the angle alpha2.
        
    plot_filename : string. The name with which the figure is to be saved. The default is None.

    Returns
    -------
    fig : matplotlib figure. The figure that was created from the data.

    """
    
    #calculate the phases from the dvector container
    phases = phase_func(dvec_container = dvec_container)
    
    #calculate the vorticities from the phases
    vorticies = vort_func(phases = phases)
    
    #calculate the step-size of the finite grid
    da1 = np.abs(a1s[1] - a1s[0]) 
    da2 = np.abs(a2s[1] - a2s[0]) 
    
    #create figure
    fig = plt.figure(figsize = (5,5))
    
    #iterate through the parameter space
    for i in range(len(a2s)):
        for j in range(len(a1s)):
            
            #if there is a point with vorticitiy close to 1 then plot it as a red dot
            if np.abs( vorticies[i,j] - 1 ) < 0.01:
                x = a1s[j] + da1 / 2
                y = a2s[i] + da2 / 2
                plt.scatter(x, y, color = "red", s = 30)
                
            #if there is a point with vorticitiy close to -1 then plot it as a blue dot
            elif np.abs( vorticies[i,j] + 1 ) < 0.01:
                x = a1s[j] + da1 / 2
                y = a2s[i] + da2 / 2
                plt.scatter(x, y, color = "blue", s = 30)
    
            else:
                pass
            
            #NOTE: the WPs will be visualized as if they were located at the center of the plaquette!
            
    #add grid
    plt.grid(True)  
    
    #add labels to axis
    plt.xlabel(r"$\alpha$", fontsize = 16)
    plt.ylabel(r"$\beta$", fontsize = 16)
    
    #add new tick labels
    tickvals = [-np.pi, -np.pi/2,0, np.pi/2, np.pi]
    ticklabels = [r"$-\pi$", r"$- \pi /2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    
    #add the ticks to the figure
    plt.xticks(tickvals, ticklabels, fontsize = 10)
    plt.yticks(tickvals, ticklabels, fontsize = 10)

    #save the figure if the plot_filename variable is not set to zero
    if plot_filename == None:
        pass
    
    else: 
        save_file = "../figures/" + plot_filename
        plt.savefig(save_file, dpi = 1200)    
        
    #return the figure
    return fig