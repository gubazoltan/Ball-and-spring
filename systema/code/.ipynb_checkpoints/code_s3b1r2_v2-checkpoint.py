import numpy as np
import sympy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors

def potprep(r1, r2, k1, k2, k3, l1, l2, l3, d, R, gamma):
    #define the coordinates
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    
    #define the angles
    a1 = sp.Symbol('alpha_1')
    a2 = sp.Symbol('alpha_2')
    
    #define the coordinates of the end of the springs
    #first spring
    r1x = - d / 2 - r1 + r1 * sp.cos( a1 )
    r1y = r1 * sp.sin( a1 )  
    
    #second spring
    r2x = d / 2 + r2 - r2 * sp.cos( a2 ) 
    r2y = r2 * sp.sin( a2 ) 
    
    #third spring
    r3x = R * np.cos( gamma )
    r3y = R * np.sin( gamma )    
    
    #define the elongation of the springs
    dl1 = sp.sqrt( (r1x - x)**2 + (r1y - y)**2 ) - l1
    dl2 = sp.sqrt( (r2x - x)**2 + (r2y - y)**2 ) - l2
    dl3 = sp.sqrt( (r3x - x)**2 + (r3y - y)**2 ) - l3
    
    #define the total spring potential
    U = 0.5 * k1 * (dl1**2) + 0.5 * k2 * (dl2**2) + 0.5 * k3 * (dl3**2)
    
    #return the potential
    return U

def myfunc(cords, l):
    #get the coordinates
    x, y = cords[0], cords[1]
    
    #evaluate the function
    val = l(x, y)
    
    #return the value
    return val

def mincords(pot, alpha1, alpha2, x0, y0):
    #evaluate the potential for the two angles
    pot_at_a1a2 = pot.evalf(subs = {'alpha_1' : alpha1, 'alpha_2' : alpha2})

    #make a lambda function out of the expression which takes the x and y values and evaluate the potential
    #this is needed so that one can find the minima of the potential using scipy.optimize
    evaled_pot = sp.utilities.lambdify(['x', 'y'], pot_at_a1a2)
    
    #define the initial guess for the coordinates that minimize the potential
    guess = [x0, y0]
    
    #find the coordinates that minimize the potential energy
    min_cords = minimize(myfunc, guess, args = (evaled_pot))
    
    minx, miny = min_cords.x
    #return the coordinates of the minima
    return minx, miny, pot_at_a1a2

def dvector(pot, alpha1, alpha2):
    
    minx, miny, pot_at_a1a2 = mincords(pot = pot, alpha1 = alpha1, alpha2 = alpha2, x0 = 0, y0 = 0)
    
    #find first order derivatives of the potential
    dx = sp.diff(pot_at_a1a2, 'x')
    dy = sp.diff(pot_at_a1a2, 'y')
    
    #then find the second order derivates
    dxdx = sp.diff(dx, 'x')
    dxdy = sp.diff(dx, 'y')
    dydy = sp.diff(dy, 'y')
    
    #evaluate the second order derivatives at the minima of the potential
    dxdx_evaled = dxdx.evalf(subs = {'x' : minx, 'y' : miny})
    dxdy_evaled = dxdy.evalf(subs = {'x' : minx, 'y' : miny})
    dydy_evaled = dydy.evalf(subs = {'x' : minx, 'y' : miny})
    
    d0 = (dxdx_evaled + dydy_evaled) / 2
    d1 = dxdy_evaled 
    d3 = (dxdx_evaled - dydy_evaled) / 2
    
    #define the d vector
    dvec = np.array([d0, d1, d3], dtype = "float64")
    
    #return the d vector
    return dvec

def dvector_field(pot, a1s, a2s, filename = None):
    
    dvec_container = np.zeros( len(a1s) * len(a2s) * 3 ).reshape( ( len(a2s), len(a1s), 3 ) )
    
    for idx_y, alpha2 in enumerate(a2s):
        for idx_x, alpha1 in enumerate(a1s): 
            dvec = dvector(pot = pot, alpha1 = alpha1, alpha2 = alpha2)
            
            dvec_container[idx_y, idx_x, :] = dvec[:]
            
    if filename == None:
        pass
    
    else:
        #save the frequency differences
        save_file1 = "../datas/" + filename + "_1"
        save_file2 = "../datas/" + filename + "_2"
        save_file3 = "../datas/" + filename + "_3"
        
        np.savetxt(fname = save_file1, X = dvec_container[:,:,0])
        np.savetxt(fname = save_file2, X = dvec_container[:,:,1])
        np.savetxt(fname = save_file3, X = dvec_container[:,:,2])
           
    return dvec_container
    
    
def eigenfreqs(dvec, what = "vals"):
    
    val1 = np.sqrt(dvec[0] - np.sqrt(dvec[1]**2 + dvec[2]**2))
    val2 = np.sqrt(dvec[0] + np.sqrt(dvec[1]**2 + dvec[2]**2))
 
    if what == "vals":
        return np.array([val1, val2])
    
    elif what == "diff":
        return np.abs(val2-val1)
    
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def freq_spectrum(dvec_container, style = "heatmap"):
    
    len_y = len(dvec_container)
    len_x = len(dvec_container[0])
    
    #make the plot depending on the style parameter
    if style == "heatmap":
        
        #create container for the frequencies
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
        
        return freqdiffs
    
    elif style == "surface":
        
        #create container for the frequencies and reshape them
        freq1s = np.zeros( len_y * len_x ).reshape(( len_y, len_x ))
        freq2s = np.zeros( len_y * len_x ).reshape(( len_y, len_x ))
        
        #iterate through the angles
        for idx_y in range(len_y):
            for idx_x in range(len_x):
                
                dvec = dvec_container[idx_y, idx_x, :]
                
                #get both of the frequencies
                ws = eigenfreqs(dvec = dvec, what = "vals")
                
                #sort the frequency values
                ws.sort()
                
                #add the values to the corresponding container
                freq1s[idx_y, idx_x] = ws[0]
                freq2s[idx_y, idx_x] = ws[1]
                
        return freq1s, freq2s
    
def freq_spectrum_plotter(dvec_container, a1s, a2s, style = "heatmap", elev = 10, azim = 40, plot_filename = None, cbar_visible = False):
    
    #create the meshgrid for the parameter space
    A1, A2 = np.meshgrid(a1s, a2s)
    
    #make the plot depending on the style parameter
    if style == "heatmap":
        
        #rename the data
        freqdiffs = freq_spectrum(dvec_container = dvec_container, style = "heatmap")
        
        #create the figure
        fig = plt.figure(figsize = (5,5))
        
        plt.xlabel(r"$\alpha$", fontsize = 16)
        plt.ylabel(r"$\beta$", fontsize = 16)
        
        #plot with this extent
        ims = plt.imshow(freqdiffs[::-1,:], extent = [-np.pi, np.pi, -np.pi, np.pi])
    
        tickvals = [-np.pi, -np.pi/2,0, np.pi/2, np.pi]
        ticklabels = [r"$-\pi$", r"$- \pi /2$", r"$0$", r"$\pi/2$", r"$\pi$"]
        
        plt.xticks(tickvals, ticklabels, fontsize = 10)
        plt.yticks(tickvals, ticklabels, fontsize = 10)
            
        if cbar_visible == True:
            #add a colorbar with labels
            cbar = plt.colorbar(ims)
            cbar.set_label(r"$|\omega_1-\omega_2|$", fontsize = 16)
        else:
            pass
        
    elif style == "surface":
        
        #expand the data
        freq1s, freq2s = freq_spectrum(dvec_container = dvec_container, style = "surface")
        
        #create 3d surface plot       
        fig = plt.figure(figsize = (5,5))
        ax = fig.add_subplot(projection = '3d')
        
        plt.xlabel(r"$\alpha$", fontsize = 16)
        plt.ylabel(r"$\beta$", fontsize = 16)
    
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(r"$\omega$", fontsize = 16, rotation = 90)

        #find values for normalization
        vmin = np.min( [ np.min( freq1s[:][:] ) ,np.min( np.min( freq2s[:][:] ) ) ] )
        vmax = np.max( [ np.max( freq1s[:][:] ) ,np.max( np.max( freq2s[:][:] ) ) ] )
        norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
    
        #define the surface plots
        surf1 = ax.plot_surface( X = A1, Y = A2, Z = freq1s[:,:], cmap = cm.viridis,
                               linewidth = 1, antialiased = False, alpha = 0.8, facecolors = plt.cm.viridis(norm(freq1s)) )
        
        surf2 = ax.plot_surface( X = A1, Y = A2, Z = freq2s[:,:], cmap = cm.viridis,
                               linewidth = 1, antialiased = False, alpha = 0.8, facecolors = plt.cm.viridis(norm(freq2s)) )
        
        if cbar_visible == True:
            
            #make colorbar
            m = cm.ScalarMappable(cmap = plt.cm.viridis, norm = norm)
            m.set_array([])
            plt.colorbar(m, shrink = 0.6, aspect = 20)
        
        else:
            pass
        
       
        tickvals = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        ticklabels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
        
        plt.xticks(tickvals, ticklabels)
        plt.yticks(tickvals, ticklabels)
        
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
    
    #find the shape of the arrays
    ylen = len(dvec_container)
    xlen = len(dvec_container[0])
    
    #define a container for the phases
    phases = np.zeros( xlen * ylen ).reshape((ylen, xlen))

    #iterate through the whole discrete parameter space
    for i in range(ylen):
        for j in range(xlen):
            #define the phase based on the angle of the d vector 
            phases[i, j] = np.arctan2( dvec_container[i, j, 2], dvec_container[i, j, 1] )
            
    #and then return the phases
    return phases

def plaquette_corners(i, j):
    b = [i, (j + 1)] #on the right
    c = [(i + 1), (j + 1)] #up
    d = [(i + 1), j] #left
    
    #return the corners of the plaquette
    return b, c, d

def vort_func(phases):
    #find the shape of the arrays
    ylen = len(phases)
    xlen = len(phases[0])
    
    vorticities =  np.zeros( xlen * ylen ).reshape((ylen, xlen))
    
    #iterate through the whole discrete parameter space
    for ia in range(ylen - 1):
        for ja in range(xlen - 1):
            #need to find the vorticity of the plaquette
            #start by calculating the edges
            #find the corners of the plaquette
            b, c, d = plaquette_corners(i = ia, j = ja)
            
            #get the phases of the corners
            pa = phases[ia, ja]
            pb = phases[b[0], b[1]]
            pc = phases[c[0], c[1]]
            pd = phases[d[0], d[1]]
            
            #find the edge phases
            phi_ab = np.angle( np.exp( 1.j * ( pb - pa )) )
            phi_bc = np.angle( np.exp( 1.j * ( pc - pb )) )
            phi_cd = np.angle( np.exp( 1.j * ( pd - pc )) )
            phi_da = np.angle( np.exp( 1.j * ( pa - pd )) )
            
            #find the vorticity of the plaquette defined by the corner a
            Qa = (phi_ab + phi_bc + phi_cd + phi_da )  / ( 2 * np.pi ) 
            
            #add the vorticitiy to the corresponding point in the discrete parameter space
            vorticities[ia, ja] = Qa
            
    #return the vorticities
    return vorticities

def weyl_points_plot(dvec_container, a1s, a2s, plot_filename = None):
    
    phases = phase_func(dvec_container = dvec_container)
    
    vorticies = vort_func(phases = phases)
    
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
            
            #note that the WPs will be visualized as if they were located at the center of the plaquette!
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

    if plot_filename == None:
        pass
    
    else: 
        save_file = "../figures/" + plot_filename
        plt.savefig(save_file, dpi = 1200)    
        
    return fig