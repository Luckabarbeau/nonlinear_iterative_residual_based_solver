# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:18:12 2022

@author: lucka
"""

# Importation des modules
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys

def R(x,problem,off_set=np.zeros(0)):
    
    D=1.0
    exp_coef=1.0# really non linear 3.30
    const_coef=0.0
    exp_variable_coef=3.0
    

    if len(off_set) == 0:
        off_set = np.zeros(len(x))
    if(problem.dim==2):
        # Initialize the result array
        r = np.zeros_like(x)
    
        # Compute grid spacings
        dx = (1 - 0) / (problem.nx - 1)
        dy = (1 - 0) / (problem.ny - 1) if problem.dim > 1 else 0
        
    
        # Pre-compute common factors
        dx2 = dx ** 2
        dy2 = dy ** 2
    
        # Indices for all points
        indices = np.arange(x.size)
        
        # Exclude boundary points for now by focusing on interior points
        # Adjust these conditions based on your problem's boundary definitions
        interior_mask = (indices % problem.ny != 0) & \
                        (indices % problem.ny != problem.ny - 1) & \
                        (indices >= problem.ny) & \
                        (indices < x.size - problem.ny)
        
        boundary_mask = ~interior_mask
        r[boundary_mask] = (0.0-x[boundary_mask])/dx/dy     
                    
        # Compute indices for neighbors
        left = indices[interior_mask] - problem.ny
        right = indices[interior_mask] + problem.ny
        up = indices[interior_mask] - 1
        down = indices[interior_mask] + 1
    
        # Apply the equation for interior points
        r[interior_mask] = D * (-2*x[interior_mask]/dx2-2*x[interior_mask]/dy2 + x[left]/dx2+ x[right]/dx2 + x[up]/dy2 + x[down]/dy2)+const_coef+exp_coef*np.exp(exp_variable_coef*x[interior_mask])
        # Handle boundary points separately, if necessary
    if(problem.dim==3):
        # Initialize the result array
        r = np.zeros_like(x)
    
        # Compute grid spacings
        dx = (1 - 0) / (problem.nx - 1)
        dy = (1 - 0) / (problem.ny - 1)
        dz = (1 - 0) / (problem.nz - 1)
    
        # Pre-compute common factors
        dx2 = dx ** 2
        dy2 = dy ** 2
        dz2 = dz ** 2
    
        # Indices for all points
        indices = np.arange(x.size)

        interior_mask = (indices-(indices-indices%(problem.nx*problem.ny))>problem.ny) & \
                        (indices-(indices-indices%(problem.nx*problem.ny))<problem.nx*(problem.ny-1)) & \
                        (indices % problem.ny != 0) & \
                        (indices % problem.ny != problem.ny - 1)  & \
                        (indices>=(problem.nx*problem.ny))  & \
                        (indices<=(problem.nz-1)*(problem.ny*problem.nx))
                       
        # Exclude boundary points for now by focusing on interior points
        # Adjust these conditions based on your problem's boundary definitions
        boundary_mask = ~interior_mask
        r[boundary_mask] = (0.0-x[boundary_mask])/dx/dy                 
        # Compute indices for neighbors
        left = indices[interior_mask] - problem.ny
        right = indices[interior_mask] + problem.ny
        up = indices[interior_mask] - 1
        down = indices[interior_mask] + 1
        front = indices[interior_mask] - problem.nx*problem.ny
        back = indices[interior_mask] + problem.nx*problem.ny
        
        # Apply the equation for interior points
        r[interior_mask] = D * (-2*x[interior_mask]/dx2-2*x[interior_mask]/dy2-2*x[interior_mask]/dz2 + x[left]/dx2+ x[right]/dx2 + x[up]/dy2 + x[down]/dy2 + x[front]/dz2 + x[back]/dz2)+const_coef+exp_coef*np.exp(exp_variable_coef*x[interior_mask])
    
        # Handle boundary points separately, if necessary
    return r- off_set

def bilinear_interpolate(coarse_grid, nx2, ny2):
    ny, nx = coarse_grid.shape  # For 2D coarse_grid
    
    # Generate grid for destination coordinates
    x = np.linspace(0, nx - 1, nx2)
    y = np.linspace(0, ny - 1, ny2)
    
    # Calculate the coordinates of the four neighbors
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    
    # Calculate interpolation weights
    wx = x - x0
    wy = y - y0
    
    # Interpolation
    # Get values from the four neighbors
    top_left = coarse_grid[y0, :][:, x0]
    top_right = coarse_grid[y0, :][:, x1]
    bottom_left = coarse_grid[y1, :][:, x0]
    bottom_right = coarse_grid[y1, :][:, x1]
    
    # Reshape for broadcasting
    wx = wx.reshape(1, -1)
    wy = wy.reshape(-1, 1)
    
    # Perform interpolation
    interpolated = top_left * (1 - wx) * (1 - wy) + top_right * wx * (1 - wy) + bottom_left * (1 - wx) * wy + bottom_right * wx * wy

    return interpolated


def trilinear_interpolate(coarse_grid, nx2, ny2, nz2):
    nz, ny, nx = coarse_grid.shape
    # Generate the new grid coordinates
    x_new = np.linspace(0, nx - 1, nx2)
    y_new = np.linspace(0, ny - 1, ny2)
    z_new = np.linspace(0, nz - 1, nz2)
    
    # Calculate the floor of these coordinates to find the "low" indices
    x0 = np.floor(x_new).astype(int)
    y0 = np.floor(y_new).astype(int)
    z0 = np.floor(z_new).astype(int)
    
    # Ensure "high" indices are within the grid bounds
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    z1 = np.clip(z0 + 1, 0, nz - 1)
    
    # Calculate the interpolation weights
    x_weight = (x_new - x0).reshape(1, 1, -1)
    y_weight = (y_new - y0).reshape(1, -1, 1)
    z_weight = (z_new - z0).reshape(-1, 1, 1)
    
    # Retrieve values at the corner points for the entire grid
    c000 = coarse_grid[z0[:, None, None], y0[None, :, None], x0[None, None, :]]
    c001 = coarse_grid[z0[:, None, None], y0[None, :, None], x1[None, None, :]]
    c010 = coarse_grid[z0[:, None, None], y1[None, :, None], x0[None, None, :]]
    c011 = coarse_grid[z0[:, None, None], y1[None, :, None], x1[None, None, :]]
    c100 = coarse_grid[z1[:, None, None], y0[None, :, None], x0[None, None, :]]
    c101 = coarse_grid[z1[:, None, None], y0[None, :, None], x1[None, None, :]]
    c110 = coarse_grid[z1[:, None, None], y1[None, :, None], x0[None, None, :]]
    c111 = coarse_grid[z1[:, None, None], y1[None, :, None], x1[None, None, :]]
    
    # Perform trilinear interpolation across the entire grid at once
    fine_grid = (c000 * (1 - x_weight) * (1 - y_weight) * (1 - z_weight) + 
                 c001 * x_weight * (1 - y_weight) * (1 - z_weight) + 
                 c010 * (1 - x_weight) * y_weight * (1 - z_weight) + 
                 c011 * x_weight * y_weight * (1 - z_weight) + 
                 c100 * (1 - x_weight) * (1 - y_weight) * z_weight + 
                 c101 * x_weight * (1 - y_weight) * z_weight + 
                 c110 * (1 - x_weight) * y_weight * z_weight + 
                 c111 * x_weight * y_weight * z_weight)

    return fine_grid



class Problem():
    """
    A class to model a problem space that can be 1D, 2D, or 3D.

    Attributes:
        dim (int): The dimensionality of the problem space (1, 2, or 3).
        nx (int): The size of the problem space in the x-direction.
        ny (int): The size of the problem space in the y-direction, relevant for 2D and 3D problems.
        nz (int): The size of the problem space in the z-direction, relevant for 3D problems.
        size (int): The total size of the problem space, calculated based on `dim`, `nx`, `ny`, and `nz`.
    """
    def __init__(self,dim=2,nx=100,ny=100,nz=100):
        self.dim=dim
        self.nx=nx
        self.ny=ny
        self.nz=nz
        if dim==1:
            self.size=int(self.nx)
        if dim==2:
            self.size=int(self.nx*self.nx)
        if dim==3:
             self.size=int(self.nx*self.nx*self.nz)
    def clone(self):
        """
        Creates a deep copy of the problem instance, including its dimensions and size.
        """
        return Problem(self.dim,self.nx,self.ny,self.nz)
    def update(self):
        """
        Updates the size of the problem space based on the current dimensions.
        This should be called if any of the dimensions or their sizes change after initialization.
        """
        if self.dim==1:
            self.size=int(self.nx)
        if self.dim==2:
            self.size=int(self.nx*self.nx)
        if self.dim==3:
             self.size=int(self.nx*self.nx*self.nz)


class Preconditionner_option():
    """
    A class representing the options for configuring a preconditioner used in solving
    systems of equations iteratively. It allows setting various parameters that influence the
    behavior and performance of preconditioned iterative solvers.

    Attributes:
        level (int): The number of derefinement apply in the  GMG preconditionner.
        alpha (float): A parameter used to define the perturbation vector size use in the solver.
        iterations_per_level (int): Number of iterations performed on fine mesh for smoothing.
        frequency_of_residual_direct_evaluation (int): How often to directly compute the residual
            instead of estimating it.
        tol (float): The relative tolerance for convergence. The solver stops when the residual norm is
            below this threshold.
        max_iterations (int): The maximum number of iterations allowed for the solver.
        max_krylov_vectors (int): The maximum number of Krylov vectors to use in Krylov-subspace methods.
        verbosity (bool): If True, the solver will output detailed progress information.
    """
    def __init__(self,level=2,alpha=1,iterations_for_smoothing=10,frequency_of_residual_direct_evaluation=10,tol=1e-6,max_iterations=1000,max_krylov_vectors=1000,verbosity=True):
        self.level=level
        self.alpha=alpha
        self.iterations_for_smoothing=iterations_for_smoothing
        self.frequency_of_residual_direct_evaluation=frequency_of_residual_direct_evaluation
        self.tol=tol
        self.max_iterations=max_iterations
        self.max_krylov_vectors=max_krylov_vectors
        self.verbosity=verbosity
    def clone(self):
        """
        Creates a deep copy of the current instance, preserving the configuration of the preconditioner options.
        """
        return Preconditionner_option(self.level,self.alpha,self.iterations_for_smoothing,self.frequency_of_residual_direct_evaluation,self.tol,self.max_iterations,self.max_krylov_vectors,self.verbosity)

class Preconditioner_GMG():
    """
    A class implementing a Geometric Multigrid (GMG) preconditioner. This preconditioner
    is designed to efficiently find adequate correction vector arising in discretized partial
    differential equations, especially those that are large and sparse.

    Attributes:
        solution_on_coarses_level (numpy.ndarray): A cache of the solution on the coarsest level,
            used to accelerate the convergence of the solver.
    """
    
    
    solution_on_coarses_level=np.zeros((0,0))

    
    def apply_preconditioner(self, x_0, r, problem, solver_options, off_set):
        """
        Applies the GMG preconditioner to the given residual vector to produce a correction vector.
    
        Parameters:
            x_0 (numpy.ndarray): The initial guess for the solution vector.
            r (numpy.ndarray): The current residual vector.
            problem (Problem): An instance of the problem being solved, including its dimensions.
            solver_options: Configuration options for the solver, including tolerance and verbosity.
            off_set (numpy.ndarray): An offset vector used in the preconditioning process.
    
        Returns:
            numpy.ndarray: A correction vector to be applied to the current solution estimate.
        """
        # Copy of the initial guess to avoid modifying it directly
        x = np.copy(x_0)  
        # Save the original problem dimensions for later use
        nx_0, ny_0, nz_0 = problem.nx, problem.ny, problem.nz
    
        # Temporarily mute verbosity for coarse level solver
        solver_options.verbosity = False
    
        # Iteratively coarsen the field down to the desired level or the smallest acceptable mesh size
        for i in range(solver_options.level + 1):
            # Check if the current iteration is at the coarsest level
            if i == solver_options.level:
                # At the coarsest level, we might adjust solver tolerance or other parameters
                solver_options.tol = solver_options.tol  # This line appears redundant but may be a placeholder for adjustments
                problem.update()  # Ensure problem dimensions and other attributes are up-to-date
                # Decision branch based on whether the problem size is above a threshold (e.g., nx > 8)
                if problem.nx > 8:
                    # If no solution has been cached for this level, solve the problem at this level
                    if self.solution_on_coarses_level.size == 0:
                        x_0_coarse = x  # Save the current guess before solving
                        # Solve the problem at the current level, with a preconditioner (indicated by True/False)
                        x, R_norm = solve(x, problem, solver_options, True, off_set)
                        # Cache the solution for future use
                        self.solution_on_coarses_level = np.copy(x)
                    else:
                        # If a solution is already cached, use it to determine the coarse correction
                        x_0_coarse, R_norm = solve(x, problem, solver_options, True, r + off_set)
                        x = self.solution_on_coarses_level
                    # Calculate the correction vector on the coarse level
                    v_coarse = x - x_0_coarse
                else:
                    # Similar logic applies if the problem size is below the threshold, but without preconditioning
                    if self.solution_on_coarses_level.size == 0:
                        x_0_coarse = x
                        x, R_norm = solve(x, problem, solver_options, False, off_set)
                        self.solution_on_coarses_level = np.copy(x)
                    else:
                        x_0_coarse, R_norm = solve(x, problem, solver_options, False, r + off_set)
                        x = self.solution_on_coarses_level
                    v_coarse = x - x_0_coarse
            # If not at the coarsest level, coarsen the fields for the next iteration
            if i < solver_options.level and problem.nx / 2 >4:
                if problem.dim == 2:
                    # Coarsen the field in 2D using bilinear interpolation
                    nx2, ny2 = int(problem.nx / 2), int(problem.ny / 2)
                    fine_grid = r.reshape((problem.nx, problem.ny), order='F')
                    coarse_grid = bilinear_interpolate(fine_grid, nx2, ny2)
                    r = coarse_grid.flatten(order='F')
                    fine_grid = x.reshape((problem.nx, problem.ny), order='F')
                    coarse_grid = bilinear_interpolate(fine_grid, nx2, ny2)
                    x = coarse_grid.flatten(order='F')
                    fine_grid = off_set.reshape((problem.nx, problem.ny), order='F')
                    coarse_grid = bilinear_interpolate(fine_grid, nx2, ny2)
                    off_set = coarse_grid.flatten(order='F')
                    problem.nx, problem.ny = nx2, ny2
                elif problem.dim == 3:
                    # Coarsen the field in 3D using trilinear interpolation
                    nx2, ny2, nz2 = int(problem.nx / 2), int(problem.ny / 2), int(problem.nz / 2)
                    fine_grid = r.reshape((problem.nx, problem.ny, problem.nz), order='F')
                    coarse_grid = trilinear_interpolate(fine_grid, nx2, ny2, nz2)
                    r = coarse_grid.flatten(order='F')
                    fine_grid = x.reshape((problem.nx, problem.ny, problem.nz), order='F')
                    coarse_grid = trilinear_interpolate(fine_grid, nx2, ny2, nz2)
                    x = coarse_grid.flatten(order='F')
                    fine_grid = off_set.reshape((problem.nx, problem.ny, problem.nz), order='F')
                    coarse_grid = trilinear_interpolate(fine_grid, nx2, ny2, nz2)
                    off_set = coarse_grid.flatten(order='F')
                    problem.nx, problem.ny, problem.nz = nx2, ny2, nz2
                    
        # After coarsening, refine the correction vector back to the original problem dimensions
        if problem.dim == 2:
            # Refine in 2D
            coarse_grid = v_coarse.reshape((problem.nx, problem.ny), order='F')
            fine_grid = bilinear_interpolate(coarse_grid, nx_0, ny_0)
            v_coarse = fine_grid.flatten(order='F')
            problem.nx, problem.ny = nx_0, ny_0
        elif problem.dim == 3:
            # Refine in 3D
            coarse_grid = v_coarse.reshape((problem.nx, problem.ny, problem.nz), order='F')
            fine_grid = trilinear_interpolate(coarse_grid, nx_0, ny_0, nz_0)
            v_coarse = fine_grid.flatten(order='F')
            problem.nx, problem.ny, problem.nz = nx_0, ny_0, nz_0
    
        return v_coarse / np.linalg.norm(v_coarse)  # Normalize the correction vector before returning

      
      



def solve(x_0,problem,solver_options,enable_preconditionner=True,off_set=np.zeros(0)):
    # Start timer
    total_time_start=time.time()
    # Check if an off_set is defined if not then create a vector of zeros of the right size
    if len(off_set)==0:
        off_set=np.zeros(len(x_0))
    # Initialized the structure and residual used in the solver
    x=np.copy(x_0)
    r=R(x_0,problem,off_set)
    R_norm=np.linalg.norm(r)
    R_norm_0=R_norm
    if(solver_options.verbosity):
        print("Initial residual = "+str(R_norm_0))
    non_linear_index=0
    previous_d=[]
    previous_dr=[]
    previous_dr_dot_product=[]

    # Initialized the preconditionner object
    preconditioner=Preconditioner_GMG()
    
    # Initialized a bunch of timer counter
    time_spent_on_perturbation_residual=0
    time_spent_on_residual=0
    time_spent_on_preconditioner=0
    time_spent_on_alphas_matrix_assembly=0
    time_spent_on_alphas_matrix_solve=0
    time_spent_on_solution_update=0
    
    # Set the size of the pertubation vector with alpha and the tolerance
    alpha=solver_options.alpha
    max_tol=max(1e-12,solver_options.tol*R_norm_0) 
    #Initialized the iteation counter
    i=0
    while R_norm>max_tol and i<solver_options.max_iterations:
        
        start_time = time.time()
        # define new direction based on preconditioner. We apply the GMG preconditioner once every few iterations. For the other iteration we smooth using no preconditionning.
        if((i%int(solver_options.iterations_for_smoothing)==0 ) and enable_preconditionner==True):
            d=preconditioner.apply_preconditioner(x,r,problem.clone(),solver_options.clone(),off_set)  
        else:
            d=r/np.linalg.norm(r)
        end_time = time.time() 
        # log the time pass on the preconditioner
        time_spent_on_preconditioner+=end_time-start_time
        
        start_time = time.time()
        # Check if the krylov vector space should be reinitialized either because the non linear index is to high which indicate that previous direction wont be usefull in the evaluation of the correction. Or that the maximum number of vector as been reach.
        if(i%solver_options.max_krylov_vectors==0 or non_linear_index>0.5):
            # Clear the previous direction ,the variation vector and the variation vector dot product matrix 
            previous_d=[]
            previous_dr=[]
            previous_dr_dot_product=[]
            # Add the new direction
            previous_d.append(d)
            # perturbed the Residual in the direction of d
            r_dx=R(x+previous_d[-1]*alpha,problem,off_set)
            # Calculate the variation vector
            dr=r_dx-r
            # reinitalized the non_linear_index
            non_linear_index=0
        else:
            # Orthogonalized the correction vector with previous vectors
            for j in range(len(previous_d)):
                d=d-np.dot(d,previous_d[j])*previous_d[j]/np.linalg.norm(previous_d[j])**2
                d=d/np.linalg.norm(d)
            # Add the new direction
            previous_d.append(d)
            # perturbed the Residual in the direction of d
            r_dx=R(x+previous_d[-1]*alpha,problem,off_set)
            # Calculate the variation vector
            dr=r_dx-r 
        
        end_time = time.time()
        # log time pass on the evaluation of the residual perturbation
        time_spent_on_perturbation_residual+=end_time-start_time
        
        # Append the new variation vector with the previous ones
        previous_dr.append(dr)
        
        start_time = time.time()
        # Initialized the variation vector dot product matrix and Right hand side (RHS) used to evaluate the alphas.
        A=np.zeros((len(previous_dr),len(previous_dr)))
        rhs=np.zeros(len(previous_dr))
        for j in range(len(previous_dr)):
            for k in range(j+1):
                if(j==len(previous_dr)-1):
                    dot_product=np.dot(previous_dr[j],previous_dr[k])
                    A[j][k]=dot_product
                    A[k][j]=dot_product
                    # Evaluate the dot product of all variation vector with the residual to account for small non linearities
                    rhs[k]=np.dot(previous_dr[k],r)
                    if(k==0):
                        previous_dr_dot_product.append([])
                    previous_dr_dot_product[j].append(dot_product)
                else:
                    A[j][k]=previous_dr_dot_product[j][k]
                    A[k][j]=previous_dr_dot_product[j][k]
        end_time = time.time()
        # Log the time pass on the assembly of the matrix and RHS
        time_spent_on_alphas_matrix_assembly+=end_time -start_time
        
        start_time = time.time()
        # solve the matrix system and find the optimal alphas
        alphas=np.linalg.solve(A, rhs)
        end_time = time.time()
        # Log the time pass on the resolution of the matrix
        time_spent_on_alphas_matrix_solve+=end_time-start_time
        
        start_time = time.time()
        # Update the solution (not vecotrized as it seems longer )
        for j in range(alphas.size):
            x = x - alphas[j] * previous_d[j] * alpha
            r = r - alphas[j] * previous_dr[j]
    
        end_time = time.time()
        # Log the time spend on the update of the solution
        time_spent_on_solution_update+=end_time-start_time
        start_time = time.time()
        # Log the time spend on the update of the solution
        if(i%solver_options.frequency_of_residual_direct_evaluation==0):
            r_exact=R(x,problem,off_set)
            non_linear_index=np.linalg.norm(r_exact-r)/np.linalg.norm(r_exact)
            r=r_exact
            
        R_norm=np.linalg.norm(r)
        end_time = time.time()
        time_spent_on_residual+=end_time-start_time

        i=i+1
        if(solver_options.verbosity):
            if(enable_preconditionner==True and solver_options.verbosity):
                print("Iteration  "+str(i)+" residue = "+ str(R_norm)+" NLI = "+ str(non_linear_index) )
            else:
                print(" coarse grid Iteration  "+str(i)+" residue = "+ str(R_norm)+" NLI = "+ str(non_linear_index) )
    if(solver_options.verbosity):
        print()      
        print("time_spent_on_perturbation_residual= "+str(time_spent_on_perturbation_residual))
        print("time_spent_on_residual= "+str(time_spent_on_residual))
        print("time_spent_on_preconditioner= "+str(time_spent_on_preconditioner))
        print("time_spent_on_alphas_matrix_assembly= "+str(time_spent_on_alphas_matrix_assembly))
        print("time_spent_on_alphas_matrix_solve= "+str(time_spent_on_alphas_matrix_solve))
        print("time_spent_on_solution_update= "+str(time_spent_on_solution_update))
        total_time_end=time.time()
        print("total time spent for nx of " +str(problem.nx)+" = "+str(total_time_end-total_time_start))
    return x,R_norm


plt.close('all')
 
n=420
#probleme size
problem=Problem(dim=3,nx=n,ny=n,nz=n)
x=np.ones(problem.size)*0



prep_options=Preconditionner_option(level=2,alpha=1,iterations_for_smoothing=5,frequency_of_residual_direct_evaluation=10,tol=1e-6,max_iterations=1000,max_krylov_vectors=1000, verbosity=True)


x,r=solve(x,problem,prep_options)

# if problem.dim==2:
#     fig,ax = plt.subplots()
#     fig1,ax1 = plt.subplots()
#     results = x.reshape(problem.nx,problem.ny).transpose()
#     X = [0,1]
#     Y = [0,1]
        
#     im3 = ax.imshow(results, extent = [min(X),max(X),min(Y),max(Y)],aspect="auto")    
    
#     multiplicator=2
    
#     nx2=int(problem.nx*multiplicator)
#     ny2=int(problem.ny*multiplicator)
#     coarse_grid=x.reshape((problem.nx, problem.ny), order='F')
#     fine_grid=bilinear_interpolate(coarse_grid, nx2, ny2)
#     x=fine_grid.flatten(order='F')
    
#     results = x.reshape(int(problem.ny*multiplicator),int(problem.nx*multiplicator)).transpose()
#     X = [0,1]
#     Y = [0,1]
        
#     im4 = ax1.imshow(results, extent = [min(X),max(X),min(Y),max(Y)],aspect="auto")
    
# if problem.dim==3:
#     x_0=np.copy(x)
#     fine_grid=x.reshape((problem.nx, problem.ny, problem.nz), order='F')
#     # Get the shape for later use
#     nz, ny, nx = fine_grid.shape
    
#     # Generate indices for the grid
#     indices = np.indices((nz, ny, nx))
    
#     # Normalize indices to the range of their respective dimensions
#     # Instead of directly multiplying, we normalize each component separately
#     x = indices[2] * (nx - 1) / (nx - 1)
#     y = indices[1] * (ny - 1) / (ny - 1)
#     z = indices[0] * (nz - 1) / (nz - 1)
#     values = fine_grid.flatten( order='F')
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Use the value as the color
#     colors = plt.cm.jet(values / max(values))
    
#     # Plotting
#     sc = ax.scatter(x, y, z, c=colors, s=100, cmap='jet')
    
#     # Colorbar
#     cbar = plt.colorbar(sc, ax=ax)
#     cbar.set_label('Value')
    
#     # Labels
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')
#     ax.set_title('3D Array Visualization')
    
#     multiplicator=2
    
#     x=x_0
#     nx2=int(problem.nx*multiplicator)
#     ny2=int(problem.ny*multiplicator)
#     nz2=int(problem.nz*multiplicator)
#     coarse_grid=x.reshape((problem.nx, problem.ny,problem.nz), order='F')
#     fine_grid=trilinear_interpolate(coarse_grid, nx2, ny2, nz2)
#     x=fine_grid.flatten(order='F')
    
#     fine_grid=x.reshape((problem.nx*multiplicator, problem.ny*multiplicator, problem.nz*multiplicator), order='F')
#     # Get the shape for later use
#     nz, ny, nx = fine_grid.shape
    
#     # Generate indices for the grid
#     indices = np.indices((nz, ny, nx))
    
#     # Normalize indices to the range of their respective dimensions
#     # Instead of directly multiplying, we normalize each component separately
#     x = indices[2] * (nx - 1) / (nx - 1)
#     y = indices[1] * (ny - 1) / (ny - 1)
#     z = indices[0] * (nz - 1) / (nz - 1)
#     values = fine_grid.flatten( order='F')
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Use the value as the color
#     colors = plt.cm.jet(values / max(values))
    
#     # Plotting
#     sc = ax.scatter(x, y, z, c=colors, s=100, cmap='jet')
    
#     # Colorbar
#     cbar = plt.colorbar(sc, ax=ax)
#     cbar.set_label('Value')
    
#     # Labels
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')
#     ax.set_title('3D Array Visualization')


# # Graphiques


# fig,ax2 = plt.subplots()
# ax2.plot(residual)
# ax2.set_yscale('log')
# ax2.set(xlabel=r'iteration', ylabel=r'residual')
# plt.show()
