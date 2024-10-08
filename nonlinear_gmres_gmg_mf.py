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
from scipy.sparse.linalg import spilu, LinearOperator, gmres, spsolve
from scipy.sparse import coo_matrix
import time
import sys
from numba import cuda, float64
import cupy as cp

np.set_printoptions(threshold=sys.maxsize)
def construct_jacobian(x, problem):
    """
    Computes the Jacobian for a system of equations characterized by nonlinearity.
    This function implements the Jacobian  calculation for equations of the form:
    D * (∇²x) + C + E * exp(F * x) - G * x * ∇ • x = 0,
    where D, C, E, F, and G are coefficients defining the problem's behavior.

    Parameters:
    - x (array_like): The current solution vector.
    - problem (object): An object encapsulating the problem specifics, including dimensions and grid properties.

    Returns:
    - J (array_like): The computed Jacobian.

    The calculation differentiates between 2D and 3D problems and handles interior and boundary points distinctly,
    applying the specified equation while taking spatial discretization into account.
    """
    if(problem.dim==2):
        nx, ny = problem.nx, problem.ny
        N = x.size
        dx = (1.0 - 0) / (nx - 1)
        dy = (1.0 - 0) / (ny - 1)
        dx_inv = 1.0 / dx
        dy_inv = 1.0 / dy
        dx2 = dx**2
        dy2 = dy**2
        D = problem.D
        E = problem.E
        F = problem.F
        G = problem.G
    
        indices = np.arange(N)
        interior_mask = (indices % ny != 0) & (indices % ny != ny - 1) & (indices >= ny) & (indices < N - ny)
        interior_indices = indices[interior_mask]
        boundary_indices = np.where(~interior_mask)[0]
    
        # Initialize arrays for interior points
        rows_i = np.repeat(interior_indices, 5)
        cols_i = np.empty_like(rows_i)
        data_i = np.empty_like(rows_i, dtype=np.float64)
    
        # Set up the Laplacian contributions for interior points
        data_i[0::5] = (-2 * D / dx2 - 2 * D / dy2) + E * F * np.exp(F * x[interior_indices])
        cols_i[0::5] = interior_indices  # Self
    
        left = interior_indices - 1
        right = interior_indices + 1
        up = interior_indices - ny
        down = interior_indices + ny
    
        cols_i[1::5] = left
        data_i[1::5] = D / dx2  # Left neighbor
        cols_i[2::5] = right
        data_i[2::5] = D / dx2  # Right neighbor
        cols_i[3::5] = up
        data_i[3::5] = D / dy2  # Up neighbor
        cols_i[4::5] = down
        data_i[4::5] = D / dy2  # Down neighbor
    
        # Adding gradient contributions to the diagonal
        # Now handle indirect effects due to the gradient term
        data_i[0::5] += -G * (-0.5 * dx_inv*x[left] + 0.5 * dx_inv* x[right] + 0.5 * dy_inv* x[up] - 0.5 * dy_inv* x[down])
        # Left neighbor contribution to gradient at x_i
        data_i[1::5] += -G * (-0.5 * dx_inv) * x[interior_mask]
        # Right neighbor contribution to gradient at x_i
        data_i[2::5] += -G * (0.5 * dx_inv) * x[interior_mask]
        # Up neighbor contribution to gradient at x_i
        data_i[3::5] += -G * (0.5 * dy_inv) * x[interior_mask]
        # Down neighbor contribution to gradient at x_i
        data_i[4::5] += -G * (-0.5 * dy_inv) * x[interior_mask]
    
        # Handle boundary conditions (Dirichlet example)
        rows_b = boundary_indices
        cols_b = boundary_indices
        data_b = np.ones(len(boundary_indices))*-1.0  # Assuming boundary condition is x=0 on the boundary
        
        
    
        # Combine interior and boundary entries
        rows = np.concatenate([rows_i, rows_b])
        cols = np.concatenate([cols_i, cols_b])
        data = np.concatenate([data_i, data_b])
        
        sort_indices = np.lexsort((cols, rows))
        rows_sorted = rows[sort_indices]
        cols_sorted = cols[sort_indices]
        data_sorted = data[sort_indices]
    
        # Create the sparse Jacobian matrix in COO format
        J = coo_matrix((data_sorted, (rows_sorted, cols_sorted)), shape=(N, N))
    
        return J.tocsr()
    if(problem.dim==3):
        nx, ny, nz = problem.nx, problem.ny, problem.nz
        N = x.size
        dx = (1.0 - 0) / (nx - 1)
        dy = (1.0 - 0) / (ny - 1)
        dz = (1.0 - 0) / (nz - 1)
        dx_inv = 1.0 / dx
        dy_inv = 1.0 / dy
        dz_inv = 1.0 / dy
        dx2 = dx**2
        dy2 = dy**2
        dz2 = dz**2
        D = problem.D
        E = problem.E
        F = problem.F
        G = problem.G
    
        indices = np.arange(N)
        interior_mask = (indices-(indices-indices%(problem.nx*problem.ny))>problem.ny) & \
                        (indices-(indices-indices%(problem.nx*problem.ny))<problem.nx*(problem.ny-1)) & \
                        (indices % problem.ny != 0) & \
                        (indices % problem.ny != problem.ny - 1)  & \
                        (indices>=(problem.nx*problem.ny))  & \
                        (indices<=(problem.nz-1)*(problem.ny*problem.nx))
        interior_indices = indices[interior_mask]
        boundary_indices = np.where(~interior_mask)[0]
    
        # Initialize arrays for interior points
        rows_i = np.repeat(interior_indices, 7)
        cols_i = np.empty_like(rows_i)
        data_i = np.empty_like(rows_i, dtype=np.float64)
    
        # Set up the Laplacian contributions for interior points
        data_i[0::7] = (-2 * D / dx2 - 2 * D / dy2 - 2 * D / dz2) + E * F * np.exp(F * x[interior_indices])
        cols_i[0::7] = interior_indices  # Self
    
        left =interior_indices - problem.ny
        right = interior_indices + problem.ny
        up = interior_indices - 1
        down = interior_indices + 1
        front = interior_indices - problem.nx*problem.ny
        back = interior_indices + problem.nx*problem.ny
    
        cols_i[1::7] = left
        data_i[1::7] = D / dx2  # Left neighbor
        cols_i[2::7] = right
        data_i[2::7] = D / dx2  # Right neighbor
        cols_i[3::7] = up
        data_i[3::7] = D / dy2  # Up neighbor
        cols_i[4::7] = down
        data_i[4::7] = D / dy2  # Down neighbor
        cols_i[5::7] = back
        data_i[5::7] = D / dy2  # Down neighbor
        cols_i[6::7] = front
        data_i[6::7] = D / dy2  # Down neighbor
    
        # Adding gradient contributions to the diagonal
        # Now handle indirect effects due to the gradient term
        data_i[0::7] += -G * (-0.5 * dx_inv*x[left] + 0.5 * dx_inv* x[right] + 0.5 * dy_inv* x[up] - 0.5 * dy_inv* x[down] + 0.5 * dy_inv* x[front] - 0.5 * dy_inv* x[back] )
        # Left neighbor contribution to gradient at x_i
        data_i[1::7] += -G * (-0.5 * dx_inv) * x[interior_mask]
        # Right neighbor contribution to gradient at x_i
        data_i[2::7] += -G * (0.5 * dx_inv) * x[interior_mask]
        # Up neighbor contribution to gradient at x_i
        data_i[3::7] += -G * (0.5 * dy_inv) * x[interior_mask]
        # Down neighbor contribution to gradient at x_i
        data_i[4::7] += -G * (-0.5 * dy_inv) * x[interior_mask]
        # Back neighbor contribution to gradient at x_i
        data_i[5::7] += -G * (-0.5 * dz_inv) * x[interior_mask]
        # Front neighbor contribution to gradient at x_i
        data_i[6::7] += -G * (0.5 * dz_inv) * x[interior_mask]
    
        # Handle boundary conditions (Dirichlet example)
        rows_b = boundary_indices
        cols_b = boundary_indices
        data_b = np.ones(len(boundary_indices))*-1.0  # Assuming boundary condition is x=0 on the boundary
        
        
    
        # Combine interior and boundary entries
        rows = np.concatenate([rows_i, rows_b])
        cols = np.concatenate([cols_i, cols_b])
        data = np.concatenate([data_i, data_b])
        
        sort_indices = np.lexsort((cols, rows))
        rows_sorted = rows[sort_indices]
        cols_sorted = cols[sort_indices]
        data_sorted = data[sort_indices]
    
        # Create the sparse Jacobian matrix in COO format
        J = coo_matrix((data_sorted, (rows_sorted, cols_sorted)), shape=(N, N))
    
        return J.tocsr()


@cuda.jit
def compute_residual_2d(x, r, D, exp_coef, const_coef, exp_variable_coef, u_div_u_coef, nx, ny, dx, dy, dx2, dy2):
    i, j = cuda.grid(2)
    if i > 0 and i < nx - 1 and j > 0 and j < ny - 1:
        idx = j + i * ny
        left = idx - ny
        right = idx + ny
        up = idx - 1
        down = idx + 1

        # Calculate the residual at this point
        r[idx] = (D * (-2 * x[idx] / dx2 - 2 * x[idx] / dy2 +
                       x[left] / dx2 + x[right] / dx2 +
                       x[up] / dy2 + x[down] / dy2) +
                  const_coef + exp_coef * math.exp(exp_variable_coef * x[idx]) -
                  u_div_u_coef * x[idx] * (-0.5 * x[left] / dx + 0.5 * x[right] / dx +
                                           0.5 * x[up] / dy - 0.5 * x[down] / dy))

@cuda.jit
def compute_residual_3d(x, r, D, exp_coef, const_coef, exp_variable_coef, u_div_u_coef, nx, ny, nz, dx, dy, dz, dx2, dy2, dz2):
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
        idx = k + j * nz + i * ny * nz
        left = idx - ny
        right = idx + ny
        up = idx - 1
        down = idx + 1
        front = idx - ny * nz
        back = idx + ny * nz

        # Calculate the residual at this point
        r[idx] = (D * (-2 * x[idx] / dx2 - 2 * x[idx] / dy2 - 2 * x[idx] / dz2 +
                       x[left] / dx2 + x[right] / dx2 +
                       x[up] / dy2 + x[down] / dy2 +
                       x[front] / dz2 + x[back] / dz2) +
                  const_coef + exp_coef * math.exp(exp_variable_coef * x[idx]) -
                  u_div_u_coef * x[idx] * (-0.5 * x[left] / dx + 0.5 * x[right] / dx +
                                           0.5 * x[up] / dy - 0.5 * x[down] / dy +
                                           0.5 * x[back] / dz - 0.5 * x[front] / dz))


def R(x, problem, offset=None):
    if len(x)<1024:
        """
        Computes the residual for a system of equations characterized by nonlinearity and spatial derivatives.
        This function implements the residual calculation for equations of the form:
        D * (∇²x) + C + E * exp(F * x) - G * x * ∇ • x = 0,
        where D, C, E, F, and G are coefficients defining the problem's behavior.
    
        Parameters:
        - x (array_like): The current solution vector.
        - problem (object): An object encapsulating the problem specifics, including dimensions and grid properties.
        - off_set (array_like, optional): An offset vector to adjust the calculations. Defaults to an empty numpy array.
    
        Returns:
        - r (array_like): The computed residual vector.
    
        The calculation differentiates between 2D and 3D problems and handles interior and boundary points distinctly,
        applying the specified equation while taking spatial discretization into account.
        """
        D=problem.D
        exp_coef=problem.E# really non linear 3.30
        const_coef=problem.C
        exp_variable_coef=problem.F
        u_div_u_coef=problem.G
    
        if len(offset) == 0:
            offset = np.zeros(len(x))
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
            r[boundary_mask] = (0.0-x[boundary_mask])     
                        
            # Compute indices for neighbors
            left = indices[interior_mask] - problem.ny
            right = indices[interior_mask] + problem.ny
            up = indices[interior_mask] - 1
            down = indices[interior_mask] + 1
        
            # Apply the equation for interior points
            r[interior_mask] = D * (-2*x[interior_mask]/dx2-2*x[interior_mask]/dy2 + x[left]/dx2+ x[right]/dx2 + x[up]/dy2 + x[down]/dy2)+const_coef+exp_coef*np.exp(exp_variable_coef*x[interior_mask])\
                - u_div_u_coef*x[interior_mask]*(-0.5*x[left]/dx+ 0.5*x[right]/dx + 0.5*x[up]/dy - 0.5*x[down]/dy)
    
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
            r[boundary_mask] = (0.0-x[boundary_mask])               
            # Compute indices for neighbors
            left = indices[interior_mask] - problem.ny
            right = indices[interior_mask] + problem.ny
            up = indices[interior_mask] - 1
            down = indices[interior_mask] + 1
            front = indices[interior_mask] - problem.nx*problem.ny
            back = indices[interior_mask] + problem.nx*problem.ny
            
            # Apply the equation for interior points
            r[interior_mask] = D * (-2*x[interior_mask]/dx2-2*x[interior_mask]/dy2-2*x[interior_mask]/dz2 + x[left]/dx2+ x[right]/dx2 + x[up]/dy2 + x[down]/dy2 + x[front]/dz2 + x[back]/dz2)+const_coef+exp_coef*np.exp(exp_variable_coef*x[interior_mask])\
                -u_div_u_coef*x[interior_mask]*(-0.5*x[left]/dx+ 0.5*x[right]/dx + 0.5*x[up]/dy - 0.5*x[down]/dy+  0.5*x[back]/dz - 0.5*x[front]/dz)
        
            # Handle boundary points separately, if necessary
        return r- offset
    else:
        if offset is None:
            offset = np.zeros_like(x)
        
        D = problem.D
        exp_coef = problem.E
        const_coef = problem.C
        exp_variable_coef = problem.F
        u_div_u_coef = problem.G
    
        r = np.zeros_like(x)
        
        if problem.dim == 2:
            nx, ny = problem.nx, problem.ny
            dx = (1.0 - 0) / (nx - 1)
            dy = (1.0 - 0) / (ny - 1)
            dx2 = dx ** 2
            dy2 = dy ** 2
    
            # Define grid dimensions
            threads_per_block = (16, 16)
            blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
            # Transfer data to GPU
            x_device = cuda.to_device(x)
            r_device = cuda.to_device(r)
    
            # Launch kernel
            compute_residual_2d[blocks_per_grid, threads_per_block](x_device, r_device, D, exp_coef, const_coef,
                                                                    exp_variable_coef, u_div_u_coef, nx, ny, dx, dy, dx2, dy2)
    
            # Copy result back to CPU
            r = r_device.copy_to_host()
            
        elif problem.dim == 3:
            nx, ny, nz = problem.nx, problem.ny, problem.nz
            dx = (1.0 - 0) / (nx - 1)
            dy = (1.0 - 0) / (ny - 1)
            dz = (1.0 - 0) / (nz - 1)
            dx2 = dx ** 2
            dy2 = dy ** 2
            dz2 = dz ** 2
    
            # Define grid dimensions
            threads_per_block = (8, 8, 8)
            blocks_per_grid_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid_z = (nz + threads_per_block[2] - 1) // threads_per_block[2]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    
            # Transfer data to GPU
            x_device = cuda.to_device(x)
            r_device = cuda.to_device(r)
    
            # Launch kernel
            compute_residual_3d[blocks_per_grid, threads_per_block](x_device, r_device, D, exp_coef, const_coef,
                                                                    exp_variable_coef, u_div_u_coef, nx, ny, nz, dx, dy, dz, dx2, dy2, dz2)
    
            # Copy result back to CPU
            r = r_device.copy_to_host()
        
        return r - offset


# def R(x,problem,off_set=np.zeros(0)):
#     """
#     Computes the residual for a system of equations characterized by nonlinearity and spatial derivatives.
#     This function implements the residual calculation for equations of the form:
#     D * (∇²x) + C + E * exp(F * x) - G * x * ∇ • x = 0,
#     where D, C, E, F, and G are coefficients defining the problem's behavior.

#     Parameters:
#     - x (array_like): The current solution vector.
#     - problem (object): An object encapsulating the problem specifics, including dimensions and grid properties.
#     - off_set (array_like, optional): An offset vector to adjust the calculations. Defaults to an empty numpy array.

#     Returns:
#     - r (array_like): The computed residual vector.

#     The calculation differentiates between 2D and 3D problems and handles interior and boundary points distinctly,
#     applying the specified equation while taking spatial discretization into account.
#     """
#     D=problem.D
#     exp_coef=problem.E# really non linear 3.30
#     const_coef=problem.C
#     exp_variable_coef=problem.F
#     u_div_u_coef=problem.G

#     if len(off_set) == 0:
#         off_set = np.zeros(len(x))
#     if(problem.dim==2):
#         # Initialize the result array
#         r = np.zeros_like(x)
    
#         # Compute grid spacings
#         dx = (1 - 0) / (problem.nx - 1)
#         dy = (1 - 0) / (problem.ny - 1) if problem.dim > 1 else 0
        
    
#         # Pre-compute common factors
#         dx2 = dx ** 2
#         dy2 = dy ** 2
    
#         # Indices for all points
#         indices = np.arange(x.size)
        
#         # Exclude boundary points for now by focusing on interior points
#         # Adjust these conditions based on your problem's boundary definitions
#         interior_mask = (indices % problem.ny != 0) & \
#                         (indices % problem.ny != problem.ny - 1) & \
#                         (indices >= problem.ny) & \
#                         (indices < x.size - problem.ny)
        
#         boundary_mask = ~interior_mask
#         r[boundary_mask] = (0.0-x[boundary_mask])     
                    
#         # Compute indices for neighbors
#         left = indices[interior_mask] - problem.ny
#         right = indices[interior_mask] + problem.ny
#         up = indices[interior_mask] - 1
#         down = indices[interior_mask] + 1
    
#         # Apply the equation for interior points
#         r[interior_mask] = D * (-2*x[interior_mask]/dx2-2*x[interior_mask]/dy2 + x[left]/dx2+ x[right]/dx2 + x[up]/dy2 + x[down]/dy2)+const_coef+exp_coef*np.exp(exp_variable_coef*x[interior_mask])\
#             - u_div_u_coef*x[interior_mask]*(-0.5*x[left]/dx+ 0.5*x[right]/dx + 0.5*x[up]/dy - 0.5*x[down]/dy)

#         # Handle boundary points separately, if necessary
#     if(problem.dim==3):
#         # Initialize the result array
#         r = np.zeros_like(x)
    
#         # Compute grid spacings
#         dx = (1 - 0) / (problem.nx - 1)
#         dy = (1 - 0) / (problem.ny - 1)
#         dz = (1 - 0) / (problem.nz - 1)
    
#         # Pre-compute common factors
#         dx2 = dx ** 2
#         dy2 = dy ** 2
#         dz2 = dz ** 2
    
#         # Indices for all points
#         indices = np.arange(x.size)

#         interior_mask = (indices-(indices-indices%(problem.nx*problem.ny))>problem.ny) & \
#                         (indices-(indices-indices%(problem.nx*problem.ny))<problem.nx*(problem.ny-1)) & \
#                         (indices % problem.ny != 0) & \
#                         (indices % problem.ny != problem.ny - 1)  & \
#                         (indices>=(problem.nx*problem.ny))  & \
#                         (indices<=(problem.nz-1)*(problem.ny*problem.nx))
                       
#         # Exclude boundary points for now by focusing on interior points
#         # Adjust these conditions based on your problem's boundary definitions
#         boundary_mask = ~interior_mask
#         r[boundary_mask] = (0.0-x[boundary_mask])               
#         # Compute indices for neighbors
#         left = indices[interior_mask] - problem.ny
#         right = indices[interior_mask] + problem.ny
#         up = indices[interior_mask] - 1
#         down = indices[interior_mask] + 1
#         front = indices[interior_mask] - problem.nx*problem.ny
#         back = indices[interior_mask] + problem.nx*problem.ny
        
#         # Apply the equation for interior points
#         r[interior_mask] = D * (-2*x[interior_mask]/dx2-2*x[interior_mask]/dy2-2*x[interior_mask]/dz2 + x[left]/dx2+ x[right]/dx2 + x[up]/dy2 + x[down]/dy2 + x[front]/dz2 + x[back]/dz2)+const_coef+exp_coef*np.exp(exp_variable_coef*x[interior_mask])\
#             -u_div_u_coef*x[interior_mask]*(-0.5*x[left]/dx+ 0.5*x[right]/dx + 0.5*x[up]/dy - 0.5*x[down]/dy+  0.5*x[back]/dz - 0.5*x[front]/dz)
    
#         # Handle boundary points separately, if necessary
#     return r- off_set

@cuda.jit
def gpu_bilinear_interpolate(first_grid, x0, x1, y0, y1, wx, wy, second_grid):
    i, j = cuda.grid(2)

    if i < second_grid.shape[0] and j < second_grid.shape[1]:
        # Perform interpolation
        top_left = first_grid[y0[i], x0[j]]
        top_right = first_grid[y0[i], x1[j]]
        bottom_left = first_grid[y1[i], x0[j]]
        bottom_right = first_grid[y1[i], x1[j]]

        second_grid[i, j] = (top_left * (1 - wx[j]) * (1 - wy[i]) +
                             top_right * wx[j] * (1 - wy[i]) +
                             bottom_left * (1 - wx[j]) * wy[i] +
                             bottom_right * wx[j] * wy[i])

def bilinear_interpolate(first_grid, nx2, ny2):
    ny, nx = first_grid.shape

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

    # If the grid size is large, use GPU; otherwise, use CPU
    if nx2 * ny2 > 2048:
        # Allocate memory for result on GPU
        second_grid_gpu = cuda.device_array((ny2, nx2), dtype=np.float64)

        # Launch kernel
        threads_per_block = (16, 16)
        blocks_per_grid_x = (nx2 + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid_y = (ny2 + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        gpu_bilinear_interpolate[blocks_per_grid, threads_per_block](
            cuda.to_device(first_grid), cuda.to_device(x0), cuda.to_device(x1),
            cuda.to_device(y0), cuda.to_device(y1), cuda.to_device(wx),
            cuda.to_device(wy), second_grid_gpu
        )

        # Copy result back to host
        second_grid = second_grid_gpu.copy_to_host()
    else:
        # CPU-based interpolation
        # Reshape for broadcasting
        wx = wx.reshape(1, -1)
        wy = wy.reshape(-1, 1)

        # Get values from the four neighbors
        top_left = first_grid[y0, :][:, x0]
        top_right = first_grid[y0, :][:, x1]
        bottom_left = first_grid[y1, :][:, x0]
        bottom_right = first_grid[y1, :][:, x1]

        # Perform interpolation
        second_grid = (top_left * (1 - wx) * (1 - wy) +
                       top_right * wx * (1 - wy) +
                       bottom_left * (1 - wx) * wy +
                       bottom_right * wx * wy)

    return second_grid


@cuda.jit
def gpu_trilinear_interpolate(first_grid, x0, x1, y0, y1, z0, z1, xw, yw, zw, second_grid):
    i, j, k = cuda.grid(3)

    if i < second_grid.shape[0] and j < second_grid.shape[1] and k < second_grid.shape[2]:
        # Retrieve values at the eight corners
        c000 = first_grid[z0[i], y0[j], x0[k]]
        c001 = first_grid[z0[i], y0[j], x1[k]]
        c010 = first_grid[z0[i], y1[j], x0[k]]
        c011 = first_grid[z0[i], y1[j], x1[k]]
        c100 = first_grid[z1[i], y0[j], x0[k]]
        c101 = first_grid[z1[i], y0[j], x1[k]]
        c110 = first_grid[z1[i], y1[j], x0[k]]
        c111 = first_grid[z1[i], y1[j], x1[k]]

        # Perform interpolation using weights
        interpolated_value = (c000 * (1 - xw[k]) * (1 - yw[j]) * (1 - zw[i]) +
                              c001 * xw[k] * (1 - yw[j]) * (1 - zw[i]) +
                              c010 * (1 - xw[k]) * yw[j] * (1 - zw[i]) +
                              c011 * xw[k] * yw[j] * (1 - zw[i]) +
                              c100 * (1 - xw[k]) * (1 - yw[j]) * zw[i] +
                              c101 * xw[k] * (1 - yw[j]) * zw[i] +
                              c110 * (1 - xw[k]) * yw[j] * zw[i] +
                              c111 * xw[k] * yw[j] * zw[i])

        # Store the result in the output grid
        second_grid[i, j, k] = interpolated_value

def trilinear_interpolate(first_grid, nx2, ny2, nz2):
    nz, ny, nx = first_grid.shape

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
    x_weight = (x_new - x0)
    y_weight = (y_new - y0)
    z_weight = (z_new - z0)

    # If the grid size is large, use GPU; otherwise, use CPU
    if nx2 * ny2 * nz2 > 2048:
        # Allocate memory for result on GPU
        second_grid_gpu = cuda.device_array((nz2, ny2, nx2), dtype=np.float64)

        # Launch kernel
        threads_per_block = (8, 8, 8)
        blocks_per_grid_x = (nx2 + threads_per_block[2] - 1) // threads_per_block[2]
        blocks_per_grid_y = (ny2 + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid_z = (nz2 + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid = (blocks_per_grid_z, blocks_per_grid_y, blocks_per_grid_x)

        gpu_trilinear_interpolate[blocks_per_grid, threads_per_block](
            cuda.to_device(first_grid), cuda.to_device(x0), cuda.to_device(x1),
            cuda.to_device(y0), cuda.to_device(y1), cuda.to_device(z0), cuda.to_device(z1),
            cuda.to_device(x_weight), cuda.to_device(y_weight), cuda.to_device(z_weight), second_grid_gpu
        )

        # Copy result back to host
        second_grid = second_grid_gpu.copy_to_host()
    else:
        # CPU-based trilinear interpolation
        # Retrieve values at the eight corners for the entire grid
        c000 = first_grid[z0[:, None, None], y0[None, :, None], x0[None, None, :]]
        c001 = first_grid[z0[:, None, None], y0[None, :, None], x1[None, None, :]]
        c010 = first_grid[z0[:, None, None], y1[None, :, None], x0[None, None, :]]
        c011 = first_grid[z0[:, None, None], y1[None, :, None], x1[None, None, :]]
        c100 = first_grid[z1[:, None, None], y0[None, :, None], x0[None, None, :]]
        c101 = first_grid[z1[:, None, None], y0[None, :, None], x1[None, None, :]]
        c110 = first_grid[z1[:, None, None], y1[None, :, None], x0[None, None, :]]
        c111 = first_grid[z1[:, None, None], y1[None, :, None], x1[None, None, :]]

        # Calculate weights
        x_weight = x_weight.reshape(1, 1, -1)
        y_weight = y_weight.reshape(1, -1, 1)
        z_weight = z_weight.reshape(-1, 1, 1)

        # Perform trilinear interpolation across the entire grid at once
        second_grid = (c000 * (1 - x_weight) * (1 - y_weight) * (1 - z_weight) + 
                       c001 * x_weight * (1 - y_weight) * (1 - z_weight) + 
                       c010 * (1 - x_weight) * y_weight * (1 - z_weight) + 
                       c011 * x_weight * y_weight * (1 - z_weight) + 
                       c100 * (1 - x_weight) * (1 - y_weight) * z_weight + 
                       c101 * x_weight * (1 - y_weight) * z_weight + 
                       c110 * (1 - x_weight) * y_weight * z_weight + 
                       c111 * x_weight * y_weight * z_weight)

    return second_grid



class Problem():
    """
    A class to model a problem space that can be 2D, or 3D. and solve :
    D * (∇²x) + C + E * exp(F * x) - G * x * ∇ • x = 0,

    Attributes:
        dim (int): The dimensionality of the problem space (1, 2, or 3).
        nx (int): The size of the problem space in the x-direction.
        ny (int): The size of the problem space in the y-direction, relevant for 2D and 3D problems.
        nz (int): The size of the problem space in the z-direction, relevant for 3D problems.
        size (int): The total size of the problem space, calculated based on `dim`, `nx`, `ny`, and `nz`.
        C (float): Constant term in the equation.
        D (float): Diffusion coefficient in the equation.
        E (float): Coefficient for the exponential term in the equation.
        F (float): Exponent multiplier in the exponential term of the equation.
        G (float): Coefficient for the nonlinear convection term in the equation.
    """
    def __init__(self,dim=2,nx=100,ny=100,nz=100,C=1,D=1,E=0,F=1,G=0):
        self.dim=dim
        self.nx=nx
        self.ny=ny
        self.nz=nz
        self.C=C
        self.D=D
        self.E=E
        self.F=F
        self.G=G
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
        minimum_mesh_size (int): The minimum mesh size used in the GMG process. If the mesh size falls between the minimum mesh size and twice the minimum mesh size, the system is solved directly without using a preconditioner.
        non_linearity_index_limit (float): The threshold for the nonlinearity index within the GMRES solver. If the index exceeds this threshold, the GMRES solver is restarted.
        verbosity (bool): If True, the solver will output detailed progress information.
          
    Tips for Nonlinear Problems:
    - **Alpha (Perturbation Size)**: For cases with nonlinear behavior, it is recommended to reduce the
      alpha value. A smaller perturbation size helps ensure that the corrections are based on a more accurate
      assessment of the variation, which is crucial in nonlinear contexts where small changes can lead to
      significantly different outcomes.
    - **Frequency of Direct Residual Evaluation**: In nonlinear cases, it's beneficial to increase the
      frequency of direct residual evaluations (effectively reducing the `frequency_of_residual_direct_evaluation`
      parameter). This adjustment helps monitor the nonlinearity more closely and prevents reliance on a potentially
      inaccurate Krylov subspace. Frequent direct evaluations ensure that the solver's corrections are based on
      the most accurate possible assessment of the current state, which is essential for effectively addressing
      nonlinear behaviors.
    """
    def __init__(self,level=2,alpha=1,iterations_for_smoothing=10,frequency_of_residual_direct_evaluation=10,tol=1e-6,max_iterations=1000,max_krylov_vectors=1000,minimum_mesh_size=4,non_linearity_index_limit=0.5,verbosity=True):
        self.level=level
        self.alpha=alpha
        self.iterations_for_smoothing=iterations_for_smoothing
        self.frequency_of_residual_direct_evaluation=frequency_of_residual_direct_evaluation
        self.tol=tol
        self.max_iterations=max_iterations
        self.max_krylov_vectors=max_krylov_vectors
        self.minimum_mesh_size=minimum_mesh_size
        self.non_linearity_index_limit=non_linearity_index_limit
        self.verbosity=verbosity
    def clone(self):
        """
        Creates a deep copy of the current instance, preserving the configuration of the preconditioner options.
        """
        return Preconditionner_option(self.level,self.alpha,self.iterations_for_smoothing,self.frequency_of_residual_direct_evaluation,self.tol,self.max_iterations,self.max_krylov_vectors,self.minimum_mesh_size,self.non_linearity_index_limit,self.verbosity)

class Preconditioner_GMG():
    """
    A class implementing a Geometric Multigrid (GMG) preconditioner with a V cycle. This preconditioner
    is designed to efficiently find adequate correction vector arising in discretized partial
    differential equations, especially those that are large and sparse.
    This version of the preconditionner is for nonlinear system of equations

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
    
        # Iteratively coarsen the field down to the desired level or the smallest acceptable mesh size (Faster to coarsen by step of two then directly to the level)
        for i in range(solver_options.level + 1):
            # Check if the current iteration is at the coarsest level
            if i == solver_options.level:
                # At the coarsest level, we might adjust solver tolerance or other parameters
                # solver_options.tol = solver_options.tol*1e-2  # This line appears redundant but may be a placeholder for adjustments
                problem.update()  # Ensure problem dimensions and other attributes are up-to-date
                # Decision branch based on whether the problem size is above a threshold (e.g., nx > 8)
                if problem.nx > solver_options.minimum_mesh_size*2:
                    # If no solution has been cached for this level, solve the problem at this level.
                    # The correction vector at the coarse level is determined by calculating the difference
                    # between two solutions at the coarse level: one being the direct coarse-level solution, 
                    # and the other being the solution that yields a residual identical to the projection of 
                    # the fine-level residual onto the coarse level.
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
                    # Calculate the correction vector on the coarse level. 
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
            if i < solver_options.level and problem.nx / 2 >solver_options.minimum_mesh_size:
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

class Preconditioner_GMG_Matrix():
    """
    A class implementing a Geometric Multigrid (GMG) preconditioner with a V cycle. This preconditioner
    is designed to efficiently find adequate correction vector arising in discretized partial
    differential equations, especially those that are large and sparse. 
    This version of the preconditionner is for linear system of equations

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
    
        # Iteratively coarsen the field down to the desired level or the smallest acceptable mesh size (Faster to coarsen by step of two then directly to the level)
        for i in range(solver_options.level + 1):
            # Check if the current iteration is at the coarsest level
            if i == solver_options.level:
                # At the coarsest level, we might adjust solver tolerance or other parameters
                # Better performance when sub-grid are solved with higher precision.
                solver_options.tol = solver_options.tol*1e-2  # This line appears redundant but may be a placeholder for adjustments
                problem.update()  # Ensure problem dimensions and other attributes are up-to-date
                # Decision branch based on whether the problem size is above a threshold (e.g., nx > 8)
                if problem.nx > solver_options.minimum_mesh_size*2:
                    # If no solution has been cached for this level, solve the problem at this level.
                    # The correction vector at the coarse level is determined by calculating the difference
                    # between two solutions at the coarse level: one being the direct coarse-level solution, 
                    # and the other being the solution that yields a residual identical to the projection of 
                    # the fine-level residual onto the coarse level.
                    if self.solution_on_coarses_level.size == 0:
                        x_0_coarse = x  # Save the current guess before solving
                        # Solve the problem at the current level, with a preconditioner (indicated by True/False)
                        x, R_norm = solve_matrix_gmg(x, problem, solver_options, True, off_set)
                        # Cache the solution for future use
                        self.solution_on_coarses_level = np.copy(x)
                    else:
                        # If a solution is already cached, use it to determine the coarse correction
                        x_0_coarse, R_norm = solve_matrix_gmg(x, problem, solver_options, True, r + off_set)
                        x = self.solution_on_coarses_level
                    # Calculate the correction vector on the coarse level. 
                    v_coarse = x - x_0_coarse
                else:
                    # Similar logic applies if the problem size is below the threshold, but without preconditioning
                    if self.solution_on_coarses_level.size == 0:
                        x_0_coarse = x
                        x, R_norm = solve_matrix_gmg(x, problem, solver_options, False, off_set)
                        self.solution_on_coarses_level = np.copy(x)
                    else:
                        x_0_coarse, R_norm = solve_matrix_gmg(x, problem, solver_options, False, r + off_set)
                        x = self.solution_on_coarses_level
                    v_coarse = x - x_0_coarse
            # If not at the coarsest level, coarsen the fields for the next iteration
            if i < solver_options.level and problem.nx / 2 >solver_options.minimum_mesh_size:
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

@cuda.jit
def gpu_batch_update(x, r, alphas, previous_d, previous_dr, alpha):
    i = cuda.grid(1)

    if i < x.shape[0]:
        # Use shared memory to accumulate results for x and r
        acc_x = 0.0
        acc_r = 0.0

        for j in range(alphas.size):
            acc_x += alphas[j] * previous_d[j, i] * alpha
            acc_r += alphas[j] * previous_dr[j, i]

        # Update x and r after the full accumulation
        x[i] -= acc_x
        r[i] -= acc_r


def update_vectors_gpu(x, r, alphas, previous_d, previous_dr, alpha):
    if(len(x)<2048):
        for j in range(alphas.size):
            x = x - alphas[j] * previous_d[j] * alpha
            r = r - alphas[j] * previous_dr[j]
    else:
        # Ensure data is on the GPU
        x_gpu = cuda.to_device(x)
        r_gpu = cuda.to_device(r)
        alphas_gpu = cuda.to_device(alphas)
        previous_d_gpu = cuda.to_device(previous_d)
        previous_dr_gpu = cuda.to_device(previous_dr)
    
        # Set up the thread and block configuration
        threads_per_block = 256
        blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block
    
        # Launch the kernel
        gpu_batch_update[blocks_per_grid, threads_per_block](x_gpu, r_gpu, alphas_gpu, previous_d_gpu, previous_dr_gpu, alpha)
    
        # Copy the updated data back to the host
        x = x_gpu.copy_to_host()
        r = r_gpu.copy_to_host()

    return x, r


def solve(x_0,problem,solver_options,enable_preconditionner=True,off_set=np.zeros(0)):
    """
    Solves a nonlinear problem using an adapted GMRES (Generalized Minimal Residual) iterative solver. 
    This function is designed to handle large, sparse systems of equations, typical in discretized partial differential equations (PDEs), 
    with a specific focus on nonlinearity.

    Parameters:
    - x_0 (array_like): Initial guess for the solution vector.
    - problem (object): An object encapsulating the problem to be solved, supporting interface methods for calculation.
    - solver_options (object): Options to control the solver behavior, including tolerance, verbosity, max iterations, 
                               and other solver-specific parameters.
    - enable_preconditioner (bool, optional): Flag to enable the use of a preconditioner to improve convergence efficiency. Defaults to True.
    - off_set (array_like, optional): An optional offset vector to adjust the problem's calculations. Defaults to an empty numpy array.

    Returns:
    - x (array_like): The computed solution vector after convergence or the last iteration.
    - R_norm (float): The norm of the residual for the computed solution, indicating the solution's accuracy.

    Features:
    - **Preconditioning**: Utilizes a Geometric Multigrid (GMG) preconditioner by default to enhance the convergence speed, 
                           especially in the presence of varying scales within the problem.
    - **Iterative Correction**: Applies corrections based on the computation of residuals and their norms, leveraging preconditioning 
                                and correction vectors dynamically for each iteration.
    - **Nonlinear Management**: Adapts to non-linearities in the problem through perturbation of residuals, orthogonalization of 
                                correction vectors, and direct evaluation of residuals, ensuring stability and accuracy.
    - **Performance Metrics**: Includes detailed logging of time spent on various stages of the computation, providing insights 
                               into the solver's performance and allowing for targeted optimization.

    Example:
    ```python
    # Define x_0, problem, and solver_options according to your specific problem
    problem=Problem(dim=2,nx=n,ny=n,nz=n)
    x=np.ones(problem.size)*0
    prep_options=Preconditionner_option(level=2,alpha=1,iterations_for_smoothing=5,frequency_of_residual_direct_evaluation=10,tol=1e-6,max_iterations=1000,max_krylov_vectors=1000, verbosity=True)
    x,r=solve(x,problem,prep_options)
    ```

    Note:
    This adaptation of GMRES for nonlinear problems incorporates checks on the system's nonlinearity. These checks involve directly
    evaluating the system's residual and comparing it to the residual as predicted by the GMRES updates. If the relative discrepancy 
    is too large, indicating that the currently built Krylov subspace is no longer valid, it is discarded, and the GMRES process is restarted.
    """
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
    non_linear_index=0 #(NLI) indicate how non linear the solution process is. use to define when to restart the solver.
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
    time_spent_on_orthogonalization=0
    
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
        if(i%solver_options.max_krylov_vectors==0 or non_linear_index>solver_options.non_linearity_index_limit):
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
            start_time_orth = time.time()
            for j in range(len(previous_d)):
                proj = np.dot(d, previous_d[j])
                d -= (proj / np.linalg.norm(previous_d[j])**2) * previous_d[j]
            end_time_orth = time.time()
            time_spent_on_orthogonalization+=end_time_orth-start_time_orth
            
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
        # This step minimize the norm of the vector r-alpha_0*dr_0-alpha_1*dr_1-alpha_1*dr_1-...--alpha_n*dr_n.
        # To do so, you need to evaluate the dot product of the variation vector and the dot product of the variation vector with the current residual.
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
        # for j in range(alphas.size):
        #     x = x - alphas[j] * previous_d[j] * alpha
        #     r = r - alphas[j] * previous_dr[j]
        x,r=update_vectors_gpu(x, r, alphas, previous_d, previous_dr, alpha)
    
        end_time = time.time()
        # Log the time spend on the update of the solution
        time_spent_on_solution_update+=end_time-start_time
        
        start_time = time.time()
        # Log the time spend on the update of the solution
        if(i%solver_options.frequency_of_residual_direct_evaluation==0 or non_linear_index==0):
            # Every few iterations, the residual is re-evaluated directly instead of relying on the residual predicted by the GMRES algorithm. This approach enables accurate monitoring of the system's nonlinearity.
            r_exact=R(x,problem,off_set)
            # Evaluates the nonlinearity index, which is calculated as the norm of the error vector in the residual divided by the norm of the residual vector itself.
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
                print(" coarse grid n= "+str(problem.nx)+" Iteration  "+str(i)+" residue = "+ str(R_norm)+" NLI = "+ str(non_linear_index) )
                
    if(solver_options.verbosity):
        print()      
        print("time_spent_on_perturbation_residual= "+str(time_spent_on_perturbation_residual))
        print("time_spent_on_residual= "+str(time_spent_on_residual))
        print("time_spent_on_preconditioner= "+str(time_spent_on_preconditioner))
        print("time_spent_on_alphas_matrix_assembly= "+str(time_spent_on_alphas_matrix_assembly))
        print("time_spent_on_alphas_matrix_solve= "+str(time_spent_on_alphas_matrix_solve))
        print("time_spent_on_solution_update= "+str(time_spent_on_solution_update))
        print("time_spent_on_orthogonalization= "+str(time_spent_on_orthogonalization))
        total_time_end=time.time()
        print("total time spent for n= " +str(len(x))+" dofs = "+str(total_time_end-total_time_start))
    return x,R_norm


def solve_matrix_gmg(x_0,problem,solver_options,enable_preconditionner=True,off_set=np.zeros(0)):
    """
    Solves matrix using GMRES with GMG preconditionner (Generalized Minimal Residual) iterative solver. 
    This function is designed to handle large, sparse systems of equations, typical in discretized partial differential equations (PDEs).

    Parameters:
    - x_0 (array_like): Initial guess for the solution vector.
    - problem (object): An object encapsulating the problem to be solved, supporting interface methods for calculation.
    - solver_options (object): Options to control the solver behavior, including tolerance, verbosity, max iterations, 
                               and other solver-specific parameters.
    - enable_preconditioner (bool, optional): Flag to enable the use of a preconditioner to improve convergence efficiency. Defaults to True.
    - off_set (array_like, optional): An optional offset vector to adjust the problem's calculations. Defaults to an empty numpy array.

    Returns:
    - x (array_like): The computed solution vector after convergence or the last iteration.
    - R_norm (float): The norm of the residual for the computed solution, indicating the solution's accuracy.

    Features:
    - **Preconditioning**: Utilizes a Geometric Multigrid (GMG) preconditioner by default to enhance the convergence speed, 
                           especially in the presence of varying scales within the problem.
    - **Iterative Correction**: Applies corrections based on the computation of residuals and their norms, leveraging preconditioning 
                                and correction vectors dynamically for each iteration.
    - **Performance Metrics**: Includes detailed logging of time spent on various stages of the computation, providing insights 
                               into the solver's performance and allowing for targeted optimization.

    Example:
    ```python
    # Define x_0, problem, and solver_options according to your specific problem
    problem=Problem(dim=2,nx=n,ny=n,nz=n)
    x=np.ones(problem.size)*0
    prep_options=Preconditionner_option(level=2,alpha=1,iterations_for_smoothing=5,frequency_of_residual_direct_evaluation=10,tol=1e-6,max_iterations=1000,max_krylov_vectors=1000, verbosity=True)
    x,r=solve_matrix_gmg(x,problem,prep_options)
    ```
    """
    # Start timer
    total_time_start=time.time()
    # Check if an off_set is defined if not then create a vector of zeros of the right size
    if len(off_set)==0:
        off_set=np.zeros(len(x_0))
    # Initialized the structure and residual used in the solver
    x=np.copy(x_0)*0      
    start_time = time.time()  
    residue=R(x_0,problem,off_set)
    Jac=construct_jacobian(x_0, problem)
    r=residue-Jac@x    
    end_time = time.time() 
    R_norm=np.linalg.norm(r)
    R_norm_0=R_norm
    if(solver_options.verbosity):
        print("    GMRES Initial residual = "+str(R_norm_0))
    previous_d=[]
    previous_dr=[]
    previous_dr_dot_product=[]

    # Initialized the preconditionner object
    preconditioner=Preconditioner_GMG_Matrix()
    
    # Initialized a bunch of timer counter
    time_spent_on_perturbation_residual=0
    time_spent_on_residual=0
    time_spent_on_preconditioner=0
    time_spent_on_alphas_matrix_assembly=0
    time_spent_on_alphas_matrix_solve=0
    time_spent_on_solution_update=0
    time_spent_on_matrix_and_residual_evaluation=0
    
    # Set the size of the pertubation vector with alpha and the tolerance
    max_tol=max(1e-12,solver_options.tol*R_norm_0) 
    #Initialized the iteation counter
    i=0
    
    # Initialized the matrix and rhs with the initial solution
    time_spent_on_matrix_and_residual_evaluation+=end_time-start_time
    
    while R_norm>max_tol and i<solver_options.max_iterations:

        start_time = time.time()
        # define new direction based on preconditioner. We apply the GMG preconditioner once every few iterations. For the other iteration we smooth using no preconditionning.
        if((i%int(solver_options.iterations_for_smoothing)==0 ) and enable_preconditionner==True ):
            # We solve the problem as a jacobian so the state of the problem is constant between the iteations and is around x_0.
            d=preconditioner.apply_preconditioner(x_0,r,problem.clone(),solver_options.clone(),off_set)  
        else:
            d=r/np.linalg.norm(r)
        end_time = time.time() 
        # log the time pass on the preconditioner
        time_spent_on_preconditioner+=end_time-start_time
        
        start_time = time.time()
        # Check if the krylov vector space should be reinitialized either because the non linear index is to high which indicate that previous direction wont be usefull in the evaluation of the correction. Or that the maximum number of vector as been reach.
        if(i%solver_options.max_krylov_vectors==0 ):
            # Clear the previous direction ,the variation vector and the variation vector dot product matrix 
            previous_d=[]
            previous_dr=[]
            previous_dr_dot_product=[]
            # Add the new direction
            previous_d.append(d)
            # Calculate the variation vector
            dr=-Jac@d
        else:
            # Orthogonalized the correction vector with previous vectors
            for j in range(len(previous_d)):
                d=d-np.dot(d,previous_d[j])*previous_d[j]/np.linalg.norm(previous_d[j])**2
                d=d/np.linalg.norm(d)
            # Add the new direction
            previous_d.append(d)
            # Calculate the variation vector
            dr=-Jac@d
        
        end_time = time.time()
        # log time pass on the evaluation of the residual perturbation
        time_spent_on_perturbation_residual+=end_time-start_time
        
        # Append the new variation vector with the previous ones
        previous_dr.append(dr)
        
        start_time = time.time()
        # Initialized the variation vector dot product matrix and Right hand side (RHS) used to evaluate the alphas.
        # This step minimize the norm of the vector r-alpha_0*dr_0-alpha_1*dr_1-alpha_1*dr_1-...--alpha_n*dr_n.
        # To do so, you need to evaluate the dot product of the variation vector and the dot product of the variation vector with the current residual (only one will be non zero.)
        A=np.zeros((len(previous_dr),len(previous_dr)))
        rhs=np.zeros(len(previous_dr))
        for j in range(len(previous_dr)):
            for k in range(j+1):
                if(j==len(previous_dr)-1):
                    dot_product=np.dot(previous_dr[j],previous_dr[k])
                    A[j][k]=dot_product
                    A[k][j]=dot_product
                    if(k==0):
                        # Only the last variation vector has a none zero dot product with the residual vector.
                        rhs[-1]=np.dot(previous_dr[-1],r)
                        previous_dr_dot_product.append([])
                    # Store the matrix entry for the next iteration to avoid recalculating the dot products.
                    previous_dr_dot_product[j].append(dot_product)
                else:
                    # Use previously stored matrix entries
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
            x = x - alphas[j] * previous_d[j]
            r = r - alphas[j] * previous_dr[j]
    
        end_time = time.time()
        # Log the time spend on the update of the solution
        time_spent_on_solution_update+=end_time-start_time
        
        start_time = time.time()
        # Log the time spend on the update of the solution
        if(i%solver_options.frequency_of_residual_direct_evaluation==0 ):
            # Every few iterations, the residual is re-evaluated directly instead of relying on the residual predicted by the GMRES algorithm. This approach enables accurate monitoring of the system's nonlinearity.
            r_exact=residue-Jac@x
            r=r_exact
            
        R_norm=np.linalg.norm(r)
        end_time = time.time()
        time_spent_on_residual+=end_time-start_time

        i=i+1
        if(solver_options.verbosity):
            if(enable_preconditionner==True and solver_options.verbosity):
                print("    GMRES Iteration  "+str(i)+" residue = "+ str(R_norm) )
            else:
                print("        coarse grid n= "+str(problem.nx)+" Iteration  "+str(i)+" residue = "+ str(R_norm) )
                
    if(solver_options.verbosity):
        print()      
        print("    time_spent_on_perturbation_residual= "+str(time_spent_on_perturbation_residual))
        print("    time_spent_on_residual= "+str(time_spent_on_residual))
        print("    time_spent_on_preconditioner= "+str(time_spent_on_preconditioner))
        print("    time_spent_on_alphas_matrix_assembly= "+str(time_spent_on_alphas_matrix_assembly))
        print("    time_spent_on_alphas_matrix_solve= "+str(time_spent_on_alphas_matrix_solve))
        print("    time_spent_on_solution_update= "+str(time_spent_on_solution_update))
        print("    time_spent_on_matrix_and_residual_evaluation= "+str(time_spent_on_matrix_and_residual_evaluation))
        total_time_end=time.time()
        print("    total time spent for n= " +str(len(x))+" dofs = "+str(total_time_end-total_time_start))
    return x,R_norm
    

def solve_jac(x_0,problem,solver_options):
    """
    Solves a nonlinear problem using a standard jacobian approach
    """
    # Start timer
    total_time_start=time.time()
    # Check if an off_set is defined if not then create a vector of zeros of the right size
    
    off_set=np.zeros(len(x_0))
    # Initialized the structure and residual used in the solver
    x=np.copy(x_0)
    r=R(x_0,problem,off_set)
    R_norm=np.linalg.norm(r)
    R_norm_0=R_norm
    previous_r_norm=R_norm_0
    if(solver_options.verbosity):
        print("NNL Initial residual = "+str(R_norm_0)) # Newton non linear (NNL) initial residual
        
    print_gmg_gmres_iterations=False
    gmg_gmres_tol=1e-4 # For Problem 3 use gmg_gmres_tol=1e-6, For Problem 4 and 5 use gmg_gmres_tol=5e-2
    #Initialize the iterative solver options
    options_for_matrix=Preconditionner_option(level=2,alpha=0.0001,iterations_for_smoothing=5,frequency_of_residual_direct_evaluation=100,tol=gmg_gmres_tol,max_iterations=1000,max_krylov_vectors=1000,minimum_mesh_size=8,non_linearity_index_limit=0.5, verbosity=print_gmg_gmres_iterations)
    # Initialized a bunch of timer counter
    time_spent_on_residual=0
    
    # Set the size of the pertubation vector with alpha and the tolerance
    max_tol=max(1e-12,solver_options.tol*R_norm_0) 
    #Initialized the iteation counter
    i=0
    
    while R_norm>max_tol and i<solver_options.max_iterations:
        
       
        # Solve update using GMRES with GMG preconditionner
        delta_x,r=solve_matrix_gmg(x,problem,options_for_matrix)
        # Uncomment to solve update using direct solver or gmres without preconditionner 
        # residue=R(x,problem,off_set)
        # Jac=construct_jacobian(x, problem)
        # delta_x=spsolve(Jac,residue) # Direct solver
        # delta_x=gmres(Jac,residue) # Direct solver
       
        x-=delta_x 
        start_time = time.time()
        r=R(x,problem,off_set)
        R_norm=np.linalg.norm(r)
        alpha=1.
        while(previous_r_norm<R_norm and alpha>0.001953125):
            x+=delta_x*alpha
            alpha=alpha*0.5
            x-=delta_x*alpha
            
            r=R(x,problem,off_set)
            R_norm=np.linalg.norm(r)
            print("alpha "+str(alpha)+" "+str(R_norm))
        previous_r_norm= R_norm
        end_time = time.time()
        time_spent_on_residual+=end_time-start_time

        if(solver_options.verbosity):
            print("NNL Iteration  "+str(i)+" residue = "+ str(R_norm) )
            
        i=i+1
                 
    if(solver_options.verbosity):
        print()      
        print("time_spent_on_residual= "+str(time_spent_on_residual))
        total_time_end=time.time()
        print("total time spent for n= " +str(len(x))+" dofs = "+str(total_time_end-total_time_start))
    return x,R_norm

plt.close('all')


############################################### Run code ################################################# 

#probleme size
n=100
# Equation solved D * (∇²x) + C + E * exp(F * x) - G * x  ∇ • x = 0,

# Problem 1
#problem=Problem(dim=2,nx=n,ny=n,nz=n,C=1,D=1,E=1,F=1,G=0) # (Change gmg_gmres_tol to 1e-4 in solve_jac)

# Problem 2
#problem=Problem(dim=3,nx=n,ny=n,nz=n,C=1,D=1,E=1,F=1,G=0) # (Change gmg_gmres_tol to 1e-4 in solve_jac)

# Problem 3
problem=Problem(dim=3,nx=n,ny=n,nz=n,C=1,D=1,E=0,F=0,G=0) # Solvable with non linear GMG gmres solver not with jacobian non linear solver (Change gmg_gmres_tol to 1e-6 in solve_jac)

# Problem 4
#problem=Problem(dim=2,nx=n,ny=n,nz=n,C=1,D=1,E=1,F=1,G=10)  #Faster with non linear GMG gmres solver then jacobian non linear solver (Change gmg_gmres_tol to 5e-2 in solve_jac)

# Problem 5
#problem=Problem(dim=3,nx=n,ny=n,nz=n,C=1,D=1,E=1,F=1,G=10) #Faster jacobian non linear solver then GMG gmres non linear solver. (Change gmg_gmres_tol to 5e-2 in solve_jac)

# Problem 6
#problem=Problem(dim=2,nx=n,ny=n,nz=n,C=1,D=1,E=1,F=1,G=100) # Solvable with non linear GMG gmres solver not with jacobian non linear solver (Change gmg_gmres_tol to 5e-2 in solve_jac)

# Problem 6
#problem=Problem(dim=3,nx=n,ny=n,nz=n,C=1,D=1,E=1,F=1,G=100) # Solvable with non linear GMG gmres solver and with jacobian non linear solver (barely) (Change gmg_gmres_tol to 5e-2 in solve_jac)

# Setup the solver options see class definition for more details
options=Preconditionner_option(level=2,alpha=0.0001,iterations_for_smoothing=5,frequency_of_residual_direct_evaluation=10,tol=1e-6,max_iterations=1000,max_krylov_vectors=1000,minimum_mesh_size=4,non_linearity_index_limit=0.5, verbosity=True)

# Initialized x
x=np.ones(problem.size)*0
# Solve the problem 

# nonlinear GMG_GMRES solver
x_1,r=solve(x,problem,options)

print("Reset solution and try using Jacobian with gmg gmres matrix solver.")
x=np.ones(problem.size)*0
x_2,r=solve_jac(x,problem,options)

x=x_1

e=x_2-x_1
print("norm of difference between the two methods = "+str(np.linalg.norm(e)))

# Graph solution if n smaller then 100 otherwise graph are to heavy.
if(len(x)<20000):
    if problem.dim==2:
        fig,ax = plt.subplots()
        fig1,ax1 = plt.subplots()
        results = x.reshape(problem.nx,problem.ny).transpose()
        X = [0,1]
        Y = [0,1]
            
        im3 = ax.imshow(results, extent = [min(X),max(X),min(Y),max(Y)],aspect="auto")    
        
        multiplicator=2
        
        nx2=int(problem.nx*multiplicator)
        ny2=int(problem.ny*multiplicator)
        coarse_grid=x.reshape((problem.nx, problem.ny), order='F')
        fine_grid=bilinear_interpolate(coarse_grid, nx2, ny2)
        x=fine_grid.flatten(order='F')
        
        results = x.reshape(int(problem.ny*multiplicator),int(problem.nx*multiplicator)).transpose()
        X = [0,1]
        Y = [0,1]
            
        im4 = ax1.imshow(results, extent = [min(X),max(X),min(Y),max(Y)],aspect="auto")
        
    if problem.dim==3:
        x_0=np.copy(x)
        fine_grid=x.reshape((problem.nx, problem.ny, problem.nz), order='F')
        # Get the shape for later use
        nz, ny, nx = fine_grid.shape
        
        # Generate indices for the grid
        indices = np.indices((nz, ny, nx))
        
        # Normalize indices to the range of their respective dimensions
        # Instead of directly multiplying, we normalize each component separately
        x = indices[2] * (nx - 1) / (nx - 1)
        y = indices[1] * (ny - 1) / (ny - 1)
        z = indices[0] * (nz - 1) / (nz - 1)
        values = fine_grid.flatten( order='F')
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use the value as the color
        colors = plt.cm.jet(values / max(values))
        
        # Plotting
        sc = ax.scatter(x, y, z, c=colors, s=100, cmap='jet')
        
        # Colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Value')
        
        # Labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Array Visualization')
        
        multiplicator=2
        
        x=x_0
        nx2=int(problem.nx*multiplicator)
        ny2=int(problem.ny*multiplicator)
        nz2=int(problem.nz*multiplicator)
        coarse_grid=x.reshape((problem.nx, problem.ny,problem.nz), order='F')
        fine_grid=trilinear_interpolate(coarse_grid, nx2, ny2, nz2)
        x=fine_grid.flatten(order='F')
        
        fine_grid=x.reshape((problem.nx*multiplicator, problem.ny*multiplicator, problem.nz*multiplicator), order='F')
        # Get the shape for later use
        nz, ny, nx = fine_grid.shape
        
        # Generate indices for the grid
        indices = np.indices((nz, ny, nx))
        
        # Normalize indices to the range of their respective dimensions
        # Instead of directly multiplying, we normalize each component separately
        x = indices[2] * (nx - 1) / (nx - 1)
        y = indices[1] * (ny - 1) / (ny - 1)
        z = indices[0] * (nz - 1) / (nz - 1)
        values = fine_grid.flatten( order='F')
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use the value as the color
        colors = plt.cm.jet(values / max(values))
        
        # Plotting
        sc = ax.scatter(x, y, z, c=colors, s=100, cmap='jet')
        
        # Colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Value')
        
        # Labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Array Visualization')


