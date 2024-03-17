# nonlinear_iterative_residual_based_solver
 
This project implements a prototype nonlinear solver based on the Generalized Minimal Residual (GMRES) method, enhanced with Geometric Multigrid (GMG) techniques, specifically utilizing a V-cycle approach. The primary focus is on exploring the capabilities of a nonlinear GMG-enhanced GMRES solver in solving equations that exhibit nonlinear behavior. As a part of the project, a comparison is made with a traditional nonlinear solver based on the Newton-Raphson method, where the matrix resolution is also performed using a GMG-enhanced GMRES solver. This setup provides an insightful comparison between two distinct approaches to tackling nonlinear numerical problems. The comparison is performed on this equation:
D⋅(∇^2 x)+C+E⋅exp(F⋅x)−G⋅x⋅∇x=0
