# GPU implementation of the FEM for the Navier-Stokes equations

This repository contains my MSc thesis on the subject of using GPUs to solve the Navier-Stokes equations using the Finite Element Method. For a brief summary check [below](#summary), for the full document check [MSc_Vasil_Pashov.pdf](MSc_Vasil_Pashov.pdf). This repo contains both CPU and GPU implementation of the described methods.

## Summary
### Navier-Stokes Equations
The Navier-Stokes equations are a system of non-linear partial differential equations which describe the flow incompressible Newtonian fluids.

$$
\begin{align}
  % Conservation of momentum
  \frac{\partial \mathbf{u}(\mathbf{x}, t)}{\partial t} + \left(\mathbf{u}(\mathbf{x}, t)\cdot\nabla\right)\mathbf{u}(\mathbf{x}, t) + \nabla p(\mathbf{x}, t) - \nu\Delta\mathbf{u}(\mathbf{x}, t) &= \mathbf{f}(\mathbf{x}, t)\\
  % Continuity
  \nabla \cdot \mathbf{u}(\mathbf{x}, t) &= 0
\end{align}
$$

The first equation is known as the Conservation of momentum. It can be derived from the Caucy's equations (Newton's second law). It consists of three major terms, diffusion, advection and pressure.

The second equation is known as the Continuity equation. It enforces the incompressibility of the fluid.

### The Finite Element Method (FEM)
The FEM is a method used to solve differential equations, on of the advantages of the method is that it can be used in complex geometries. Not all elements are appropriate for solving the Navier-Stokes equations. In order for an element to be stable if has to conform to the LBB condition. In this work we use P2-P1 Tylor-Hood element. It has 6 degrees of freedom (one in each vertex and one in the middle of each side), the pressure has 3 degrees of freedom (on in each vertex).

### Operator Splitting
Solving the equations as-is is a difficult problem, applying the FEM directly would require to solve non-linear system of equations. Thus we split the system of equations into three parts:
* Advection - solved via the semi-Lagrangian method
* Diffusion - solved via FEM, the linear system is solved via Conjugate Gradient Method
* Pressure Projection - imposes the incompressibility, the advection must always be performed in a divergence free velocity field. This basically requires solving a linear system of equations via the Conjugate Gradient Method

### Conclusions
* The semi-Lagrangian method used for the advection is perfect for GPU implementation, however only small portion of the time is spent in it.
* Most of the time is spent in the Conjugate Gradient method. Megakernel implementation of the method (with a global thread lock inside the kernel) is far superior to multikernel implementation.

### Further Research
* Better algorithm for building the KD Tree for the advection
* Use second order operator splitting (Strang splitting)
* Research suitable preconditioners. The IL0 and Incomplete Cholesky are slow to build and apply and are not trivial to implement in multithreaded environment.

### Limitations
* Only Dirichlet and homogeneous second order boundary conditions are allowed.
* Cannot mesh the geometry it has to be split into triangles

### Dependencies
* Check the [conanfile.txt](conanfile.txt).
* The [Sparse Matrix Math](https://github.com/vasil-pashov/sparse_matrix_math/tree/master) library is added as a submodule

### Source code map
You can start reading from [main.cpp](cpp/main.cpp), the assembling of the FEM matrices and the solution of the equations happens in [assembly.h](include/assembly.h). The CPU implementation of the Conjugate Gradient method is implemented in the [Sparse Matrix Math](https://github.com/vasil-pashov/sparse_matrix_math/tree/master) library. The GPU implementation is in [gpu](gpu), [gpu_cpu_shared](gpu_cpu_shared) contains headers with structures which can be used both on CPU and on GPU. Most of the GPU device setup happens in [gpu_host_common.cpp](cpp/gpu_host_common.cpp) and [gpu_simulation_device.cpp](cpp/gpu_simulation_device.cpp), [expression.h](include/expression.h) and [expression.cpp](cpp/expression.cpp) contain expression tree which parses simple math formulas and can then plug variables into them and compute the result, it's used for the boundary conditions.
