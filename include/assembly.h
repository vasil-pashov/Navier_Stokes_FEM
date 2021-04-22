#pragma once
#include <sparse_matrix_math/sparse_matrix_math.h>
#include <grid.h>

namespace NSFem {

using real = double;

class NavierStokesAssembly {
public:
    NavierStokesAssembly(FemGrid2D&& grid, const real dt, const real viscosity);
    void assemble();
private:
    /// Unstructured triangluar grid where the fulid simulation will be computed
    FemGrid2D grid;
    /// Mass matrix for the velocity formed by (fi_i, fi_j) : forall i, j in 0...numVelocityNodes - 1
    /// Where fi_i is the i-th velocity basis function. This matrix is the same for the u and v components of the velocity,
    /// thus we will assemble it only once and use the same matrix to compute all velocity components.
    /// Used to compute the tentative velocity at step i + 1/2. This matrix is constant for the given mesh and does not change
    /// when the time changes.
    SMM::CSRMatrix velocityMassMatrix;
    /// Stiffness is the stiffness matrix for the velocity formed by (del(fi_i), del(fi_j)) : forall i, j in 0...numVelocityNodes - 1
    /// Where fi_i is the i-th velocity basis function and viscosity is the fluid viscosity. This matrix is the same for the u and v
    /// components of the velocity, thus we will assemble it only once and use the same matrix to compute all velocity components.
    /// Used to compute the tentative velocity at step i + 1/2. This matrix is constant for the given mesh and does not change
    /// when the time changes.
    SMM::CSRMatrix stiffnessMatrix;
    /// Convection is the convection matrix formed by (dot(u_h, del(fi_i)), fi_j) : forall i, j in 0...numVelocityNodes - 1
    /// Where fi_i is the i-th velocity basis function and viscosity is the fluid viscosity. This matrix is the same for the u and v
    /// components of the velocity, thus we will assemble it only once and use the same matrix to compute all velocity components.
    /// Used to compute the tentative velocity at step i + 1/2. The matrix depends on the current solution for the velocity, thus it
    /// changes over time and must be reevaluated at each step.
    SMM::CSRMatrix convectionMatrix;
    /// Divergence matrices formed by (dfi_i/dx, chi_j) and (dfi_i/dy, chi_j) : forall i in numVelocityNodes - 1, j in 0...numPressureNodes - 1
    /// Where fi_i is the i-th velocity basis function and chi_j is the j-th pressure basis function
    /// These are used when pressure is found from the tentative velocity. These matrices are constant for the given mesh and do not change
    /// when the time changes. 
    SMM::CSRMatrix divergenceMatrix[2];

    /// Vector containing the approximate solution at each mesh node for the current time step
    /// First are the values in u direction for all nodes and the the values in v direction for all nodes
    /// When using P2-P1 elements the pressure is at the verices of the triangle and the midpoints of each side
    SMM::Vector currentVelocitySolution;

    /// Vector containing the approximate solution for the pressure at each pressure nodes
    /// When using P2-P1 elements the pressure is only at the vertices of the triangle
    SMM::Vector currentPressureSolution;

    template<typename TLocalF, int localRows, int localCols>
    void assembleMatrix(const TLocalF& localFunction, SMM::CSRMatrix& out);

    /// Handles assembling the velocity mass matrix. It precomputes the integrals of each pair
    /// shape function and then calls assembleMatrix with functor which takes advantage of this optimization
    void assemblVelocityMassMatrix();

    /// Handles assembling the velocity mass stiffness. It precomputes the integrals of each pair
    /// shape function and then calls assembleMatrix with functor which takes advantage of this optimization
    void assembleVelocityStiffnessMatrix();

    void assembleConvectionMatrix();

    /// Viscosity of the fluid
    real viscosity;
    /// Size of the time step used when approximating derivatives with respect to time
    real dt;
};

}