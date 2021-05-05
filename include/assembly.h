#pragma once
#include <sparse_matrix_math/sparse_matrix_math.h>
#include <grid.h>

namespace NSFem {

using real = double;

/// Structure to represent first order polynomial shape functions for triangular elements
struct P1 {
    /// The number of shape functions (in eval)
    static constexpr int size = 3;
    /// Shape functions for 2D triangluar element with degrees of freedom in each triangle node
    /// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
    /// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
    /// @param[out] out - The value of each shape function at (xi, eta). The array must have at least 6 elements.
    /// Order: [0, 1, 2] - The values at the nodes of the triangle
    static constexpr void eval(const real xi, const real eta, real out[P1::size]) {
        out[0] = 1 - xi - eta;
        out[1] = xi;
        out[2] = eta;
    }
    /// Compute the derivatives in xi and eta directions, at point (xi, eta) of the shape functions for triangular
    /// element with degrees of freedom in each triangle node
    /// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
    /// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
    /// @param[out] out - This will hold the gradient. The first 3 elements are the derivatives of the shape functions with
    /// respect to xi, next 3 elements are the derivatives of the shape functions with respect to eta
    static constexpr void del([[maybe_unused]]const real xi, [[maybe_unused]]const real eta, real out[2][P1::size]) {
        // dpsi/dxi
        out[0][0] = -1.0f;
        out[0][1] = 1.0f;
        out[0][2] = 0.0f;

        // dpsi/deta
        out[1][0] = -1.0f;
        out[1][1] = 0.0f;
        out[1][2] = 1.0f;
    }
};

/// Structure to represent second order polynomial shape functions for triangular elements
struct P2 {
    /// The number of shape functions (in eval)
    static constexpr int size = 6;
    /// Shape functions for 2D triangluar element with degrees of freedom in each triangle node and in the middle of each side
    /// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
    /// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
    /// @param[out] out - The value of each shape function at (xi, eta). The array must have at least 6 elements.
    /// Order is as follows:
    /// [0, 1, 2] - The values at the nodes of the triangle
    /// 3 - the value at the point between 1 and 2
    /// 4 - the value at the point between 0 and 2
    /// 5 - the value at the point between 0 and 1
    static constexpr void eval(const real xi, const real eta, real out[P2::size]) {
        out[0] = 1 - 3 * xi - 3 * eta + 2 * xi * xi + 4 * xi * eta + 2 * eta * eta;
        out[1] = 2 * xi * xi - xi;
        out[2] = 2 * eta * eta - eta;
        out[3] = 4 * xi * eta;
        out[4] = 4 * eta - 4 * xi * eta - 4 * eta * eta;
        out[5] = 4 * xi - 4 * xi * xi - 4 * xi * eta;
    }

    /// Compute the derivatives in xi and eta directions, at point (xi, eta) of the shape functions for triangular 
    ///element with degrees of freedom in each triangle node and in the middle of each side
    /// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
    /// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
    /// @param[out] out - This will hold the gradient. The first 6 elements are the derivatives of the shape functions with
    /// respect to xi, next 6 elements are the derivatives of the shape functions with respect to eta
    static constexpr void del(const real xi, const real eta, real out[2 * P2::size]) {
        // dpsi/dxi
        out[0] = -3 + 4 * eta + 4 * xi;
        out[1] = -1 + 4 * xi;
        out[2] = 0.0f;
        out[3] = 4 * eta;
        out[4] = -4 * eta;
        out[5] = 4 - 4 * eta - 8 * xi;
        // dpsi/deta
        out[6] = -3 + 4 * eta + 4 * xi;
        out[7] = 0;
        out[8] = -1 + 4 * eta;
        out[9] = 4 * xi;
        out[10] = 4 - 8 * eta - 4 * xi;
        out[11] = -4 * xi;
    }
};

class NavierStokesAssembly {
public:
    NavierStokesAssembly(FemGrid2D&& grid, const real dt, const real viscosity);
    void solve(const float totalTime);
private:
    /// Function which wraps the assembling of all constant matrices. These matrices will stay the same
    /// during the whole solving phase. This function must be called once before we start time iterations.
    void assembleConstantMatrices();
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
    SMM::CSRMatrix velocityStiffnessMatrix;
    /// Divergence matrices formed by (dfi_i/dx, chi_j) and (dfi_i/dy, chi_j) : forall i in numVelocityNodes - 1, j in 0...numPressureNodes - 1
    /// Where fi_i is the i-th velocity basis function and chi_j is the j-th pressure basis function
    /// These are used when pressure is found from the tentative velocity. These matrices are constant for the given mesh and do not change
    /// when the time changes. 
    SMM::CSRMatrix divergenceMatrix;

    /// Vector containing the approximate solution at each mesh node for the current time step
    /// First are the values in u direction for all nodes and the the values in v direction for all nodes
    /// When using P2-P1 elements the pressure is at the verices of the triangle and the midpoints of each side
    SMM::Vector currentVelocitySolution;

    /// Vector containing the approximate solution for the pressure at each pressure nodes
    /// When using P2-P1 elements the pressure is only at the vertices of the triangle
    SMM::Vector currentPressureSolution;

    template<typename TLocalF, int localRows, int localCols>
    void assembleMatrix(const TLocalF& localFunction, SMM::CSRMatrix& out);

    /// Handles assembling of the velocity mass matrix. It precomputes the integrals of each pair
    /// shape function and then calls assembleMatrix with functor which takes advantage of this optimization
    void assemblVelocityMassMatrix();

    /// Handles assembling of the velocity mass stiffness. It precomputes the integrals of each pair
    /// shape function and then calls assembleMatrix with functor which takes advantage of this optimization
    void assembleVelocityStiffnessMatrix();

    /// Handles assembling of the convection matrix. It does it directly by the formula and does not use
    /// precomputed integrals. In theory it's possible, but it would make the code too complicated as it
    /// would require combined integral of 3 basis functions (i,j,k forall i,j,k). The convection matrix
    /// depends on the solution at the current time step. It changes at each time step.
    /// @param[out] outConvectionMatrix The resulting convection matrix
    void assembleConvectionMatrix(SMM::CSRMatrix& outConvectionMatrix);

    /// This handles the assembling of the divergence matrix. This function looks a lot like assembleMatrix.
    /// In fact we could split the divergence matrix into two matrices (one for x direction and one for y direction)
    /// each with it's own local function. This way assembleMatrix could be used, but this would mean that we have
    /// to iterate over all elements twice and also offset all indexes by the number of nodes for the second (y drection)
    /// matrix.
    void assembleDivergenceMatrix();

    /// Impose Dirichlet Boundary conditions on the velocity vector passed as an input
    /// @param[in, out] velocityVector Velocity vector where the Dirichlet BC will be imposed
    /// The vector must have all its u-velocity components at the begining, followed by all
    /// v-velocity components.
    void imposeVelocityDirichlet(SMM::Vector& velocityVector);

    /// Viscosity of the fluid
    real viscosity;
    /// Size of the time step used when approximating derivatives with respect to time
    real dt;
};

}