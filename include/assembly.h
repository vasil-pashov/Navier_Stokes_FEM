#pragma once
#include <sparse_matrix_math/sparse_matrix_math.h>
#include <grid.h>
#include "static_matrix.h"

namespace NSFem {

using real = double;

/// Structure to represent first order polynomial shape functions for triangular elements
struct P1 {
    /// The dimension of the element space
    static constexpr int dim = 2;
    /// The number of shape functions (in eval)
    static constexpr int size = 3;
    /// The total number of functions evaluated by the del operator
    static constexpr int delSize = P1::dim * P1::size;
    /// Shape functions for 2D triangluar element with degrees of freedom in each triangle node
    /// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
    /// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
    /// @param[out] out - The value of each shape function at (xi, eta). The array must have at least 6 elements.
    /// Order: [0, 1, 2] - The values at the nodes of the triangle
    static constexpr void eval(const real xi, const real eta, real (&out)[P1::size]) {
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
    static constexpr void del([[maybe_unused]]const real xi, [[maybe_unused]]const real eta, real (&out)[P1::delSize]) {
        // dpsi/dxi
        out[0] = -1.0f;
        out[1] = 1.0f;
        out[2] = 0.0f;

        // dpsi/deta
        out[3] = -1.0f;
        out[4] = 0.0f;
        out[5] = 1.0f;
    }
};

/// Structure to represent second order polynomial shape functions for triangular elements
struct P2 {
    /// The dimension of the element space
    static constexpr int dim = 2;
    /// The number of shape functions (in eval)
    static constexpr int size = 6;
    /// The total number of functions evaluated by the del operator
    static constexpr int delSize = P2::dim * P2::size;
    /// Shape functions for 2D triangluar element with degrees of freedom in each triangle node and in the middle of each side
    /// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
    /// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
    /// @param[out] out - The value of each shape function at (xi, eta). The array must have at least 6 elements.
    /// Order is as follows:
    /// [0, 1, 2] - The values at the nodes of the triangle
    /// 3 - the value at the point between 1 and 2
    /// 4 - the value at the point between 0 and 2
    /// 5 - the value at the point between 0 and 1
    static constexpr void eval(const real xi, const real eta, real (&out)[P2::size]) {
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
    static constexpr void del(const real xi, const real eta, real (&out)[P2::delSize]) {
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

struct TriangleIntegrator {
    /// The integration will be exact for polynomials which have order less or equal TriangleIntegrator::order
    static constexpr int order = 3;

    template<typename TFunctor, int size>
    constexpr static void integrate(const TFunctor& f, real (&out)[size]) {
        assert(std::all_of(&out[0], &out[0] + size, [](const real x){return x == real(0);}));
        real tmp[size] = {};
        for(int i = 0; i < numIntegrationPoints; ++i) {
            const real x = nodes[2 * i];
            const real y = nodes[2 * i + 1];
            f(x, y, tmp);
            for(int j = 0; j < size; ++j) {
                out[j] += tmp[j] * weights[i];
            }
        }
    }

    template<typename TFunctor, int rows, int cols>
    constexpr static void integrate(const TFunctor& f, StaticMatrix<real, rows, cols>& out) {
        assert(std::all_of(out.data(), out.data() + out.getRows() * out.getCols(), [](const real x){return x == real(0);}));
        StaticMatrix<real, rows, cols> tmp;
        for(int i = 0; i < numIntegrationPoints; ++i) {
            const real x = nodes[2 * i];
            const real y = nodes[2 * i + 1];
            f(x, y, tmp);
            out += tmp;
        }
    }
private:
    static const int numIntegrationPoints = 8;
    static constexpr real nodes[2 * TriangleIntegrator::numIntegrationPoints] = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        0.5, 0.0,
        0.5, 0.5,
        0.0, 0.5,
        1.0/3.0, 1.0/3.0
    };
    static constexpr real weights[TriangleIntegrator::numIntegrationPoints] = {
        3.0 / 120.0, 3.0 / 120.0, 3.0 / 120.0,
        8.0 / 120.0, 8.0 / 120.0, 8.0 / 120.0,
        27.0 / 120.0
    };
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

    /// Handles assembling of a general stiffness matrix. It precomputes the integrals of each pair
    /// shape function and then calls assembleMatrix with functor which takes advantage of this optimization
    /// @tparam[Shape] The class representing the shape functions which are going to be used to assemble this matrix
    /// @param[out] out The resulting stiffness matrix
    template<typename Shape>
    void assembleStiffnessMatrix(SMM::CSRMatrix& out);

    /// Handles assembling of the velocity mass matrix. It precomputes the integrals of each pair
    /// shape function and then calls assembleMatrix with functor which takes advantage of this optimization
    void assemblVelocityMassMatrix();

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

    template<typename Shape>
    void NavierStokesAssembly::assembleStiffnessMatrix(SMM::CSRMatrix& out) {
        // Compute the mass matrix. Local stiffness matrix is of the form Integral(Transpose(B.DPSI(xi, eta)/|J|).B.DPSI(xi, eta)/|J| * abs(|J|) dxi, deta),
        // |J| cancel out to produce more readable result 1/abs(|J|) * Integral(Transpose(B.PSI(xi, eta)).B.PSI(xi, eta) * dxi, deta)
        // Where DPSI(xi, eta) = {dpsi_1(xi, eta)/dxi, ..., dpsi_n(xi, eta)/dxi, dpsi_1(xi, eta)/deta ... dpsi_n(xi, eta)/deta} is a
        // row vector containing the gradient of each shape function, first are the derivatives in the xi direction then are the derivatives
        // in the eta direction |J| is the determinant of the linear transformation to the unit triangle and B/|J| is a 2x2 matrix which
        // represents the Grad operator in terms of xi and eta. As with mass matrix note, that B and |J| do not depend on xi and eta, only
        // the gradient of the shape functions dependon xi and eta. Thus we can precompute all pairs of shape function integrals and reuse
        // them in each element. The main complexity comes from the matrix B, which multiples the shape functions.

        // Compute the integral of each pair shape function derivatives.
        const auto squareDelP = [](const real xi, const real eta, StaticMatrix<real, Shape::delSize, Shape::delSize>& out) -> void {
            real delP2[Shape::delSize] = {};
            Shape::del(xi, eta, delP2);
            for(int i = 0; i < Shape::delSize; ++i) {
                for(int j = 0; j < Shape::delSize; ++j) {
                    out[i][j] = delP2[i] * delP2[j];
                }
            }
        };
        StaticMatrix<real, Shape::delSize, Shape::delSize> delPSq;
        TriangleIntegrator::integrate(squareDelP, delPSq);

        const auto localStiffness = [&delPSq](
            [[maybe_unused]]const int* elementIndexes,
            const real* elementNodes,
            real localMatrixOut[Shape::size][Shape::size]
        ) -> void {
            StaticMatrix<real, Shape::dim, Shape::dim> b;
            real J;
            differentialOperator(elementNodes, J, b);
            J = real(1.0) / std::abs(J);
            for(int i = 0; i < Shape::size; ++i) {
                for(int j = 0; j < Shape::size; ++j) {
                    localMatrixOut[i][j] = real(0);
                    const real sq[4] = {
                        delPSq[i][j],
                        delPSq[i][Shape::size + j],
                        delPSq[Shape::size + i][j],
                        delPSq[Shape::size + i][Shape::size + j]
                    };
                    for(int k = 0; k < 2; ++k) {
                        // This function takes advantage that in the integral for the local matrix only del(DPSI) depends on xi and eta
                        // The problem is the matrix B which represents the Grad operator. We have Transpose(B.DPSI(xi, eta)).B.DPSI(xi, eta) = 
                        // Transpose(DPSI).Transpose(B).B.DPSI. Let us denote U = Transpose(DPSI).Transpose(B) and V = B.DPSI
                        // U[i][j] = Sum(B[j][k]*DPSI[k][i], k=0, 1) = Sum(Transpose(DPSI)[i][k]*Transpose(B)[k][j], k = 0, 1) = 
                        // V[i][j] = Sum(B[i][k]*DPSI[k][j], k=0, 1)
                        // Let us denote the result - localMatrixOut with R = U.V
                        // R[i][j] = Sum(U[i][k]*V[k][j], k=0, 1)
                        // R[i][j] = Sum(Sum(DPSI[k'][i]*B[k][k'], k'=0, 1) * Sum(B[k][k'] * DPSI[k']j[],k'=0, 1), k=0, 1)
                        // When we expand the two sums and the multiplication we get the expression for localMatrixOut
                        // This way we have separated the pairs of shape function derivatives and we can use the precomputed values
                        localMatrixOut[i][j] += sq[0]*b[k][0]*b[k][0] + sq[1]*b[k][0]*b[k][1] + sq[2]*b[k][0]*b[k][1] + sq[3]*b[k][1]*b[k][1];
                    }
                    localMatrixOut[i][j] *= J;
                }
            }
        };

        assembleMatrix<decltype(localStiffness), Shape::size, Shape::size>(localStiffness, out);
    }

}