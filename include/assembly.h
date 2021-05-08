#pragma once
#include <sparse_matrix_math/sparse_matrix_math.h>
#include <unordered_set>
#include <grid.h>
#include "static_matrix.h"
#include <string>

namespace NSFem {

using real = double;

/// Find the determinant of the Jacobi matrix for a linear transformation of random triangle to the unit one
/// @param[in] kx World x coordinate of the node which will be transformed to (0, 0)
/// @param[in] ky World y coordinate of the node which will be transformed to (0, 0)
/// @param[in] lx World x coordinates of the node which will be transformed to (1, 1)
/// @param[in] my World y coordinates of the node which will be transformed to (0, 1)
/// @return The determinant of the Jacobi matrix which transforms k, l, m to the unit triangle
inline real linTriangleTmJacobian(const real* elementNodes) {
    const real xk = elementNodes[0];
    const real yk = elementNodes[1];
    const real xl = elementNodes[2];
    const real yl = elementNodes[3];
    const real xm = elementNodes[4];
    const real ym = elementNodes[5];
    // The Jacobi matrix is given by
    // |xl - xk     yl - yk|
    // |xm - xk     ym - yk|
    const real a = xl - xk;
    const real b = yl - yk;
    const real c = xm - xk;
    const real d = ym - yk;
    return a * d - b * c;
}

/// Compute the differential operator:
///                            B
/// |d/dx|            |ym - yk  yk - yl| |d/dxi |              
/// |    | = 1/det(J) |                | |      | 
/// |d/dy|            |xk - xm  xl - xk| |d/deta|
/// Where J is the Jacobi matrix:
/// |dx/dxi   dy/dxi |     |xl - xk     yl - yk|
/// |                |  =  |                   | 
/// |dx/deta  dy/deta|     |xm - xk     ym - yk|
/// @param[in] nodes The coordinates of the tirangle vertices. The array is expected to be of the form
/// (x0, y0), (x1, y1), (x2, y2) and they are supposed to be remapped to (0, 0), (1, 0), (0, 1) in the unit tirangle
/// @param[out] outDetJ The determinant of the Jacobi matrix
/// @param[out] outB The matrix B used in the operator as illustrated above it must be of size 4
template<typename TMatrix>
inline void differentialOperator(const real* nodes, real& outDetJ, TMatrix& outB) {
    const real xk = nodes[0];
    const real yk = nodes[1];
    const real xl = nodes[2];
    const real yl = nodes[3];
    const real xm = nodes[4];
    const real ym = nodes[5];
    outB[0][0] = ym - yk;
    outB[0][1] = yk - yl;
    outB[1][0] = xk - xm;
    outB[1][1] = xl - xk;
    // Important note that the Jacobi matrix and matrix (B * 1/detJ) are actually the inverse of each other
    // The inverse of 2d matrix given by:
    // |a   b|              |d   -b|
    // |     | = a*d * c*b  |      | 
    // |c   d|              |-c   a|
    // Looking at the above picture we can find the determinant of matrix B instead of Jacobi and it will be the same
    outDetJ = outB[0][0] * outB[1][1] - outB[0][1] * outB[1][0];
}

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

    static constexpr void eval(const real xi, const real eta, StaticMatrix<real, 1, size>& out) {
        out[0][0] = 1 - xi - eta;
        out[0][1] = xi;
        out[0][2] = eta;
    }
    /// Compute the derivatives in xi and eta directions, at point (xi, eta) of the shape functions for triangular
    /// element with degrees of freedom in each triangle node
    /// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
    /// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
    /// @param[out] out - This will hold the gradient. The first 3 elements are the derivatives of the shape functions with
    /// respect to xi, next 3 elements are the derivatives of the shape functions with respect to eta
    static constexpr void del(
        [[maybe_unused]]const real xi,
        [[maybe_unused]]const real eta,
        StaticMatrix<real, dim, size>& out
    ) {
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

    static constexpr void eval(const real xi, const real eta, StaticMatrix<real, 1, size>& out) {
        out[0][0] = 1 - 3 * xi - 3 * eta + 2 * xi * xi + 4 * xi * eta + 2 * eta * eta;
        out[0][1] = 2 * xi * xi - xi;
        out[0][2] = 2 * eta * eta - eta;
        out[0][3] = 4 * xi * eta;
        out[0][4] = 4 * eta - 4 * xi * eta - 4 * eta * eta;
        out[0][5] = 4 * xi - 4 * xi * xi - 4 * xi * eta;
    }

    /// Compute the derivatives in xi and eta directions, at point (xi, eta) of the shape functions for triangular 
    ///element with degrees of freedom in each triangle node and in the middle of each side
    /// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
    /// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
    /// @param[out] out - This will hold the gradient. The first 6 elements are the derivatives of the shape functions with
    /// respect to xi, next 6 elements are the derivatives of the shape functions with respect to eta
    static constexpr void del(const real xi, const real eta, StaticMatrix<real, dim, size>& out) {
        // dpsi/dxi
        out[0][0] = -3 + 4 * eta + 4 * xi;
        out[0][1] = -1 + 4 * xi;
        out[0][2] = 0.0f;
        out[0][3] = 4 * eta;
        out[0][4] = -4 * eta;
        out[0][5] = 4 - 4 * eta - 8 * xi;
        // dpsi/deta
        out[1][0] = -3 + 4 * eta + 4 * xi;
        out[1][1] = 0;
        out[1][2] = -1 + 4 * eta;
        out[1][3] = 4 * xi;
        out[1][4] = 4 - 8 * eta - 4 * xi;
        out[1][5] = -4 * xi;
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
        assert(std::all_of(out.begin(), out.end(), [](const real x){return x == real(0);}));
        StaticMatrix<real, rows, cols> tmp;
        for(int i = 0; i < numIntegrationPoints; ++i) {
            const real x = nodes[2 * i];
            const real y = nodes[2 * i + 1];
            f(x, y, tmp);
            out += tmp * weights[i];
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

template<typename VelocityShape, typename PressureShape>
class NavierStokesAssembly {
public:
    NavierStokesAssembly(FemGrid2D&& grid, const real dt, const real viscosity, const std::string& outFolder);
    void solve(const float totalTime);
private:
    /// Unstructured triangluar grid where the fulid simulation will be computed
    FemGrid2D grid;

    /// Mass matrix for the velocity formed by (fi_i, fi_j) : forall i, j in 0...numVelocityNodes - 1
    /// Where fi_i is the i-th velocity basis function. This matrix is the same for the u and v components of the velocity,
    /// thus we will assemble it only once and use the same matrix to compute all velocity components.
    /// Used to compute the tentative velocity at step i + 1/2. This matrix is constant for the given mesh and does not change
    /// when the time changes.
    SMM::CSRMatrix velocityMassMatrix;

    /// Stiffness matrix for the velocity formed by (del(fi_i), del(fi_j)) : forall i, j in 0...numVelocityNodes - 1
    /// Where fi_i is the i-th velocity basis function and viscosity is the fluid viscosity. This matrix is the same for the u and v
    /// components of the velocity, thus we will assemble it only once and use the same matrix to compute all velocity components.
    /// Used to compute the tentative velocity at step i + 1/2. This matrix is constant for the given mesh and does not change
    /// when the time changes.
    SMM::CSRMatrix velocityStiffnessMatrix;

    /// Stiffness matrix for the velocity formed by (del(fi_i), del(fi_j)) : forall i, j in 0...numPressureNodes - 1
    /// Where fi_i is the i-th pressure basis function. This matrix is constant for the given mesh and does not change
    /// when the time changes.
    SMM::CSRMatrix pressureStiffnessMatrix;

    /// Divergence matrices formed by (dfi_i/dx, chi_j) and (dfi_i/dy, chi_j) : forall i in numVelocityNodes - 1, j in 0...numPressureNodes - 1
    /// Where fi_i is the i-th velocity basis function and chi_j is the j-th pressure basis function
    /// These are used when pressure is found from the tentative velocity. These matrices are constant for the given mesh and do not change
    /// when the time changes. 
    SMM::CSRMatrix velocityDivergenceMatrix;

    /// Divergence matrices formed by (fi_i, dchi_j/dx) and (fi_i, dchi_j/dy) : forall i in numVelocityNodes - 1, j in 0...numPressureNodes - 1
    /// Where fi_i is the i-th velocity basis function and chi_j is the j-th pressure basis function
    /// These are used when pressure is found from the tentative velocity. These matrices are constant for the given mesh and do not change
    /// when the time changes. 
    SMM::CSRMatrix pressureDivergenceMatrix;

    /// Vector containing the approximate solution at each mesh node for the current time step
    /// First are the values in u direction for all nodes and the the values in v direction for all nodes
    /// When using P2-P1 elements the pressure is at the verices of the triangle and the midpoints of each side
    SMM::Vector currentVelocitySolution;

    /// Vector containing the approximate solution for the pressure at each pressure nodes
    /// When using P2-P1 elements the pressure is only at the vertices of the triangle
    SMM::Vector currentPressureSolution;

    /// Path to a folder where the result for each iteration will be saved
    std::string outFolder;

    /// Viscosity of the fluid
    real viscosity;

    /// Size of the time step used when approximating derivatives with respect to time
    real dt;

    template<int localRows, int localCols, typename TLocalF>
    void assembleMatrix(const TLocalF& localFunction, SMM::CSRMatrix& out);

    /// Assemble global matrix which will be used to solve problem with Dirichlet conditions. Rows and colums of the
    /// out matrix with indexes matching a boundary nodes will be filled  with zero (except the diagonal elements).
    /// The out matrix will have value one on the diagonal elements corresponding to the boundary nodes.
    /// The elements which were dropped from the out matrix will be added to outBondaryWeights, these weights must
    /// be subtracted from the right hand side when the linear system is solved
    /// @tparam TLocalF Type of the functor which will compute the local matrix
    /// @tparam localRows Number of rows in the local matrix
    /// @tparam localCols Number of columns in the local matrix
    /// @param[in] localFunction Functor which will compute the local matrix
    /// @param[in] boundaryNodes Map between boundary node and the boundary to which it belogns
    /// @param[out] out The result of the assembling. Ones will appear on the main diagonal elements corresponding
    /// to each bondary nodes
    /// @param[out] outBondaryWeights For each boundary this holds the columns which were zeroed out when
    /// the diagonal element was set to 1. The rows corresponding to boundary nodes are ommitted as they are
    /// filled with 0 anyway
    template<int localRows, int localCols, typename TLocalF>
    void assembleBCMatrix(
        const TLocalF& localFunction,
        const std::unordered_set<int>& boundaryNodes,
        SMM::TripletMatrix& out,
        SMM::TripletMatrix& outBondaryWeights
    );

    /// Export the current solutions for velocity and pressure to a file
    void exportSolution(const int timeStep);

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
    template<typename RegularShape, typename DelShape, bool sideBySide>
    void assembleDivergenceMatrix(SMM::CSRMatrix& out);

    /// Impose Dirichlet Boundary conditions on the velocity vector passed as an input
    /// @param[in, out] velocityVector Velocity vector where the Dirichlet BC will be imposed
    /// The vector must have all its u-velocity components at the begining, followed by all
    /// v-velocity components.
    void imposeVelocityDirichlet(SMM::Vector& velocityVector);

    template<typename Shape>
    struct LocalStiffnessFunctor {
        LocalStiffnessFunctor() {
            // Compute the integral of each pair shape function derivatives.
            const auto squareDelP = [](
                const real xi,
                const real eta,
                StaticMatrix<real, Shape::delSize, Shape::delSize>& out
            ) -> void {
                StaticMatrix<real, 2, Shape::size> del;
                Shape::del(xi, eta, del);
                auto it = out.begin();
                for(const auto i : del) {
                    for(const auto j : del) {
                        *it = i * j;
                        ++it;
                    }
                }
            };
            TriangleIntegrator::integrate(squareDelP, delPSq);
        }
        void operator()(
            [[maybe_unused]]const int* elementIndexes,
            const real* elementNodes,
            StaticMatrix<real, Shape::size, Shape::size>& localMatrixOut
        ) const {
            // Compute the mass matrix. Local stiffness matrix is of the form Integral(Transpose(B.DPSI(xi, eta)/|J|).B.DPSI(xi, eta)/|J| * abs(|J|) dxi, deta),
            // |J| cancel out to produce more readable result 1/abs(|J|) * Integral(Transpose(B.PSI(xi, eta)).B.PSI(xi, eta) * dxi, deta)
            // Where DPSI(xi, eta) = {dpsi_1(xi, eta)/dxi, ..., dpsi_n(xi, eta)/dxi, dpsi_1(xi, eta)/deta ... dpsi_n(xi, eta)/deta} is a
            // row vector containing the gradient of each shape function, first are the derivatives in the xi direction then are the derivatives
            // in the eta direction |J| is the determinant of the linear transformation to the unit triangle and B/|J| is a 2x2 matrix which
            // represents the Grad operator in terms of xi and eta. As with mass matrix note, that B and |J| do not depend on xi and eta, only
            // the gradient of the shape functions dependon xi and eta. Thus we can precompute all pairs of shape function integrals and reuse
            // them in each element. The main complexity comes from the matrix B, which multiples the shape functions.
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
        }
    private:
        StaticMatrix<real, Shape::delSize, Shape::delSize> delPSq;
    };

    template<typename RegularShape, typename DelShape>
    class LocalDivergenceFunctor {
    public:
        LocalDivergenceFunctor() {
            auto combine = [](
                const real xi,
                const real eta,
                StaticMatrix<real, RegularShape::size, DelShape::delSize>& out
            ) -> void {
                real p1Res[RegularShape::size] = {};
                RegularShape::eval(xi, eta, p1Res);

                StaticMatrix<real, 2, DelShape::size> delP2Res;
                DelShape::del(xi, eta, delP2Res);

                for(int i = 0; i < RegularShape::size; ++i) {
                    for(int j = 0; j < DelShape::size; ++j) {
                        for(int k = 0; k < 2; ++k) {
                            out[i][j + k * DelShape::size] = p1Res[i] * delP2Res[k][j];
                        }
                    }
                }
            };
            TriangleIntegrator::integrate(combine, combined);
        }
        void operator()(
            [[maybe_unused]]const int* elementIndexes,
            const real* elementNodes,
            StaticMatrix<real, RegularShape::size, DelShape::size>& delXShape,
            StaticMatrix<real, RegularShape::size, DelShape::size>& delYShape
        ) const {
            real J;
            StaticMatrix<real, 2, 2> B;
            differentialOperator(elementNodes, J, B);
            // Compute local matrices
            for(int p = 0; p < RegularShape::size; ++p) {
                for(int q = 0; q < DelShape::size; ++q) {
                    delXShape[p][q] = B[0][0] * combined[p][q] + B[0][1] * combined[p][q + DelShape::size];
                    delYShape[p][q] = B[1][0] * combined[p][q] + B[1][1] * combined[p][q + DelShape::size];
                }
            }
        }
    private:
        /// Matrix to hold combined values for the integrals: Integrate(psi_i(xi, eta) * dpsi_j(xi, eta)/dxi * dxi * deta) and
        /// Integrate(psi(xi, eta) * dpsi(xi, eta)/deta * dxi * deta). Where i is in [0;p1Size-1] and j is in [0;p2Size-1]
        /// The first p2Size entries in each row are the first integral and the second represent the second integral 
        StaticMatrix<real, RegularShape::size, DelShape::delSize> combined;
    };
};

template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::exportSolution(const int timeStep) {
    const int nodesCount = grid.getNodesCount();
     nlohmann::json outJSON = {
        {"u", nlohmann::json::array()},
        {"v", nlohmann::json::array()},
        {"timeStep", timeStep}
    };
    for(int i = 0; i < nodesCount; ++i) {
        outJSON["u"].push_back(currentVelocitySolution[i]);
        outJSON["v"].push_back(currentVelocitySolution[i + nodesCount]);
    }
    for(int i = 0; i < grid.getPressureNodesCount(); ++i) {
        outJSON["p"].push_back(currentPressureSolution[i]);
    }
    // outJSON["timeStep"] = timeStep;
    const std::string& path = outFolder + "/out_" + std::to_string(timeStep) + ".json";
    std::ofstream outFile(path);
    if(outFile.is_open()) {
        outFile  << std::setw(4) << outJSON << std::endl;
    } else {
        assert(false && "Failed to open file for writing the result");
    }
}

template<typename VelocityShape, typename PressureShape>
template<int localRows, int localCols, typename TLocalF>
void NavierStokesAssembly<VelocityShape, PressureShape>::assembleBCMatrix(
    const TLocalF& localFunction,
    const std::unordered_set<int>& boundaryNodes,
    SMM::TripletMatrix& out,
    SMM::TripletMatrix& outBondaryWeights
) {
    // IMPORTANT: This will work for symmetric matrices. It might work for not symmetric matrices
    // but I'm not sure. This procedure will keep the outBondaryWeights weights in transposed manner
    // The elements nodes which are in the same column in out will appear in the same row in outBondaryWeights
    // This is safe for symmetric matrices.
    const int numElements = grid.getElementsCount();
    const int elementSize = std::max(VelocityShape::size, PressureShape::size);
    int elementIndexes[elementSize];
    real elementNodes[2 * elementSize];
    assert(elementSize == grid.getElementSize());
    StaticMatrix<real, localRows, localCols> localMatrix;
    for(int i = 0; i < numElements; ++i) {
        grid.getElement(i, elementIndexes, elementNodes);
        localFunction(elementIndexes, elementNodes, localMatrix);
        for(int localRow = 0; localRow < localRows; ++localRow) {
            const int globalRow = elementIndexes[localRow];
            const bool isRowBoundary = boundaryNodes.find(globalRow) != boundaryNodes.end();
            for(int localCol = 0; localCol < localCols; ++localCol) {
                const int globalCol = elementIndexes[localCol];
                const bool isColBoundary = boundaryNodes.find(globalCol) != boundaryNodes.end();
                if(!isRowBoundary && !isColBoundary) {
                    out.addEntry(globalRow, globalCol, localMatrix[localRow][localCol]);
                } else if(!isRowBoundary && isColBoundary) {
                    outBondaryWeights.addEntry(globalCol, globalRow, localMatrix[localRow][localCol]);
                }
            }
        }
    }
    for(const auto& boundaryNode : boundaryNodes) {
        out.addEntry(boundaryNode, boundaryNode, 1);
    }
}

template<typename VelocityShape, typename PressureShape>
template<int localRows, int localCols, typename TLocalF>
void NavierStokesAssembly<VelocityShape, PressureShape>::assembleMatrix(const TLocalF& localFunction, SMM::CSRMatrix& out) {
    const int numNodes = grid.getNodesCount();
    const int numElements = grid.getElementsCount();
    const int elementSize = std::max(VelocityShape::size, PressureShape::size);
    int elementIndexes[elementSize];
    real elementNodes[2 * elementSize];
    assert(elementSize == grid.getElementSize());
    StaticMatrix<real, localRows, localCols> localMatrix;
    SMM::TripletMatrix triplet(numNodes, numNodes);
    for(int i = 0; i < numElements; ++i) {
        grid.getElement(i, elementIndexes, elementNodes);
        localFunction(elementIndexes, elementNodes, localMatrix);
        for(int localRow = 0; localRow < localRows; ++localRow) {
            const int globalRow = elementIndexes[localRow];
            for(int localCol = 0; localCol < localCols; ++localCol) {
                const int globalCol = elementIndexes[localCol];
                triplet.addEntry(globalRow, globalCol, localMatrix[localRow][localCol]);
            }
        }
    }
    out.init(triplet);
}

template<typename VelocityShape, typename PressureShape>
NavierStokesAssembly<VelocityShape, PressureShape>::NavierStokesAssembly(
    FemGrid2D&& grid,
    const real dt,
    const real viscosity,
    const std::string& outFolder
) :
    grid(std::move(grid)),
    outFolder(outFolder),
    viscosity(viscosity),
    dt(dt)
{ }

template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::solve(const float totalTime) {
    
    assemblVelocityMassMatrix();

    // Assemble global velocity stiffness matrix
    LocalStiffnessFunctor<VelocityShape> localVelocityStiffness;
    assembleMatrix<VelocityShape::size, VelocityShape::size>(localVelocityStiffness, velocityStiffnessMatrix);

    // Assemble global pressure stiffness matrix.
    const int numPressureDirichletBoundaries = grid.getPressureDirichletSize();
    // The key is the node index, the value is which boundary it belongs to
    // This is used during the assembling to check if a node belongs to a boundary
    // Since a node can belong to one and only one boundary we have vector of unordered
    // triplet matrices which will hold the weights
    std::unordered_set<int> allBoundaryNodes;
    FemGrid2D::PressureDirichletConstIt pressureDrichiletIt = grid.getPressureDirichlet();
    for(int i = 0; i < numPressureDirichletBoundaries; ++i, ++pressureDrichiletIt) {
        const int* boundary = pressureDrichiletIt->getNodeIndexes();
        const int boundarySize = pressureDrichiletIt->getSize();
        allBoundaryNodes.insert(boundary, boundary + boundarySize);

    }
    
    SMM::TripletMatrix pressureDirichletWeightsTriplet(grid.getPressureNodesCount(), grid.getPressureNodesCount());
    LocalStiffnessFunctor<PressureShape> localPressureStuffness;
    SMM::TripletMatrix pressureStiffnessTriplet(grid.getPressureNodesCount(), grid.getPressureNodesCount());
    assembleBCMatrix<PressureShape::size, PressureShape::size>(
        localPressureStuffness,
        allBoundaryNodes,
        pressureStiffnessTriplet,
        pressureDirichletWeightsTriplet
    );
    pressureStiffnessMatrix.init(pressureStiffnessTriplet);
    SMM::CSRMatrix pressureDirichletWeights;
    pressureDirichletWeights.init(pressureDirichletWeightsTriplet);
     
    assembleDivergenceMatrix<PressureShape, VelocityShape, true>(velocityDivergenceMatrix);
    assembleDivergenceMatrix<VelocityShape, PressureShape, false>(pressureDivergenceMatrix);

    velocityStiffnessMatrix *= viscosity;
    const real dtInv = real(1) / dt;
    velocityDivergenceMatrix *= -dtInv;
    pressureDivergenceMatrix *= -dt;

    const int steps = totalTime / dt;
    const int nodesCount = grid.getNodesCount();
    currentVelocitySolution.init(nodesCount * 2, 0.0f);
    currentPressureSolution.init(grid.getPressureNodesCount(), real(0));
    imposeVelocityDirichlet(currentVelocitySolution);
    SMM::Vector velocityRhs(nodesCount * 2, 0);
    SMM::Vector pressureRhs(grid.getPressureNodesCount(), real(0));

    std::unordered_map<char, float> pressureVars;

    exportSolution(0);

    for(int timeStep = 1; timeStep < steps; ++timeStep) {
        // Convection is the convection matrix formed by (dot(u_h, del(fi_i)), fi_j) : forall i, j in 0...numVelocityNodes - 1
        // Where fi_i is the i-th velocity basis function and viscosity is the fluid viscosity. This matrix is the same for the u and v
        // components of the velocity, thus we will assemble it only once and use the same matrix to compute all velocity components.
        // Used to compute the tentative velocity at step i + 1/2. The matrix depends on the current solution for the velocity, thus it
        // changes over time and must be reevaluated at each step.
        // TODO: Do not allocate space on each iteration, but reuse the matrix sparse structure
        SMM::CSRMatrix convectionMatrix;
        assembleConvectionMatrix(convectionMatrix);
        assert(convectionMatrix.hasSameNonZeroPattern(velocityMassMatrix) && convectionMatrix.hasSameNonZeroPattern(velocityStiffnessMatrix));
        SMM::CSRMatrix::ConstIterator convectionIt = convectionMatrix.begin();
        SMM::CSRMatrix::ConstIterator massIt = velocityMassMatrix.begin();
        SMM::CSRMatrix::ConstIterator velStiffnessIt = velocityStiffnessMatrix.begin();
        // TODO: The expression for the right-hand side in matrix form is: velocityMass - dt * (viscosity * velocityStiffness + convection)
        // velocity mass and velocity stifness are constant matrices. They can be combined before the iterations start.
        for(;convectionIt != convectionMatrix.end(); ++convectionIt, ++massIt, ++velStiffnessIt) {
            const int row = convectionIt->getRow();
            const int col = convectionIt->getCol();
            const real uVal = currentVelocitySolution[col];
            const real vVal = currentVelocitySolution[col + nodesCount];
            velocityRhs[row] += (massIt->getValue() - dt * (convectionIt->getValue() + velStiffnessIt->getValue())) * uVal;
            velocityRhs[row + nodesCount] += (massIt->getValue() - dt * (convectionIt->getValue() + velStiffnessIt->getValue())) * vVal;
        }
        SMM::SolverStatus solveStatus = SMM::SolverStatus::SUCCESS;

        // Solve for the u component
        solveStatus = SMM::ConjugateGradient(velocityMassMatrix, velocityRhs, currentVelocitySolution, -1, 1e-6);
        assert(solveStatus == SMM::SolverStatus::SUCCESS);

        // Solve for the v component
        solveStatus = SMM::ConjugateGradient(velocityMassMatrix, velocityRhs + nodesCount, currentVelocitySolution + nodesCount, -1, 1e-6);
        assert(solveStatus == SMM::SolverStatus::SUCCESS);

        imposeVelocityDirichlet(currentVelocitySolution);
       
        // Solve for the pressure. As pressure is "implicitly" stepped Dirchlet boundary conditions cannot be imposed after
        // solving the linear system. For this reason the pressure stiffness matrix was tweaked before time iterations begin.
        // Now at each time step the right hand side must be tweaked as well.

        // Find the right hand side
        velocityDivergenceMatrix.rMult(currentVelocitySolution, pressureRhs);

        // Now impose the Dirichlet Boundary Conditions
        pressureDrichiletIt = grid.getPressureDirichlet();
        for(int boundaryIndex = 0; boundaryIndex < numPressureDirichletBoundaries; ++boundaryIndex, ++pressureDrichiletIt) {
            const FemGrid2D::PressureDirichlet& boundary = *pressureDrichiletIt;
            for(int boundaryNodeIndex = 0; boundaryNodeIndex < boundary.getSize(); ++boundaryNodeIndex) {
                const int nodeIndex = boundary.getNodeIndexes()[boundaryNodeIndex];
                const real x = grid.getNodesBuffer()[nodeIndex * 2];
                const real y = grid.getNodesBuffer()[nodeIndex * 2 + 1];
                pressureVars['x'] = x;
                pressureVars['y'] = y;
                float pBoundary = 0;
                boundary.eval(&pressureVars, pBoundary);
                pressureRhs[nodeIndex] = pBoundary;
                SMM::CSRMatrix::ConstRowIterator it = pressureDirichletWeights.rowBegin(nodeIndex);
                const SMM::CSRMatrix::ConstRowIterator end = pressureDirichletWeights.rowEnd(nodeIndex);
                while(it != end) {
                    pressureRhs[it->getCol()] -= it->getValue() * pBoundary;
                    ++it;
                }
            }
        }

        // Finally solve the linear system for the pressure
        solveStatus = SMM::ConjugateGradient(pressureStiffnessMatrix, pressureRhs, currentPressureSolution, -1, 1e-6);
        assert(solveStatus == SMM::SolverStatus::SUCCESS);

        // Combine the tentative velocity and the pressure to find the real velocity at time step i + 1
        // TODO: If we split the solution to velocityMassMatrix(x1 + x2) = b1 + b2 = 
        // velocityMassMatrix.(x1 + x2) = velocityMassMatrix.currentVelocitySolution - 1/dt * pressureDivergenceMatrix.currentPressure = 
        // x1 + x2 = currentVelocitySolution - currentVelocitySolution^-1.(1/dt * pressureDivergenceMatrix.currentPressure)
        // Then we can solve only for currentVelocitySolution^-1.(1/dt * pressureDivergenceMatrix.currentPressure)
        velocityMassMatrix.rMult(currentVelocitySolution, velocityRhs);
        pressureDivergenceMatrix.rMultAdd(velocityRhs, currentPressureSolution, velocityRhs);

        // Solve for the u component
        solveStatus = SMM::ConjugateGradient(velocityMassMatrix, velocityRhs, currentVelocitySolution, -1, 1e-6);
        assert(solveStatus == SMM::SolverStatus::SUCCESS);

        // Solve for the v component
        solveStatus = SMM::ConjugateGradient(velocityMassMatrix, velocityRhs + nodesCount, currentVelocitySolution + nodesCount, -1, 1e-6);
        assert(solveStatus == SMM::SolverStatus::SUCCESS);

        // Prepare the velocity rhs vector for the next iteration
        velocityRhs.fill(0);
        // There is no need to fill the pressure rhs as the matrix vector product does not need it to be 0

        exportSolution(timeStep);;
    }
}

template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::imposeVelocityDirichlet(SMM::Vector& velocityVector) {
    const int nodesCount = grid.getNodesCount();
    const int velocityDirichletCount = grid.getVelocityDirichletSize();
    FemGrid2D::VelocityDirichletConstIt velocityDirichletBoundaries = grid.getVelocityDirichlet();
    for(int boundaryIndex = 0; boundaryIndex < velocityDirichletCount; ++boundaryIndex) {
        const FemGrid2D::VelocityDirichlet& boundary = velocityDirichletBoundaries[boundaryIndex];
        std::unordered_map<char, float> variables;
        for(int boundaryNodeIndex = 0; boundaryNodeIndex < boundary.getSize(); ++boundaryNodeIndex) {
            const int nodeIndex = boundary.getNodeIndexes()[boundaryNodeIndex];
            const real x = grid.getNodesBuffer()[nodeIndex * 2];
            const real y = grid.getNodesBuffer()[nodeIndex * 2 + 1];
            variables['x'] = x;
            variables['y'] = y;
            float uBoundary = 0, vBoundary = 0;
            boundary.eval(&variables, uBoundary, vBoundary);
            velocityVector[nodeIndex] = uBoundary;
            velocityVector[nodeIndex + nodesCount] = vBoundary;
        }
    }
}

template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::assemblVelocityMassMatrix() {
    // Compute the mass matrix. Local mass matrix is of the form Integral(dot(Transpose(PSI(xi, eta)), PSI(xi, eta)) * abs(|J|) dxi * deta). 
    // Where PSI(xi, eta) = {psi1(xi, eta), psi2(xi, eta), psi3(xi, eta), ...} is a row vector containing all shape functions and
    // |J| is the determinant of Jacobi matrix for the transformation to the unit triangle. Not that |J| is scalar which does not
    // depend of xi and eta, thus we can write the formula as |J| * integral(psi_i(xi, eta) * psi_j(xi, eta) * dxi * deta). So
    // there is no need to integrate the shape funcions for each element. We shall precompute the integral and then for each element
    // find |J| and multiply the precompute integral by it.

    const auto squareShape = [](const real xi, const real eta, StaticMatrix<real, VelocityShape::size, VelocityShape::size> & out) -> void {
        real p2Res[VelocityShape::size];
        VelocityShape::eval(xi, eta, p2Res);
        for(int i = 0; i < VelocityShape::size; ++i) {
            for(int j = 0; j < VelocityShape::size; ++j) {
                out[i][j] = p2Res[i]*p2Res[j];
            }
        }        
    };
    // This will hold the result of integral(psi_i(xi, eta) * psi_j(xi, eta) dxi * deta)
    StaticMatrix<real, VelocityShape::size, VelocityShape::size> shapeSquared;
    TriangleIntegrator::integrate(squareShape, shapeSquared);

    // Lambda wich takes advantage of precomputed shape function integral
    const auto localMass = [&](
        [[maybe_unused]]const int* elementIndexes,
        const real* elementNodes,
        StaticMatrix<real, VelocityShape::size, VelocityShape::size>& localMatrixOut
    ) -> void {
        const real jDetAbs = std::abs(linTriangleTmJacobian(elementNodes));
        for(int i = 0; i < VelocityShape::size; ++i) {
            for(int j = 0; j < VelocityShape::size; ++j) {
                localMatrixOut[i][j] = shapeSquared[i][j] * jDetAbs;
            }
        }
    };
    assembleMatrix<VelocityShape::size, VelocityShape::size>(localMass, velocityMassMatrix);
}

template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::assembleConvectionMatrix(SMM::CSRMatrix& convectionMatrix) {
    const int nodesCount = grid.getNodesCount();

    const auto localConvection = [&](
        const int* elementIndexes,
        const real* elementNodes,
        StaticMatrix<real, VelocityShape::size, VelocityShape::size>& localMatrixOut
    ) -> void {
        
        StaticMatrix<real, VelocityShape::size, 2> velocity;
        for(int i = 0; i < VelocityShape::size; ++i) {
            const int uIndex = elementIndexes[i];
            const int vIndex = elementIndexes[i] + nodesCount;
            velocity[i][0] = currentVelocitySolution[uIndex];
            velocity[i][1] = currentVelocitySolution[vIndex];
        }

        real J;
        StaticMatrix<real, 2, 2> B;
        differentialOperator(elementNodes, J, B);
        const real sign = J > 0 ? real(1) : real(-1);
        // The local convection matrix is of the form Integrate(Transpose(PSI(xi, eta)).PSI(xi, eta).UV.B.DPSI(xi, eta) * dxi * deta)
        // This functor is the function which is being integrated, it's later passed to integrate over triangle to get the localMatrix
        const auto convectionIntegrant = [&](
            const real xi,
            const real eta,
            StaticMatrix<real, VelocityShape::size, VelocityShape::size>& outIntegrated
        ) -> void {
            // TODO: p2Shape and delP2Shape can be cached for various xi and eta used by the integrator
            StaticMatrix<real, 1, VelocityShape::size> psi;
            VelocityShape::eval(xi, eta, psi);

            StaticMatrix<real, 2, VelocityShape::size> delPsi;
            VelocityShape::del(xi, eta, delPsi);
            
            outIntegrated = (psi.getTransposed() * psi * velocity * B * delPsi) * sign;
        };
        std::fill(localMatrixOut.begin(), localMatrixOut.end(), real(0));
        TriangleIntegrator::integrate(convectionIntegrant, localMatrixOut);
    };

    assembleMatrix<VelocityShape::size , VelocityShape::size>(localConvection, convectionMatrix);
}

template<typename VelocityShape, typename PressureShape>
template<typename RegularShape, typename DelShape, bool sideBySide>
void NavierStokesAssembly<VelocityShape, PressureShape>::assembleDivergenceMatrix(SMM::CSRMatrix& out) {
    const int numNodes = grid.getNodesCount();
    const int numElements = grid.getElementsCount();
    const int elementSize = std::max(RegularShape::size, DelShape::size);

    int elementIndexes[elementSize];
    real elementNodes[2 * elementSize];
    // IMPORTANT: The current implementation works only for P2-P1 shape function combination
    const int rows = sideBySide ? grid.getPressureNodesCount() : numNodes * 2;
    const int cols = sideBySide ? numNodes * 2 : grid.getPressureNodesCount();
    SMM::TripletMatrix triplet(rows, cols);
    StaticMatrix<real, RegularShape::size, DelShape::size> divLocalX;
    StaticMatrix<real, RegularShape::size, DelShape::size> divLocalY;
    LocalDivergenceFunctor<RegularShape, DelShape> localFunctor;
    for(int i = 0; i < numElements; ++i) {
        grid.getElement(i, elementIndexes, elementNodes);
        localFunctor(elementIndexes, elementNodes, divLocalX, divLocalY);

        // Put local matrices into the global matrix
        for(int localRow = 0; localRow < RegularShape::size; ++localRow) {
            const int globalRow = elementIndexes[localRow];
            for(int localCol = 0; localCol < DelShape::size; ++localCol) {
                const int globalCol = elementIndexes[localCol];
                if constexpr (sideBySide) {
                    triplet.addEntry(globalRow, globalCol, divLocalX.element(localRow, localCol));
                    triplet.addEntry(globalRow, globalCol + numNodes, divLocalY.element(localRow, localCol));
                } else {
                    triplet.addEntry(globalRow, globalCol, divLocalX.element(localRow, localCol));
                    triplet.addEntry(globalRow + numNodes, globalCol, divLocalY.element(localRow, localCol));
                }
            }
        }
    }
    out.init(triplet);
}

}