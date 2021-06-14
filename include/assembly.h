#pragma once
// Make sparse_matrix_math library use double. Must be before the incude of sparse_matrix_math.h
#define SMM_DEBUG_DOUBLE
#include <sparse_matrix_math/sparse_matrix_math.h>
#include <unordered_set>
#include <grid.h>
#include "error_code.h"
#include "static_matrix.h"
#include "kd_tree.h"
#include <string>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

namespace NSFem {

/// Draw 2D vector plot of the velocity field and save it as an image in the specified path
/// @param[in] grid The grid where the simulation was done
/// @param[in] uVec The velocity in u direction. Ordered the same way as the nodes in the grid are.
/// @param[in] vVec The velocity in v direction. Ordered the same way as the nodes in the grid are.
/// @param[in] pressure The pressure which will be plotted. Ordered the same way as the nodes in the grid are.
/// @param[in] path Path where the image must be saved. (Will not create missing folders on the path)
/// @param[in] width The width of the resulting image in pixels
/// @param[in] height The height of the resulting image in pixels
void drawVectorPlot(
    cv::Mat& outputImage,
    const FemGrid2D& grid,
    const SMM::real* const uVec,
    const SMM::real* const vVec,
    const SMM::real* const pressure,
    const std::string& path,
    const int width,
    const int height
);


/// Find all nodes which are part a boundary. The boundary has imposed condition for one channel (velocity, pressure, etc.)
/// @tparam IteratorT The type of the iterator which iterates over all boundaries for the given channel
/// @param[in] begin (Forward) Iterator to the first boundary
/// @param[in] end Iterator to the ending of the boundary. (One element past the last element. It will not be dereferenced)
/// @param[out] allBoundaryNodesOut The indices of all boundary nodes in all boundaries which [begin;end) spans.
template<typename IteratorT>
void collectBoundaryNodes(
    IteratorT begin,
    IteratorT end,
    std::unordered_set<int>& allBoundaryNodesOut
) {
    while(begin != end) {
        const int* boundary = begin->getNodeIndexes();
        const int boundarySize = begin->getSize();
        allBoundaryNodesOut.insert(boundary, boundary + boundarySize);
        ++begin;
    }
}

using real = double;

/// Find the determinant of the Jacobi matrix for a linear transformation of random triangle to the unit one
/// @param[in] elementNodes List of (x, y) coordinates in world space of the points which are going to be transformed.
/// The order is (0, 0), (1, 1), (0, 1)
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
    NavierStokesAssembly();
    EC::ErrorCode init(const char* simDescriptionPath);
    void solve();
    void semiLagrangianSolve();
    void setTimeStep(const real dt);
    void setOutputDir(std::string outputDir);
    void setOutputDir(std::string&& outputDir);
private:
    /// Unstructured triangluar grid where the fulid simulation will be computed
    FemGrid2D grid;

    /// KDTree which is used semi-Lagrangian solver is used.
    TriangleKDTree kdTree;

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

    /// The total time over the simulation is going to be executed
    real totalTime;

    /// The width of the output image in pixels
    int outputImageWidth;

    /// The height of the output image in pixels
    int outputImageHeight;

    /// OpenCV matrix which will is used when exporting image results. It will be filled with
    /// the corresponding colors and then written to an image file on the hard disk inside the output folder.
    cv::Mat outputImage;

    template<int localRows, int localCols, typename TLocalF, typename Triplet>
    void assembleMatrix(const TLocalF& localFunction, Triplet& triplet);

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

    /// Handles assembling of the convection matrix. It does it directly by the formula and does not use
    /// precomputed integrals. In theory it's possible, but it would make the code too complicated as it
    /// would require combined integral of 3 basis functions (i,j,k forall i,j,k). The convection matrix
    /// depends on the solution at the current time step. It changes at each time step.
    /// @param[out] outConvectionMatrix The resulting convection matrix
    template<typename Triplet>
    void assembleConvectionMatrix(Triplet& triplet);

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

    /// Use the semi-Lagrangian method to find the approximate value for the advectable quantities (velocity)
    /// The semi-Lagrangian method approximates directly the material derivative D()/Dt=0. For this method we are
    /// thinking in Lagrangian perspective. We imagine that the fluid as particles and at each grid point we measure
    /// the velocity of some particle, so to find the velocity at some point in space (x, y) at time step i+1, would
    /// require us to find a particle which at time point i was at some position (x1, y1), but has moved to (x, y) in
    /// time step i+1. Since each particle has a velocity and it "moves with the particle", the velocity of the particle
    /// which has moved to (x, y) at i+1 will be the velocity which we want to find.
    void advect(
        const real* const uVelocity,
        const real* const vVelocity,
        real* const uVelocityOut,
        real* const vVelocityOut
    );
    
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

    /// Handles assembling of the velocity mass matrix. It precomputes the integrals of each pair shape functions
    template<typename Shape>
    struct LocalMassFunctor {
        LocalMassFunctor() {
            const auto squareShape = [](
                const real xi,
                const real eta,
                StaticMatrix<real, VelocityShape::size, VelocityShape::size>& out
            ) -> void {
                real p2Res[VelocityShape::size];
                VelocityShape::eval(xi, eta, p2Res);
                for(int i = 0; i < VelocityShape::size; ++i) {
                    for(int j = 0; j < VelocityShape::size; ++j) {
                        out[i][j] = p2Res[i]*p2Res[j];
                    }
                }
            };
            TriangleIntegrator::integrate(squareShape, shapeSquared);
        }

        void operator()(
            [[maybe_unused]]const int* elementIndexes,
            const real* elementNodes,
            StaticMatrix<real, Shape::size, Shape::size>& localMatrixOut
        ) const {
            const real jDetAbs = std::abs(linTriangleTmJacobian(elementNodes));
            for(int i = 0; i < VelocityShape::size; ++i) {
                for(int j = 0; j < VelocityShape::size; ++j) {
                    localMatrixOut[i][j] = shapeSquared[i][j] * jDetAbs;
                }
            }
        }
    private:
        // Local mass matrix is of the form Integral(dot(Transpose(PSI(xi, eta)), PSI(xi, eta)) * abs(|J|) dxi * deta). 
        // Where PSI(xi, eta) = {psi1(xi, eta), psi2(xi, eta), psi3(xi, eta), ...} is a row vector containing all shape functions and
        // |J| is the determinant of Jacobi matrix for the transformation to the unit triangle. Not that |J| is scalar which does not
        // depend of xi and eta, thus we can write the formula as |J| * integral(psi_i(xi, eta) * psi_j(xi, eta) * dxi * deta). So
        // there is no need to integrate the shape funcions for each element. We shall precompute the integral and then for each element
        // find |J| and multiply the precompute integral by it.
        StaticMatrix<real, Shape::size, Shape::size> shapeSquared;
    };
};

template<typename VelocityShape, typename PressureShape>
EC::ErrorCode NavierStokesAssembly<VelocityShape, PressureShape>::init(const char* simDescriptionPath) {
    std::fstream simDescriptionFile(simDescriptionPath);
    if(!simDescriptionFile.is_open()) {
        return EC::ErrorCode(1, "Could not find desciption file");
    }
    nlohmann::basic_json simJson;
    simDescriptionFile >> simJson;
    if(!simJson["mesh_path"].is_string() ||
        grid.loadJSON(simJson["mesh_path"].get_ptr<std::string*>()->c_str())
    ) {
        return EC::ErrorCode(2, "Could not load mesh");
    }
    if(simJson.contains("dt") && simJson["dt"].is_number()) {
        dt = simJson["dt"];
    } else {
        return EC::ErrorCode(3, "Missing required param: dt");
    }
    if(simJson.contains("viscosity") && simJson["viscosity"].is_number()) {
        viscosity = simJson["viscosity"];
    } else {
        return EC::ErrorCode(3, "Missing required param: viscosity");
    }
    if(simJson.contains("total_time") && simJson["total_time"].is_number()) {
        totalTime = simJson["total_time"];
    } else {
        return EC::ErrorCode(3, "Missing required param: total_time");
    }
    if(simJson.contains("output_folder") && simJson["output_folder"].is_string()) {
        outFolder = simJson["output_folder"];
    }

    kdTree.init(&grid);
    return EC::ErrorCode();

}

template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::setTimeStep(const real dt) {
    this->dt = dt;
}

template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::setOutputDir(std::string outputDir) {
    this->outFolder = std::move(outputDir);
}

template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::setOutputDir(std::string&& outputDir) {
    this->outFolder = std::move(outputDir);
}


template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::exportSolution(const int timeStep) {
    if(outFolder.empty()) {
        return;
    }
    const int nodesCount = grid.getNodesCount();
    /*nlohmann::json outJSON = {
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
    }*/

    const std::string& velocityFieldPath = outFolder + "/velocity_field_" + std::to_string(timeStep) + ".bmp";
    drawVectorPlot(
        outputImage,
        grid,
        currentVelocitySolution,
        currentVelocitySolution + grid.getNodesCount(),
        currentPressureSolution,
        velocityFieldPath.c_str(),
        outputImageWidth,
        outputImageHeight
    );
    
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
template<int localRows, int localCols, typename TLocalF, typename Triplet>
void NavierStokesAssembly<VelocityShape, PressureShape>::assembleMatrix(const TLocalF& localFunction, Triplet& triplet) {
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
            for(int localCol = 0; localCol < localCols; ++localCol) {
                const int globalCol = elementIndexes[localCol];
                triplet.addEntry(globalRow, globalCol, localMatrix[localRow][localCol]);
            }
        }
    }
}

template<typename VelocityShape, typename PressureShape>
NavierStokesAssembly<VelocityShape, PressureShape>::NavierStokesAssembly() :
    outputImageWidth(1366),
    outputImageHeight(768),
    outputImage(outputImageHeight, outputImageWidth, CV_8UC3, cv::Scalar(255, 255, 255))
{
    if(this->outFolder.empty()) {
        printf("[Warning] No output path is added. Graphics and solutions will not be saved.\n");
    }
    
}

template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::solve() {
    
    SMM::CSRMatrix convectionMatrix; 

    SMM::TripletMatrix triplet(grid.getNodesCount(), grid.getNodesCount());

    LocalMassFunctor<VelocityShape> localVelocityMass;
    triplet.init(grid.getNodesCount(), grid.getNodesCount(), -1);
    assembleMatrix<VelocityShape::size, VelocityShape::size>(localVelocityMass, triplet);
    velocityMassMatrix.init(triplet);
    triplet.deinit();

    // Assemble global velocity stiffness matrix
    LocalStiffnessFunctor<VelocityShape> localVelocityStiffness;
    triplet.init(grid.getNodesCount(), grid.getNodesCount(), -1);
    assembleMatrix<VelocityShape::size, VelocityShape::size>(localVelocityStiffness, triplet);
    velocityStiffnessMatrix.init(triplet);
    triplet.deinit();

    // The key is the node index, the value is which boundary it belongs to
    // This is used during the assembling to check if a node belongs to a boundary
    // Since a node can belong to one and only one boundary we have vector of unordered
    // triplet matrices which will hold the weights
    std::unordered_set<int> allBoundaryNodes;
    collectBoundaryNodes(grid.getPressureDirichletBegin(), grid.getPressureDirichletEnd(), allBoundaryNodes);   
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
    SMM::Vector tmp(nodesCount);

    std::unordered_map<char, float> pressureVars;

    exportSolution(0);

    SMM::CSRMatrix::IC0Preconditioner velocityMassIC0(velocityMassMatrix);
    {
        [[maybe_unused]]const int preconditionError = velocityMassIC0.init();
        assert(preconditionError == 0 && "Failed to precondition the velocity mass matrix. It should be SPD");
    }
    SMM::CSRMatrix::IC0Preconditioner pressureStiffnessIC0(pressureStiffnessMatrix);
    {
        [[maybe_unused]]const int preconditionError = pressureStiffnessIC0.init();
        assert(preconditionError == 0 && "Failed to precondition the pressure stiffness matrix. It should be SPD");
    }

    const real eps = 1e-8;

    triplet.init(grid.getNodesCount(), grid.getNodesCount(), -1);
    assembleConvectionMatrix(triplet);
    convectionMatrix.init(triplet);

    for(int timeStep = 1; timeStep < steps; ++timeStep) {

        // Find the tentative velocity. The system is:
        // velocityMassMatrix.tentative = 
        //      = velocityMassMatrix.currentVelocitySolution 
        //       -dt * (convectionMatrix + viscosity * velocityStiffness).currentVelocitySolution
        // Note that on the right hand side the currentVelocitySolution is multiplied by the velocity mass matrix. So we can rewrite
        // the linear system as follows:
        // velocityMassMatrix.y = -dt * (convectionMatrix + viscosity * velocityStiffness).currentVelocitySolutions
        // y = tentative - currentVelocitySolution
        // tentative = y + currentVelocitySolution
        // Doing this seems to improve the stability of the method

        // Assemble tentative velocity system rhs
        assert(convectionMatrix.hasSameNonZeroPattern(velocityMassMatrix) && convectionMatrix.hasSameNonZeroPattern(velocityStiffnessMatrix));
        SMM::CSRMatrix::ConstIterator convectionIt = convectionMatrix.begin();
        SMM::CSRMatrix::ConstIterator velStiffnessIt = velocityStiffnessMatrix.begin();
        for(;convectionIt != convectionMatrix.end(); ++convectionIt, ++velStiffnessIt) {
            const int row = convectionIt->getRow();
            const int col = convectionIt->getCol();
            const real uVal = currentVelocitySolution[col];
            const real vVal = currentVelocitySolution[col + nodesCount];
            const real matrixPart = -dt * (convectionIt->getValue() + velStiffnessIt->getValue());
            velocityRhs[row] += matrixPart * uVal;
            velocityRhs[row + nodesCount] += matrixPart * vVal;
        }

        SMM::SolverStatus solveStatus = SMM::SolverStatus::SUCCESS;

        // Solve for the u component.
        solveStatus = SMM::ConjugateGradient(
            velocityMassMatrix,
            velocityRhs,
            currentVelocitySolution,
            tmp,
            -1,
            eps,
            velocityMassIC0
        );
        assert(solveStatus == SMM::SolverStatus::SUCCESS);
        for(int i = 0; i < nodesCount; ++i) {
            currentVelocitySolution[i] += tmp[i];
        }

        // Solve for the v component
        solveStatus = SMM::ConjugateGradient(
            velocityMassMatrix,
            velocityRhs + nodesCount,
            currentVelocitySolution + nodesCount,
            tmp,
            -1,
            eps,
            velocityMassIC0
        );
        assert(solveStatus == SMM::SolverStatus::SUCCESS);
        for(int i = 0; i < nodesCount; ++i) {
            currentVelocitySolution[i + nodesCount] += tmp[i];
        }

        imposeVelocityDirichlet(currentVelocitySolution);
       
        // Solve for the pressure. As pressure is "implicitly" stepped Dirchlet boundary conditions cannot be imposed after
        // solving the linear system. For this reason the pressure stiffness matrix was tweaked before time iterations begin.
        // Now at each time step the right hand side must be tweaked as well.

        // Find the right hand side
        velocityDivergenceMatrix.rMult(currentVelocitySolution, pressureRhs);

        // Now impose the Dirichlet Boundary Conditions
        FemGrid2D::PressureDirichletConstIt pressureDrichiletIt = grid.getPressureDirichletBegin();
        const FemGrid2D::PressureDirichletConstIt pressureDrichiletEnd = grid.getPressureDirichletEnd();
        for(; pressureDrichiletIt != pressureDrichiletEnd; ++pressureDrichiletIt) {
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
        solveStatus = SMM::ConjugateGradient(
            pressureStiffnessMatrix,
            pressureRhs,
            currentPressureSolution,
            currentPressureSolution,
            -1,
            eps,
            pressureStiffnessIC0
        );
        assert(solveStatus == SMM::SolverStatus::SUCCESS);



        // Combine the tentative velocity and the pressure to find the actual velocity. The system is:
        // velocityMassMatrix.currentVelocitySolution = velocityMassMatrix.tentative - dt * pressureDivergenceMatrix.currentPressureSolution
        // Note that on the right hand side the tentative velocity is multiplied by the velocity mass matrix.
        // velocityMassMatrix.y = - dt * pressureDivergenceMatrix.currentPressureSolution
        // y = currentVelocitySolution - tentative
        // currentVelocitySolution = y + tentative
        pressureDivergenceMatrix.rMult(currentPressureSolution, velocityRhs);

        // Solve for the u component
        solveStatus = SMM::ConjugateGradient(
            velocityMassMatrix,
            velocityRhs,
            currentVelocitySolution,
            tmp,
            -1,
            eps,
            velocityMassIC0
        );
        assert(solveStatus == SMM::SolverStatus::SUCCESS);
        for(int i = 0; i < nodesCount; ++i) {
            currentVelocitySolution[i] += tmp[i];
        }

        // Solve for the v component
        solveStatus = SMM::ConjugateGradient(
            velocityMassMatrix,
            velocityRhs + nodesCount,
            currentVelocitySolution + nodesCount,
            tmp,
            -1,
            eps,
            velocityMassIC0
        );
        assert(solveStatus == SMM::SolverStatus::SUCCESS);
        for(int i = 0; i < nodesCount; ++i) {
            currentVelocitySolution[i + nodesCount] += tmp[i];
        }

        // This prepares the convection matrix for the next step
        // Convection is the convection matrix formed by (dot(u_h, del(fi_i)), fi_j) : forall i, j in 0...numVelocityNodes - 1
        // Where fi_i is the i-th velocity basis function and viscosity is the fluid viscosity. This matrix is the same for the u and v
        // components of the velocity, thus we will assemble it only once and use the same matrix to compute all velocity components.
        // Used to compute the tentative velocity at step i + 1/2. The matrix depends on the current solution for the velocity, thus it
        // changes over time and must be reevaluated at each step.
        // TODO: Do not allocate space on each iteration, but reuse the matrix sparse structure
        convectionMatrix.zeroValues();
        assembleConvectionMatrix(convectionMatrix);

        // Prepare the velocity rhs vector for the next iteration
        velocityRhs.fill(0);
        // There is no need to fill the pressure rhs as the matrix vector product does not need it to be 0

        exportSolution(timeStep);
    }
}

#define IMPLICIT_DIFFUSION

template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::semiLagrangianSolve() {
    
    printf("Begin solution using semi Lagrangian method\n");
    printf("Elements: %d\n", grid.getElementsCount());
    printf("Velocity nodes: %d\n", grid.getNodesCount());
    printf("Pressure nodes: %d\n", grid.getPressureNodesCount());
    printf("Total time: %f\n", totalTime);
    printf("dt: %f\n", dt);
    if(outFolder.empty()) {
        printf("[Warning] No output folder. Results of the solver will not be written to disk\n");
    } else {
        printf("Output folder: %s\n", outFolder.c_str());
    }

    SMM::TripletMatrix triplet;
    SMM::TripletMatrix dirichletWeightsTriplet;
    std::unordered_set<int> allBoundaryNodes;
    
    
    // ======================================================================
    // ====================== ASSEMBLE VELOCITY MASS ========================
    // ======================================================================

    triplet.init(grid.getNodesCount(), grid.getNodesCount(), -1);
    LocalMassFunctor<VelocityShape> localVelocityMass;
    assembleMatrix<VelocityShape::size, VelocityShape::size>(localVelocityMass, triplet);
    velocityMassMatrix.init(triplet);
    triplet.deinit();

    // =====================================================================
    // =================== ASSEMBLE DIFFUSION MATRIX =======================
    // =====================================================================

    LocalStiffnessFunctor<VelocityShape> localVelocityStiffness;
#ifdef IMPLICIT_DIFFUSION
    const auto diffusionMatrixLocal = [&localVelocityMass, &localVelocityStiffness, this](
        [[maybe_unused]]const int* elementIndexes,
        const real* elementNodes,
        StaticMatrix<real, VelocityShape::size, VelocityShape::size>& localMatrixOut
    ) {
        localVelocityMass(elementIndexes, elementNodes, localMatrixOut);

        StaticMatrix<real, VelocityShape::size, VelocityShape::size> stiffness;
        localVelocityStiffness(elementIndexes, elementNodes, stiffness);

        localMatrixOut += stiffness * dt * viscosity;
    };

    collectBoundaryNodes(grid.getVelocityDirichletBegin(), grid.getVelocityDirichletEnd(), allBoundaryNodes);
    triplet.init(grid.getNodesCount(), grid.getNodesCount(), -1);
    dirichletWeightsTriplet.init(grid.getNodesCount(), grid.getNodesCount(), -1);
    SMM::CSRMatrix diffusionMatrix;
    assembleBCMatrix<VelocityShape::size, VelocityShape::size>(
        diffusionMatrixLocal,
        allBoundaryNodes,
        triplet,
        dirichletWeightsTriplet
    );
    diffusionMatrix.init(triplet);

    SMM::CSRMatrix velocityDirichletWeights;
    velocityDirichletWeights.init(dirichletWeightsTriplet);

    triplet.deinit();
    dirichletWeightsTriplet.deinit();
    allBoundaryNodes.clear();

    SMM::CSRMatrix::IC0Preconditioner diffusionIC0(diffusionMatrix);
    {
        [[maybe_unused]]const int preconditionError = diffusionIC0.init();
        assert(preconditionError == 0 && "Failed to precondition the pressure stiffness matrix. It should be SPD");
    }
#else
    triplet.init(grid.getNodesCount(), grid.getNodesCount(), -1);
    assembleMatrix<VelocityShape::size, VelocityShape::size>(localVelocityStiffness, triplet);
    velocityStiffnessMatrix.init(triplet);
    triplet.deinit();
    velocityStiffnessMatrix *= -dt * viscosity;
#endif
    // ======================================================================
    // ==================== ASSEMBLE PRESSURE STIFFNESS =====================
    // ======================================================================

    collectBoundaryNodes(grid.getPressureDirichletBegin(), grid.getPressureDirichletEnd(), allBoundaryNodes);  
    triplet.init(grid.getPressureNodesCount(), grid.getPressureNodesCount(), -1);
    dirichletWeightsTriplet.init(grid.getPressureNodesCount(), grid.getPressureNodesCount(), -1);
    LocalStiffnessFunctor<PressureShape> localPressureStuffness;
    assembleBCMatrix<PressureShape::size, PressureShape::size>(
        localPressureStuffness,
        allBoundaryNodes,
        triplet,
        dirichletWeightsTriplet
    );
    pressureStiffnessMatrix.init(triplet);

    SMM::CSRMatrix pressureDirichletWeights;
    pressureDirichletWeights.init(dirichletWeightsTriplet);

    triplet.deinit();
    dirichletWeightsTriplet.deinit();
    allBoundaryNodes.clear();

    // ======================================================================
    // ==================== ASSEMBLE DIVERGENCE MATRICES ====================
    // ======================================================================

    assembleDivergenceMatrix<PressureShape, VelocityShape, true>(velocityDivergenceMatrix);
    assembleDivergenceMatrix<VelocityShape, PressureShape, false>(pressureDivergenceMatrix);

    // ====================================================================

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
    SMM::Vector tmp(nodesCount * 2, 0);

    std::unordered_map<char, float> pressureVars;
    std::unordered_map<char, float> velocityVars;

    exportSolution(0);

    SMM::CSRMatrix::IC0Preconditioner velocityMassIC0(velocityMassMatrix);
    {
        [[maybe_unused]]const int preconditionError = velocityMassIC0.init();
        assert(preconditionError == 0 && "Failed to precondition the velocity mass matrix. It should be SPD");
    }

    SMM::CSRMatrix::IC0Preconditioner pressureStiffnessIC0(pressureStiffnessMatrix);
    {
        [[maybe_unused]]const int preconditionError = pressureStiffnessIC0.init();
        assert(preconditionError == 0 && "Failed to precondition the pressure stiffness matrix. It should be SPD");
    }

    const real eps = 1e-8;

    for(int timeStep = 1; timeStep < steps; ++timeStep) {
        SMM::SolverStatus solveStatus = SMM::SolverStatus::SUCCESS;
// ==================================================================================
// =============================== ADVECTION ========================================
// ==================================================================================
        advect(
            currentVelocitySolution,
            currentVelocitySolution + nodesCount,
            tmp,
            tmp + nodesCount
        );        

// ==================================================================================
// ============================== PRESSURE SOLVE ====================================
// ==================================================================================

        // Solve for the pressure. As pressure is "implicitly" stepped Dirchlet boundary conditions cannot be imposed after
        // solving the linear system. For this reason the pressure stiffness matrix was tweaked before time iterations begin.
        // Now at each time step the right hand side must be tweaked as well.

        // Find the right hand side
        velocityDivergenceMatrix.rMult(tmp, pressureRhs);

        // Now impose the Dirichlet Boundary Conditions
        FemGrid2D::PressureDirichletConstIt pressureDrichiletIt = grid.getPressureDirichletBegin();
        const FemGrid2D::PressureDirichletConstIt pressureDrichiletEnd = grid.getPressureDirichletEnd();
        for(;pressureDrichiletIt != pressureDrichiletEnd; ++pressureDrichiletIt) {
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
        solveStatus = SMM::ConjugateGradient(
            pressureStiffnessMatrix,
            pressureRhs,
            currentPressureSolution,
            currentPressureSolution,
            -1,
            eps,
            pressureStiffnessIC0
        );
        assert(solveStatus == SMM::SolverStatus::SUCCESS);

        // Combine the tentative velocity and the pressure to find the actual velocity. The system is:
        // velocityMassMatrix.currentVelocitySolution = velocityMassMatrix.tentative - dt * pressureDivergenceMatrix.currentPressureSolution
        // Note that on the right hand side the tentative velocity is multiplied by the velocity mass matrix.
        // velocityMassMatrix.y = - dt * pressureDivergenceMatrix.currentPressureSolution
        // y = currentVelocitySolution - tentative
        // currentVelocitySolution = y + tentative
        pressureDivergenceMatrix.rMult(currentPressureSolution, velocityRhs);

        // Solve for the u component
        solveStatus = SMM::ConjugateGradient(
            velocityMassMatrix,
            velocityRhs,
            tmp,
            currentVelocitySolution,
            -1,
            eps,
            velocityMassIC0
        );
        assert(solveStatus == SMM::SolverStatus::SUCCESS);
        for(int i = 0; i < nodesCount; ++i) {
            currentVelocitySolution[i] += tmp[i];
        }

        // Solve for the v component
        solveStatus = SMM::ConjugateGradient(
            velocityMassMatrix,
            velocityRhs + nodesCount,
            tmp + nodesCount,
            currentVelocitySolution + nodesCount,
            -1,
            eps,
            velocityMassIC0
        );
        assert(solveStatus == SMM::SolverStatus::SUCCESS);
        for(int i = 0; i < nodesCount; ++i) {
            currentVelocitySolution[i + nodesCount] += tmp[i + nodesCount];
        }


// ==================================================================================
// ============================= DIFFUSION SOLVE ====================================
// ==================================================================================
#ifdef IMPLICIT_DIFFUSION
        velocityMassMatrix.rMult(currentVelocitySolution, velocityRhs);
        velocityMassMatrix.rMult(currentVelocitySolution + nodesCount, velocityRhs + nodesCount);

        // Now impose the Dirichlet Boundary Conditions
        FemGrid2D::VelocityDirichletConstIt velocityDrichiletIt = grid.getVelocityDirichletBegin();
        const FemGrid2D::VelocityDirichletConstIt velocityDrichiletEnd = grid.getVelocityDirichletEnd();
        for(;velocityDrichiletIt != velocityDrichiletEnd; ++velocityDrichiletIt) {
            const FemGrid2D::VelocityDirichlet& boundary = *velocityDrichiletIt;
            for(int boundaryNodeIndex = 0; boundaryNodeIndex < boundary.getSize(); ++boundaryNodeIndex) {
                const int nodeIndex = boundary.getNodeIndexes()[boundaryNodeIndex];
                const real x = grid.getNodesBuffer()[nodeIndex * 2];
                const real y = grid.getNodesBuffer()[nodeIndex * 2 + 1];
                velocityVars['x'] = x;
                velocityVars['y'] = y;
                float uBoundary = 0, vBoundary = 0;
                boundary.eval(&velocityVars, uBoundary, vBoundary);
                velocityRhs[nodeIndex] = uBoundary;
                velocityRhs[nodeIndex + nodesCount] = vBoundary;
                SMM::CSRMatrix::ConstRowIterator it = velocityDirichletWeights.rowBegin(nodeIndex);
                const SMM::CSRMatrix::ConstRowIterator end = velocityDirichletWeights.rowEnd(nodeIndex);
                while(it != end) {
                    velocityRhs[it->getCol()] -= it->getValue() * uBoundary;
                    velocityRhs[it->getCol() + nodesCount] -= it->getValue() * vBoundary;;
                    ++it;
                }
            }
        }

        solveStatus = SMM::ConjugateGradient(
            diffusionMatrix,
            velocityRhs,
            currentVelocitySolution,
            currentVelocitySolution,
            -1,
            eps,
            diffusionIC0
        );
        assert(solveStatus == SMM::SolverStatus::SUCCESS);

        solveStatus = SMM::ConjugateGradient(
            diffusionMatrix,
            velocityRhs + nodesCount,
            currentVelocitySolution + nodesCount,
            currentVelocitySolution + nodesCount,
            -1,
            eps,
            diffusionIC0
        );
        assert(solveStatus == SMM::SolverStatus::SUCCESS);
#else
        velocityStiffnessMatrix.rMult(currentVelocitySolution, velocityRhs);

        solveStatus = SMM::ConjugateGradient(
            velocityMassMatrix,
            velocityRhs,
            currentVelocitySolution,
            tmp,
            -1,
            eps,
            velocityMassIC0
        );
        assert(solveStatus == SMM::SolverStatus::SUCCESS);
        for(int i = 0; i < nodesCount; ++i) {
            currentVelocitySolution[i] += tmp[i];
        }

        velocityStiffnessMatrix.rMult(currentVelocitySolution + nodesCount, velocityRhs + nodesCount);
        solveStatus = SMM::ConjugateGradient(
            velocityMassMatrix,
            velocityRhs + nodesCount,
            currentVelocitySolution + nodesCount,
            tmp + nodesCount,
            -1,
            eps,
            velocityMassIC0
        );
        assert(solveStatus == SMM::SolverStatus::SUCCESS);
        for(int i = 0; i < nodesCount; ++i) {
            currentVelocitySolution[i + nodesCount] += tmp[i + nodesCount];
        }

        imposeVelocityDirichlet(currentVelocitySolution);
#endif
        exportSolution(timeStep);

    }
}

template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::imposeVelocityDirichlet(SMM::Vector& velocityVector) {
    const int nodesCount = grid.getNodesCount();
    const int velocityDirichletCount = grid.getVelocityDirichletSize();
    FemGrid2D::VelocityDirichletConstIt velocityDirichletBoundaries = grid.getVelocityDirichletBegin();
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
template<typename Triplet>
void NavierStokesAssembly<VelocityShape, PressureShape>::assembleConvectionMatrix(Triplet& triplet) {
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

    assembleMatrix<VelocityShape::size , VelocityShape::size>(localConvection, triplet);
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

template<typename VelocityShape, typename PressureShape>
void NavierStokesAssembly<VelocityShape, PressureShape>::advect(
    const real* const uVelocity,
    const real* const vVelocity,
    real* const uVelocityOut,
    real* const vVelocityOut
) {
    const int velocityNodesCount = grid.getNodesCount();
    const real* velocityNodes = grid.getNodesBuffer();
    const int elementsCount = grid.getElementsCount();

    static_assert(VelocityShape::size == 6, "Only P2-P1 elements are supported");
    assert(grid.getElementSize() == VelocityShape::size && "Only P2-P1 elements are supported");
    Point2D elementNodes[VelocityShape::size];
    int elementIndexes[VelocityShape::size];

    for(int i = 0; i < velocityNodesCount; ++i) {
        const Point2D position(velocityNodes[2*i], velocityNodes[2*i + 1]);
        const Point2D velocity(uVelocity[i], vVelocity[i]);
        const Point2D start = position - velocity * dt;

// #define BF_SEMI_LAGRANGIAN
#ifdef BF_SEMI_LAGRANGIAN
        real uResult = 0, vResult = 0;
        bool isPointInsideMesh = false;
        for(int j = 0; j < elementsCount; ++j) {
            grid.getElement(j, elementIndexes, reinterpret_cast<real*>(elementNodes));
            // Check if start lies inside an element. This uses the barrycentric coordinates to check if a point lies in a triangle.
            // Since we are in 2D we can describe point p as a linear combination of two vectors, we choose
            // them to be AB and AC. We want (0, 0) in this new space to be the point A
            // p = A + l1*AB + l2*AC
            // |px|  = |Ax + l1(Bx - Ax) + l2(Cx - Ax)|
            // |py|  = |Ay + l1(By - Ay) + l2(Cy - Ay)|
            //
            // |px - ax| = |Bx - Ax  Cx - Ax||l1|
            // |py - ay| = |By - Ay  Cy - Ay||l2|
            //
            // |l1| = 1/D * |Cy - Ay  Ax - Cx||px - ax|
            // |l2| = 1/D * |Ay - By  Bx - Ax||py - ay|
            const Point2D AB = elementNodes[1] - elementNodes[0];
            const Point2D AC = elementNodes[2] - elementNodes[0];
            const Point2D AP = start - elementNodes[0];
            const real D = (AB.x * AC.y) - (AB.y * AC.x);
            assert(D != 0);
            const int DSign = D > 0 ? 1 : -1;
            // We do not want to divide by the determinant, so when we solve the system we get the result
            // scaled by the determinant.
            const real scaledBarry1 =  (AP.x * AC.y - AP.y * AC.x) * DSign;
            if(scaledBarry1 < 0 || scaledBarry1 > D) {
                continue;
            }
            const real scaledBarry2 = (AP.y * AB.x - AP.x * AB.y) * DSign;
            if(scaledBarry2 < 0 || scaledBarry1 > D - scaledBarry2) {
                continue;
            }
            
            // When we reach this point we know that the point "start" is inside the triangle
            // We need to scale down the barrycentroc coords and we'll get the point inside the triangle
            const real dInv = 1 / std::abs(D);
            const real xi = scaledBarry1 * dInv;
            const real eta = scaledBarry2 * dInv;
            
            real interpolationCoefficients[VelocityShape::size];
            VelocityShape::eval(xi, eta, interpolationCoefficients);
            for(int k = 0; k < VelocityShape::size; ++k) {
                uResult += interpolationCoefficients[k] * uVelocity[elementIndexes[k]];
                vResult += interpolationCoefficients[k] * vVelocity[elementIndexes[k]];
            }
            isPointInsideMesh = true;
            break;
        }
        if(!isPointInsideMesh) {
            real minDistSq = std::numeric_limits<real>::infinity();
            for(int j = 0; j < velocityNodesCount; ++j) {
                const Point2D meshPoint(velocityNodes[2 * j], velocityNodes[2 * j + 1]);
                const real currentDistSq = start.distToSq(meshPoint);
                if(currentDistSq < minDistSq) {
                    minDistSq = currentDistSq;
                    uResult = uVelocity[j];
                    vResult = vVelocity[j];
                }
            }
        }
        uVelocityOut[i] = uResult;
        vVelocityOut[i] = vResult;
#else
        // If start is inside some element xi and eta will be the barrycentric coordinates
        // of start inside that element.
        real xi, eta;
        // If start does not lie in any triangle this will be the index of the nearest node to start
        int nearestNeighbour;
        const int element = kdTree.findElement(start, xi, eta, nearestNeighbour);
        if(element > -1) {
            // Start point lies in an element, interpolate it by using the shape functions.
            // This is possible because xi and eta do not change when the element is transformed
            // to the unit element where the shape functions are defined. 
            real uResult = 0, vResult = 0;
            grid.getElement(element, elementIndexes, reinterpret_cast<real*>(elementNodes));
            real interpolationCoefficients[VelocityShape::size];
            VelocityShape::eval(xi, eta, interpolationCoefficients);
            for(int k = 0; k < VelocityShape::size; ++k) {
                uResult += interpolationCoefficients[k] * uVelocity[elementIndexes[k]];
                vResult += interpolationCoefficients[k] * vVelocity[elementIndexes[k]];
            }
            uVelocityOut[i] = uResult;
            vVelocityOut[i] = vResult;
        } else {
            // Start point does not lie in any element (probably it's outside the mesh)
            // Use the closest point in the mesh to approximate the velocity
            uVelocityOut[i] = uVelocity[nearestNeighbour];
            vVelocityOut[i] = vVelocity[nearestNeighbour];
        }
    }
#endif
}

}