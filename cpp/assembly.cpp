#include <cassert>
#include <cstring>
#include "assembly.h"
#include "grid.h"
#include "static_matrix.h"

namespace NSFem {

template<typename TLocalF, int localRows, int localCols>
void NavierStokesAssembly::assembleMatrix(const TLocalF& localFunction, SMM::CSRMatrix& out) {
    const int numNodes = grid.getNodesCount();
    const int numElements = grid.getElementsCount();
    const int elementSize = std::max(localRows, localCols);
    int elementIndexes[elementSize];
    real elementNodes[2 * elementSize];
    real localMatrix[localRows][localCols];
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

/// Multiply two matrices represented as 2D arrays.
/// @tparam aRows Number of rows of the left hand side matrix.
/// @tparam aCols Number of columns of the left hand side matrix. The operation is valid
/// only if the right hand side matrix has the same number of rows.
/// @tparam bCols Number of columns of the right hand side matrix.
/// @param[in] a The left hand side matrix
/// @param[in] b The right hand side matrix
/// @param[out] out The result matrix. The matrix must be filled with zeroes when passed to the function.
template<int aRows, int aCols, int bCols>
inline constexpr void multiplyMatrix(const real (&a)[aRows][aCols], const real (&b)[aCols][bCols], real (&out)[aRows][bCols]) {
    for(int i = 0; i < aRows; ++i) {
        for(int j = 0; j < bCols; ++j) {
            for(int k = 0; k < aCols; ++k) {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

/// Multiply matrix by its transpose
/// @tparam rows Number of rows in the matrix
/// @tparam cols Number of columns in the matrix
/// @param[in] a The matrix which will be multiplied by its transpose
/// @param[out] out The resulting matrix. The matrix must be filled with zeroes when passed to the function.
template<int rows, int cols>
inline constexpr void multiplyByTranspose(const real (&a)[rows][cols], real (&out)[rows][rows]) {
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < rows; ++j) {
            for(int k = 0; k < cols; ++k) {
                out[i][j] += a[i][k] * a[j][k];
            }
        }
    }
}

/// Compute Transpose(A).A. First transpose A and the multiply the transpose by A, without explicitly computing the transpose
/// @tparam rows Number of rows in the matrix
/// @tparam cols Number of columns in the matrix
/// @param[in] a The matrix which will be multiplied by its transpose
/// @param[out] out The resulting matrix. The matrix must be filled with zeroes when passed to the function.
template<int rows, int cols>
inline constexpr void transposeMutiply(const real (&a)[rows][cols], real (&out)[cols][cols]) {
    for(int i = 0; i < cols; ++i) {
        for(int j = 0; j < cols; ++j) {
            for(int k = 0; k < rows; ++k) {
                out[i][j] += a[k][i] * a[k][j];
            }
        }
    }
}

/// Shape functions for 2D triangluar element with degrees of freedom in each triangle node
/// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
/// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
/// @param[out] out - The value of each shape function at (xi, eta). The array must have at least 6 elements.
///< Order: [0, 1, 2] - The values at the nodes of the triangle
inline void p1Shape(const real xi, const real eta, real out[3]) {
    out[0] = 1 - xi - eta;
    out[1] = xi;
    out[2] = eta;
}

/// Compute the gradient, at point (xi, eta) of the shape functions for triangular element with degrees of freedom in each triangle node
/// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
/// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
/// @param[out] out - This will hold the gradient. The first 3 elements are the derivatives of the shape functions with
///< respect to xi, next 3 elements are the derivatives of the shape functions with respect to eta
inline void delP1Shape([[maybe_unused]]const real xi, [[maybe_unused]]const real eta, real out[2][3]) {
    // dpsi/dxi
    out[0][0] = -1.0f;
    out[0][1] = 1.0f;
    out[0][2] = 0.0f;

    // dpsi/deta
    out[1][0] = -1.0f;
    out[1][1] = 0.0f;
    out[1][2] = 1.0f;
}

/// Shape functions for 2D triangluar element with degrees of freedom in each triangle node and in the middle of each side
/// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
/// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
/// @param[out] out - The value of each shape function at (xi, eta). The array must have at least 6 elements.
///< Order is as follows:
///< [0, 1, 2] - The values at the nodes of the triangle
///< 3 - the value at the point between 1 and 2
///< 4 - the value at the point between 0 and 2
///< 5 - the value at the point between 0 and 1
inline void p2Shape(const real xi, const real eta, real out[6]) {
    out[0] = 1 - 3 * xi - 3 * eta + 2 * xi * xi + 4 * xi * eta + 2 * eta * eta;
    out[1] = 2 * xi * xi - xi;
    out[2] = 2 * eta * eta - eta;
    out[3] = 4 * xi * eta;
    out[4] = 4 * eta - 4 * xi * eta - 4 * eta * eta;
    out[5] = 4 * xi - 4 * xi * xi - 4 * xi * eta;
}

/// Compute the gradient, at point (xi, eta) of the shape functions for triangular element with degrees of freedom
/// in each triangle node and in the middle of each side
/// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
/// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
/// @param[out] out - This will hold the gradient. The first 6 elements are the derivatives of the shape functions with
/// respect to xi, next 6 elements are the derivatives of the shape functions with respect to eta
inline void delP2Shape(const real xi, const real eta, real out[12]) {
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

/// Integrate vector function of two dimensional vector argument over an unit triangle
/// The unit triangle is right angle triangle with vertices in (0, 0), (0, 1) and (1, 0)
/// This numerical integration formula is exact for polynomials of degree <= 3
/// @tparam TFunctor Type of a functor parameter which takes has arguments:
/// xi, eta - coordinates in the unit triangle and out parameter which is pointer to array
/// of reals where the result of the functor at (xi, eta) will be returned
/// @tparam outSize The size of the otput parameter of TFunctor
/// @param[in] f Vector function of two arguments which will be integrated over the unit triangle.
/// @param[out] out Result from integrating f over the unit triangle (must of size outSize)
/// The out array must be contain only zeroes when passed to this function
template<int outSize, typename TFunctor>
void integrateOverTriangle(const TFunctor& f, real* out) {
    assert(std::all_of(out, out + outSize, [](const real x){return x == real(0);}));
    const int numIntegrationPoints = 8;
    const real weights[numIntegrationPoints] = {
        3.0 / 120.0, 3.0 / 120.0, 3.0 / 120.0,
        8.0 / 120.0, 8.0 / 120.0, 8.0 / 120.0,
        27.0 / 120.0
    };
    const real nodes[2 * numIntegrationPoints] = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        0.5, 0.0,
        0.5, 0.5,
        0.0, 0.5,
        1.0/3.0, 1.0/3.0
    };

    real tmp[outSize] = {};
    for(int i = 0; i < numIntegrationPoints; ++i) {
        const real x = nodes[2 * i];
        const real y = nodes[2 * i + 1];
        f(x, y, tmp);
        for(int j = 0; j < outSize; ++j) {
            out[j] += tmp[j] * weights[i];
        }
    }
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

inline constexpr int linearize2DIndex(const int numCols, const int row, const int col) {
    return row * numCols + col;
}

NavierStokesAssembly::NavierStokesAssembly(FemGrid2D&& grid, const real dt, const real viscosity) :
    grid(std::move(grid)),
    viscosity(viscosity),
    dt(dt)
{

}

void NavierStokesAssembly::solve(const float totalTime) {
    assembleConstantMatrices();
    velocityStiffnessMatrix *= viscosity;
    const int steps = totalTime / dt;
    const int nodesCount = grid.getNodesCount();
    currentVelocitySolution.init(nodesCount * 2, 0.0f);
    imposeVelocityDirichlet(currentVelocitySolution);
    SMM::Vector rhs(nodesCount * 2, 0.0f);
    for(int i = 0; i < steps; ++i) {
        // Convection is the convection matrix formed by (dot(u_h, del(fi_i)), fi_j) : forall i, j in 0...numVelocityNodes - 1
        // Where fi_i is the i-th velocity basis function and viscosity is the fluid viscosity. This matrix is the same for the u and v
        // components of the velocity, thus we will assemble it only once and use the same matrix to compute all velocity components.
        // Used to compute the tentative velocity at step i + 1/2. The matrix depends on the current solution for the velocity, thus it
        // changes over time and must be reevaluated at each step.
        // TODO: Do not allocate space on each iteration, but reuse the matrix sparse structure
        SMM::CSRMatrix convectionMatrix;
        assembleConvectionMatrix(convectionMatrix);
        assert(convectionMatrix.hasSameNonZeroPattern(velocityMassMatrix));
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
            rhs[row] += (massIt->getValue() - dt * (convectionIt->getValue() + velStiffnessIt->getValue())) * uVal;
            rhs[row + nodesCount] += (massIt->getValue() - dt * (convectionIt->getValue() + velStiffnessIt->getValue())) * vVal;
        }
        SMM::SolverStatus solveStatus = SMM::SolverStatus::SUCCESS;

        // Solve for the u component
        solveStatus = SMM::ConjugateGradient(velocityMassMatrix, rhs, currentVelocitySolution, -1, 1e-6);
        assert(solveStatus == SMM::SolverStatus::SUCCESS);

        // Solve for the v component
        solveStatus = SMM::ConjugateGradient(velocityMassMatrix, rhs + nodesCount, currentVelocitySolution + nodesCount, -1, 1e-6);
        assert(solveStatus == SMM::SolverStatus::SUCCESS);

        imposeVelocityDirichlet(currentVelocitySolution);

    }
}

void NavierStokesAssembly::imposeVelocityDirichlet(SMM::Vector& velocityVector) {
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


void NavierStokesAssembly::assembleConstantMatrices() {
    assemblVelocityMassMatrix();
    assembleVelocityStiffnessMatrix();
    assembleDivergenceMatrix();
}

void NavierStokesAssembly::assemblVelocityMassMatrix() {
    // Compute the mass matrix. Local mass matrix is of the form Integral(dot(Transpose(PSI(xi, eta)), PSI(xi, eta)) * abs(|J|) dxi * deta). 
    // Where PSI(xi, eta) = {psi1(xi, eta), psi2(xi, eta), psi3(xi, eta), ...} is a row vector containing all shape functions and
    // |J| is the determinant of Jacobi matrix for the transformation to the unit triangle. Not that |J| is scalar which does not
    // depend of xi and eta, thus we can write the formula as |J| * integral(psi_i(xi, eta) * psi_j(xi, eta) * dxi * deta). So
    // there is no need to integrate the shape funcions for each element. We shall precompute the integral and then for each element
    // find |J| and multiply the precompute integral by it.

    const int p2Size = 6;
    // This will hold the result of integral(psi_i(xi, eta) * psi_j(xi, eta) dxi * deta)
    real p2Squared[p2Size][p2Size] = {};
    const auto squareP2 = [](const real xi, const real eta, real* out) -> void {
        real p2Res[p2Size];
        p2Shape(xi, eta, p2Res);
        for(int i = 0; i < p2Size; ++i) {
            for(int j = 0; j < p2Size; ++j) {
                const int idx = i * 6 + j;
                out[idx] = p2Res[i]*p2Res[j];
            }
        }        
    };
    integrateOverTriangle<p2Size * p2Size>(squareP2, reinterpret_cast<real*>(p2Squared));

    // Lambda wich takes advantage of precomputed shape function integral
    const auto localMass = [&p2Squared]([[maybe_unused]]const int* elementIndexes, const real* elementNodes, real localMatrixOut[p2Size][p2Size]) -> void {
        const real jDetAbs = std::abs(linTriangleTmJacobian(elementNodes));
        for(int i = 0; i < p2Size; ++i) {
            for(int j = 0; j < p2Size; ++j) {
                localMatrixOut[i][j] = p2Squared[i][j] * jDetAbs;
            }
        }
    };
    assembleMatrix<decltype(localMass), p2Size, p2Size>(localMass, velocityMassMatrix);
}

void NavierStokesAssembly::assembleVelocityStiffnessMatrix() {
    // Compute the mass matrix. Local stiffness matrix is of the form Integral(Transpose(B.DPSI(xi, eta)/|J|).B.DPSI(xi, eta)/|J| * abs(|J|) dxi, deta),
    // |J| cancel out to produce more readable result 1/abs(|J|) * Integral(Transpose(B.PSI(xi, eta)).B.PSI(xi, eta) * dxi, deta)
    // Where DPSI(xi, eta) = {dpsi_1(xi, eta)/dxi, ..., dpsi_n(xi, eta)/dxi, dpsi_1(xi, eta)/deta ... dpsi_n(xi, eta)/deta} is a
    // row vector containing the gradient of each shape function, first are the derivatives in the xi direction then are the derivatives
    // in the eta direction |J| is the determinant of the linear transformation to the unit triangle and B/|J| is a 2x2 matrix which
    // represents the Grad operator in terms of xi and eta. As with mass matrix note, that B and |J| do not depend on xi and eta, only
    // the gradient of the shape functions dependon xi and eta. Thus we can precompute all pairs of shape function integrals and reuse
    // them in each element. The main complexity comes from the matrix B, which multiples the shape functions.

    const int p2Size = 6;
    const int delP2Size = p2Size * 2;
    // Compute the integral of each pair shape function derivatives.
    const auto squareDelP = [](const real xi, const real eta, real* out) -> void{
        real delP2[delP2Size] = {};
        delP2Shape(xi, eta, delP2);
        for(int i = 0; i < delP2Size; ++i) {
            for(int j = 0; j < delP2Size; ++j) {
                const int outIndex = linearize2DIndex(delP2Size, i, j);
                out[outIndex] = delP2[i] * delP2[j];
            }
        }
    };
    real delPSq[delP2Size][delP2Size] = {};
    integrateOverTriangle<delP2Size * delP2Size>(squareDelP, reinterpret_cast<real*>(delPSq));

    const auto localStiffness = [&delPSq]([[maybe_unused]]const int* elementIndexes, const real* elementNodes, real localMatrixOut[p2Size][p2Size]) -> void {
        real b[2][2];
        real J;
        differentialOperator(elementNodes, J, b);
        J = real(1.0) / std::abs(J);
        for(int i = 0; i < p2Size; ++i) {
            for(int j = 0; j < p2Size; ++j) {
                localMatrixOut[i][j] = real(0);
                const real sq[4] = {
                    delPSq[i][j],
                    delPSq[i][p2Size + j],
                    delPSq[p2Size + i][j],
                    delPSq[p2Size + i][p2Size + j]
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

    assembleMatrix<decltype(localStiffness), p2Size, p2Size>(localStiffness, velocityStiffnessMatrix);
}

void NavierStokesAssembly::assembleConvectionMatrix(SMM::CSRMatrix& convectionMatrix) {
    const int p2Size = 6;
    const int nodesCount = grid.getNodesCount();
    const auto localConvection = [&](const int* elementIndexes, const real* elementNodes, real localMatrixOut[p2Size][p2Size]) -> void {
        StaticMatrix<real, p2Size, 2> velocity;
        for(int i = 0; i < p2Size; ++i) {
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
        const auto convectionIntegrant = [&](const real xi, const real eta, real* outIntegrated) -> void {
            // TODO: p2Shape and delP2Shape can be cached for various xi and eta used by the integrator
            const int p2Size = 6;
            StaticMatrix<real, 1, p2Size> psi;
            p2Shape(xi, eta, psi.data());

            StaticMatrix<real, 2, p2Size> delPsi;
            delP2Shape(xi, eta, delPsi.data());
            
            StaticMatrix<real, p2Size, p2Size> result = (psi.getTransposed() * psi * velocity * B * delPsi) * sign;
            memcpy(outIntegrated, result.data(), sizeof(real) * p2Size * p2Size);
        };
        std::fill_n(reinterpret_cast<real*>(localMatrixOut), p2Size * p2Size, real(0));
        integrateOverTriangle<p2Size * p2Size>(convectionIntegrant, reinterpret_cast<real*>(localMatrixOut));
    };

    assembleMatrix<decltype(localConvection), p2Size, p2Size>(localConvection, convectionMatrix);
}

void NavierStokesAssembly::assembleDivergenceMatrix() {
    const int p1Size = 3;
    const int p2Size = 6;
    const int numNodes = grid.getNodesCount();
    const int numElements = grid.getElementsCount();
    const int elementSize = std::max(p1Size, p2Size);

    // Matrix to hold combined values for the integrals: Integrate(psi_i(xi, eta) * dpsi_j(xi, eta)/dxi * dxi * deta) and
    // Integrate(psi(xi, eta) * dpsi(xi, eta)/deta * dxi * deta). Where i is in [0;p1Size-1] and j is in [0;p2Size-1]
    // The first p2Size entries in each row are the first integral and the second represent the second integral 
    StaticMatrix<real, p1Size, 2 * p2Size> p1DelP2Combined;
    auto combineP1P2 = [&](const real xi, const real eta, real* out) -> void {
        real p1Res[p1Size] = {};
        p1Shape(xi, eta, p1Res);

        real delP2Res[2][p2Size] = {};
        delP2Shape(xi, eta, reinterpret_cast<real*>(delP2Res));

        for(int i = 0; i < p1Size; ++i) {
            for(int j = 0; j < p2Size; ++j) {
                for(int k = 0; k < 2; ++k) {
                    const int idx = linearize2DIndex(2 * p2Size, i, j + k * p2Size);
                    out[idx] = p1Res[i] * delP2Res[k][j];
                }
            }
        }
    };
    integrateOverTriangle<p1Size * 2 * p2Size>(combineP1P2, p1DelP2Combined.data());

    int elementIndexes[elementSize];
    real elementNodes[2 * elementSize];
    SMM::TripletMatrix triplet(numNodes, numNodes * 2);
    real J;
    StaticMatrix<real, 2, 2> B;
    StaticMatrix<real, p1Size, p2Size> b1Local;
    StaticMatrix<real, p1Size, p2Size> b2Local;
    for(int i = 0; i < numElements; ++i) {
        grid.getElement(i, elementIndexes, elementNodes);
        differentialOperator(elementNodes, J, B);

        // Compute local matrices
        for(int p = 0; p < p1Size; ++p) {
            for(int q = 0; q < p2Size; ++q) {
                b1Local[p][q] = B[0][0] * p1DelP2Combined[p][q] + B[0][1] * p1DelP2Combined[p][q + p2Size];
                b2Local[p][q] = B[1][0] * p1DelP2Combined[p][q] + B[1][1] * p1DelP2Combined[p][q + p2Size];
            }
        }
        // Put local matrices into the global matrix
        for(int localRow = 0; localRow < p1Size; ++localRow) {
            const int globalRow = elementIndexes[localRow];
            for(int localCol = 0; localCol < p2Size; ++localCol) {
                const int globalCol = elementIndexes[localCol];
                triplet.addEntry(globalRow, globalCol, b1Local[localRow][localCol]);
                triplet.addEntry(globalRow, globalCol + numNodes, b2Local[localRow][localCol]);
            }
        }
    }
    divergenceMatrix.init(triplet);
}

}