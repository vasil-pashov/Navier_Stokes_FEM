#include <cassert>
#include "assembly.h"
#include "grid.h"

namespace NSFem {


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
inline void delP1Shape(const real xi, const real eta, real out[2][3]) {
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
inline void delP2Shape(const real xi, const real eta, real out[2][6]) {
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

/// Integrate vector function of two dimensional vector argument over an unit triangle
/// The unit triangle is right angle triangle with vertices in (0, 0), (0, 1) and (1, 0)
/// This numerical integration formula is exact for polynomials of degree <= 3
/// @tparam TFunctor Type of a functor parameter which takes has arguments:
/// xi, eta - coordinates in the unit triangle and out parameter which is pointer to array
/// of reals where the result of the functor at (xi, eta) will be returned
/// @tparam outSize The size of the otput parameter of TFunctor
/// @param[in] f Vector function of two arguments which will be integrated over the unit triangle.
/// @param[out] out Result from integrating f over the unit triangle (must of of size outSize)
template<int outSize, typename TFunctor>
void integrateOverTriangle(const TFunctor& f, real* out) {
    const int numIntegrationPoints = 8;
    const real weights[numIntegrationPoints] = {
        3.0f / 120.0f, 3.0f / 120.0f, 3.0f / 120.0f,
        8.0f / 120.0f, 8.0f / 120.0f, 8.0f / 120.0f,
        27.0f / 120.0f
    };
    const real nodes[2 * numIntegrationPoints] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.5f, 0.0f,
        0.5f, 0.5f,
        0.0f, 0.5f,
        1.0f/3.0f, 1.0f/3.0f
    };

    real tmp[outSize];
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
inline void differentialOperator(const real* nodes, real& outDetJ, real outB[2][2]) {
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
/// @param[in] lx World x coordinates of the node which will be transformed to (1, 0)
/// @param[in] ly World y coordinates of the node which will be transformed to (1, 0)
/// @param[in] mx World x coordinates of the node which will be transformed to (0, 1)
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
    dt(dt),
    viscosity(viscosity)
{

}

void NavierStokesAssembly::assemble() {
    assemblVelocityMassMatrix();
    assembleVelocityStiffnessMatrix();
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
    const auto squareP2 = [p2Size](const real xi, const real eta, real* out) -> void {
        real p2Res[p2Size];
        p2Shape(xi, eta, p2Res);
        for(int i = 0; i < p2Size; ++i) {
            for(int j = 0; j < p2Size; ++j) {
                const int idx = i * 6 + j;
                out[idx] = p2Res[i]*p2Res[j];
            }
        }        
    };
    integrateOverTriangle<p2Size * p2Size>(squareP2, (real*)p2Squared);

    // Lambda wich takes advantage of precomputed shape function integral
    const auto localMass = [&p2Squared, p2Size](real* elementNodes, real* localMatrixOut) -> void {
        const real jDetAbs = std::abs(linTriangleTmJacobian(elementNodes));
        for(int i = 0; i < p2Size; ++i) {
            for(int j = 0; j < p2Size; ++j) {
                const int index = i * p2Size + j;
                localMatrixOut[index] = p2Squared[i][j] * jDetAbs;
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

    const int pSize = 6;
    const int delPSize = pSize * 2;
    // Compute the integral of each pair shape function derivatives.
    const auto squareDelP = [delPSize, pSize](const real xi, const real eta, real* out) -> void{
        real delP[2][pSize] = {};
        delP2Shape(xi, eta, delP);
        for(int i = 0; i < delPSize; ++i) {
            for(int j = 0; j < delPSize; ++j) {
                const int outIndex = linearize2DIndex(delPSize, i, j);
                const real delPsi1 = *(reinterpret_cast<real*>(delP) + i);
                const real delPsi2 = *(reinterpret_cast<real*>(delP) + j);
                out[outIndex] = delPsi1 * delPsi2;
            }
        }
    };
    real delPSq[delPSize * delPSize] = {};
    integrateOverTriangle<delPSize * delPSize>(squareDelP, delPSq);

    const auto localStiffness = [&delPSq, pSize](real* elementNodes, real* localMatrixOut) -> void {
        real b[2][2];
        real J;
        differentialOperator(elementNodes, J, b);
        J = real(1.0) / std::abs(J);
        for(int i = 0; i < pSize; ++i) {
            for(int j = 0; j < pSize; ++j) {
                const int outIndex = linearize2DIndex(pSize, i, j);
                localMatrixOut[outIndex] = 0;
                const real sq[4] = {
                    delPSq[linearize2DIndex(2 * pSize, i, j)],
                    delPSq[linearize2DIndex(2 * pSize, i, pSize + j)],
                    delPSq[linearize2DIndex(2 * pSize, pSize + i, j)], 
                    delPSq[linearize2DIndex(2 * pSize, pSize + i, pSize + j)]
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
                    localMatrixOut[outIndex] += sq[0]*b[k][0]*b[k][0] + sq[1]*b[k][0]*b[k][1] + sq[2]*b[k][0]*b[k][1] + sq[3]*b[k][1]*b[k][1];
                }
                localMatrixOut[outIndex] *= J;
            }
        }
    };

    assembleMatrix<decltype(localStiffness), pSize, pSize>(localStiffness, stiffnessMatrix);
}

}