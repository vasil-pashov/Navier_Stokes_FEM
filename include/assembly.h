#pragma once
#include <sparse_matrix_math/sparse_matrix_math.h>
#include <grid.h>

namespace NSFem {

/// Shape functions for 2D triangluar element with degrees of freedom in each triangle node
/// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
/// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
/// @param[out] out - The value of each shape function at (xi, eta). The array must have at least 6 elements.
///< Order: [0, 1, 2] - The values at the nodes of the triangle
inline void p1Shape(const float xi, const float eta, float* out) {
    out[0] = 1 - xi - eta;
    out[1] = xi;
    out[2] = eta;
}

/// Compute the gradient, at point (xi, eta) of the shape functions for triangular element with degrees of freedom in each triangle node
/// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
/// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
/// @param[out] out - This will hold the gradient. The first 3 elements are the derivatives of the shape functions with
///< respect to xi, next 3 elements are the derivatives of the shape functions with respect to eta
inline void delP1Shape(const float xi, const float eta, float* out) {
    // dpsi/dxi
    out[0] = -1.0f;
    out[1] = 1.0f;
    out[2] = 0.0f;

    // dpsi/deta
    out[3] = -1.0f;
    out[4] = 0.0f;
    out[5] = 1.0f;
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
inline void p2Shape(const float xi, const float eta, float* out) {
    out[0] = 1 - 3 * xi - 3 * eta + 2 * xi * xi + 4 * xi * eta + 2 * xi * xi;
    out[1] = 2 * xi * xi - xi;
    out[2] = 2 * eta * eta - eta;
    out[3] = 4 * xi * eta;
    out[4] = 4 * xi - 4 * xi * eta - 4 * eta * eta;
    out[5] = 4 * xi - 4 * eta * eta - 4 * xi * eta;
}

/// Compute the gradient, at point (xi, eta) of the shape functions for triangular element with degrees of freedom
/// in each triangle node and in the middle of each side
/// @param[in] xi - Coordinate in the (transformed) unit triangle along the xi (aka x) axis
/// @param[in] eta - Coordinate in the (transformed) unit triangle anong the eta (aka y) axis
/// @param[out] out - This will hold the gradient. The first 6 elements are the derivatives of the shape functions with
/// respect to xi, next 6 elements are the derivatives of the shape functions with respect to eta
inline void delP2Shape(const float xi, const float eta, float* out) {
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
/// of floats where the result of the functor at (xi, eta) will be returned
/// @tparam outSize The size of the otput parameter of TFunctor
/// @param[in] f Vector function of two arguments which will be integrated over the unit triangle.
/// @param[out] out Result from integrating f over the unit triangle (must of of size outSize)
template<int outSize, typename TFunctor>
void integrateOverTriangle(const TFunctor& f, float* out) {
    const int numIntegrationPoints = 8;
    const float weights[numIntegrationPoints] = {
        3.0f / 120.0f, 3.0f / 120.0f, 3.0f / 120.0f,
        8.0f / 120.0f, 8.0f / 120.0f, 8.0f / 120.0f,
        27.0f / 120.0f
    };
    const float nodes[2 * numIntegrationPoints] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.5f, 0.0f,
        0.5f, 0.5f,
        0.0f, 0.5f,
        1.0f/3.0f, 1.0f/3.0f
    };

    float tmp[outSize];
    for(int i = 0; i < numIntegrationPoints; ++i) {
        const float x = nodes[2 * i];
        const float y = nodes[2 * i + 1];
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
inline void differentialOperator(const float* nodes, float& outDetJ, float outB[2][2]) {
    const float xk = nodes[0];
    const float yk = nodes[1];
    const float xl = nodes[2];
    const float yl = nodes[3];
    const float xm = nodes[4];
    const float ym = nodes[5];
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
inline float linTriangleTmJacobian(
    const float kx, const float ky,
    const float lx, const float ly,
    const float mx, const float my
) {
    // The Jacobi matrix is given by
    // |xl - xk     yl - yk|
    // |xm - xk     ym - yk|
    const float a = lx - kx;
    const float b = ly - ky;
    const float c = mx - kx;
    const float d = my - ky;
    return a * d - b * c;
}

class NavierStokesAssembly {
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

    template<typename TLocalF, int localRows, int localCols>
    void assembleMatrix(const TLocalF& localFunction, SMM::CSRMatrix& out);

    void assembleMassMatrix();

    /// Viscosity of the fluid
    float viscosity;
    /// Size of the time step used when approximating derivatives with respect to time
    float dt;
};

template<typename TLocalF, int localRows, int localCols>
void NavierStokesAssembly::assembleMatrix(const TLocalF& localFunction, SMM::CSRMatrix& out) {
    const int numNodes = grid.getNodesCount();
    const int numElements = grid.getElementsCount();
    const int elementSize = std::max(localRows, localCols);
    int element[elementSize];
    float elementNodes[2 * elementSize];
    float localMatrix[localRows][localCols];
    SMM::TripletMatrix triplet;
    for(int i = 0; i < numElements; ++i) {
        grid.getElement(i, element, elementNodes);
        localFunction(elementNodes, &localMatrix[0][0]);
        for(int localRow = 0; localRow < localRows; ++localRow) {
            const int globalRow = element[localRow];
            for(int localCol = 0; localCol < localCols; ++localCol) {
                const int globalCol = element[localCol];
                triplet.addEntry(globalRow, globalCol, localMatrix[localRow][localCol]);
            }
        }
    }
    out.init(triplet);
}

}