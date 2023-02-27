#ifndef MATRIX_MATH_COMMON_H
#define MATRIX_MATH_COMMON_H
#include<type_traits>

/// When the GPU does not support cooperative groups we want all intermediate
/// computations to be volatile. Otherwise when we call sync the threads will
/// be synced, but some of the memory will be cached in each block and the
/// cache will be different than the real value.
/// TODO: Remove the custom sync and the need for volatile pointers
template<bool useVolatile = false>
struct CGParams {
    using T = std::conditional_t<useVolatile, volatile float*, float*>;
    const int* rowStart;
    const int* columnIndex;
    const float* values;
    T x;
    T p;
    T ap;
    T r;
    T residualNormSquared;
    T newResidualNormSquared;
    T pAp;
    unsigned int* barrier; // TODO remove not needed
    unsigned int* generation; // TODO remove not needed
    int rows;
    int maxIterations;
    float epsSq;
};
#endif