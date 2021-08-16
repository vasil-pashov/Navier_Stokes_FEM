#ifndef MATRIX_MATH_COMMON_H
#define MATRIX_MATH_COMMON_H
struct CGParams {
    int* rowStart;
    int* columnIndex;
    float* values;
    float* x;
    float* p;
    float* ap;
    float* r;
    float* residualNormSquared;
    float* newResidualNormSquared;
    float* pAp;
    unsigned int* barrier;
    unsigned int* generation;
    int rows;
    int maxIterations;
    float epsSq;
};
#endif