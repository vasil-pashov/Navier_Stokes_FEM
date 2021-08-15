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
    int* barrier;
    int rows;
};
#endif