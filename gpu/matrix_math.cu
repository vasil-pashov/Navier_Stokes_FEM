__global__ void sparseMatrixVectorProduct(
    const int rows,
    const int* rowStart,
    const int* columnIndex,
    const float* values,
    const float* x,
    float* res
) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const int currentRowStart = rowStart[row];
    const int currentRowEnd = rowStart[row + 1];
    float sum = 0.0f;
    for(int i = currentRowStart; i < currentRowEnd; ++i) {
        const int column = columnIndex[i];
        sum += values[i] * x[column];
    }
    res[row] = sum;
}