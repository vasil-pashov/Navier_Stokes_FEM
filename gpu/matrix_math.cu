/// Multuply a matrix in CSR format with a dense vector. The vector is on the right hand side of the matrix.
/// @param[in] rows The number of rows of the matrix
/// @param[in] rowStart Array with length the number of rows + 1,
/// holding where each row starts in columnIndex and values arrays
/// @param[in] columnIndex elements in range [columnIndex[rowStart[i]]]...columnIndex[rowStart[i+1]]
/// are the columns of the elements of the i-th row
/// @param[in] values elements in range [columnIndex[rowStart[i]]]...columnIndex[rowStart[i+1]] are 
/// the values of the elements in the row.
/// @param[in] mult The vector which multiples the matrix (should not overlap with res)
/// @param[out] res The result of the vector matrix product (should not overlap with mult)
__global__ void spRMult(
    const int rows,
    const int* rowStart,
    const int* columnIndex,
    const float* values,
    const float* mult,
    float* res
) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const int currentRowStart = rowStart[row];
    const int currentRowEnd = rowStart[row + 1];
    float sum = 0.0f;
    for(int i = currentRowStart; i < currentRowEnd; ++i) {
        const int column = columnIndex[i];
        sum += values[i] * mult[column];
    }
    res[row] = sum;
}


/// Multuply a matrix in CSR format with a dense vector and subtract this from a vector. Performing lhs - A * mult
/// @param[in] rows The number of rows of the matrix
/// @param[in] rowStart Array with length the number of rows + 1,
/// holding where each row starts in columnIndex and values arrays
/// @param[in] columnIndex elements in range [columnIndex[rowStart[i]]]...columnIndex[rowStart[i+1]]
/// are the columns of the elements of the i-th row
/// @param[in] values elements in range [columnIndex[rowStart[i]]]...columnIndex[rowStart[i+1]] are 
/// the values of the elements in the row.
/// @param[in] lhs The vector from which A * rhs will be subtracted (can overlap with res)
/// @param[in] mult The vector which multiples the matrix (should not overlap with res)
/// @param[out] res The result of the vector matrix product (should not overlap with rhs, can overlap with lhs)
__global__ void spRMultSub(
    const int rows,
    const int* rowStart,
    const int* columnIndex,
    const float* values,
    const float* lhs,
    const float* mult,
    float* res
) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const int currentRowStart = rowStart[row];
    const int currentRowEnd = rowStart[row + 1];
    float sum = 0.0f;
    for(int i = currentRowStart; i < currentRowEnd; ++i) {
        const int column = columnIndex[i];
        sum += values[i] * mult[column];
    }
    res[row] = lhs[row] - sum;
}