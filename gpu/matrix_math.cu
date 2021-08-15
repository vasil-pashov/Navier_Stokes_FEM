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
__device__ void spRMult(
    const int rows,
    const int* rowStart,
    const int* columnIndex,
    const float* values,
    const float* mult,
    float* res
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    while(row < rows) {
        const int currentRowStart = rowStart[row];
        const int currentRowEnd = rowStart[row + 1];
        float sum = 0.0f;
        for(int i = currentRowStart; i < currentRowEnd; ++i) {
            const int column = columnIndex[i];
            sum += values[i] * mult[column];
        }
        res[row] = sum;
        row += gridDim.x * blockDim.x;
    }
}

extern "C" __global__ void spRMultKernel(
    const int rows,
    const int* rowStart,
    const int* columnIndex,
    const float* values,
    const float* mult,
    float* res
) {
    spRMult(rows, rowStart, columnIndex, values, mult, res);
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
__device__ void spRMultSub(
    const int rows,
    const int* rowStart,
    const int* columnIndex,
    const float* values,
    const float* lhs,
    const float* mult,
    float* res
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    while(row < rows) {
        const int currentRowStart = rowStart[row];
        const int currentRowEnd = rowStart[row + 1];
        float sum = 0.0f;
        for(int i = currentRowStart; i < currentRowEnd; ++i) {
            const int column = columnIndex[i];
            sum += values[i] * mult[column];
        }
        res[row] = lhs[row] - sum;
        row += gridDim.x * blockDim.x;
    }
}

extern "C" __global__ void spRMultSubKernel(
    const int rows,
    const int* rowStart,
    const int* columnIndex,
    const float* values,
    const float* lhs,
    const float* mult,
    float* res
) {
    spRMultSub(rows, rowStart, columnIndex, values, lhs, mult, res);
}

/// Perform a * x + y where a is scalar, x and y are vectors. The result is stored in y
/// @param[in] vectorLength The number of elemens in both x and y vectors
/// @param[in] a The scalar which will multiply each element of x vector
/// @param[in] x x vector from the equation y = a * x + y
/// @param[inout] y y vector from the equation y = a * x + y. The result is stored in this vector
__device__ void saxpy(
    const int vectorLength,
    const float a,
    const float* x,
    float* y
) {
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  while(i < vectorLength) {
      y[i] += a*x[i];
      i += gridDim.x * blockDim.x;
  }
}

extern "C" __global__ void saxpyKernel(
    const int vectorLength,
    const float a,
    const float* x,
    float* y
) {
    saxpy(vectorLength, a, x, y);
}

/// Perform a * x + b * y where a and b are scalars and x and y are vectors.
/// @param[in] vectorLength The number of elements in vectors x and y
/// @param[in] a Scalar multiplier for the x vector
/// @param[in] b Scalar multiplier for the y vector
/// @param[in] x Vector multiplied by a
/// @param[in] y Vector multiplied by b
/// @param[out] result Vector where the result is stored
__device__ void saxpby(
    const int vectorLength,
    const float a,
    const float b,
    const float* x,
    const float* y,
    float* result
) {
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  while(i < vectorLength) {
      result[i] = a * x[i] + b * y[i];
      i += gridDim.x * blockDim.x;
  }
}
extern "C" __global__ void saxpbyKernel(
    const int vectorLength,
    const float a,
    const float b,
    const float* x,
    const float* y,
    float* result
) {
    saxpby(vectorLength, a, b, x, y, result);
}

/// Perform dot product between a and b vectors and store in result
/// @param[in] vectorLength The length of both a and b vectors
/// @param[in] a The first vector to dot
/// @param[in] b The second vector to dot
/// @param[out] result The result from dot(a, b)
__device__ void dotProduct(
    const int vectorLength,
    const float* a,
    const float* b,
    float* result
) {
    __shared__ float cache[512];

    unsigned tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int cacheIndex = threadIdx.x;
    float sum = 0.0f;
    while(tid < vectorLength) {
        sum += a[tid] * b[tid];
        tid += gridDim.x * blockDim.x;
    }
    cache[cacheIndex] = sum;
    __syncthreads();

    for(int i = blockDim.x / 2; i > 0; i >>= 1) {
        if(cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
    }
    if(cacheIndex == 0) {
        atomicAdd(result, cache[0]);
    }
}

extern "C" __global__ void dotProductKernel(
    const int vectorLength,
    const float* a,
    const float* b,
    float* result
) {
    dotProduct(vectorLength, a, b, result);
}