#include "kd_tree_common.cuh"

extern "C" __global__ void saxpy(int n, float a, float *x, float *y) {
  const unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a*x[i] + y[i];
  }
}
