#include "kd_tree_common.cuh"
#include "gpu_grid.cuh"
#include "kd_tree.cuh"

static __device__ void P2ShapeEval(const float xi, const float eta, float (&out)[6]) {
  out[0] = 1 - 3 * xi - 3 * eta + 2 * xi * xi + 4 * xi * eta + 2 * eta * eta;
  out[1] = 2 * xi * xi - xi;
  out[2] = 2 * eta * eta - eta;
  out[3] = 4 * xi * eta;
  out[4] = 4 * eta - 4 * xi * eta - 4 * eta * eta;
  out[5] = 4 * xi - 4 * xi * xi - 4 * xi * eta;
}

extern "C" __global__ void advect(
  NSFem::KDTree<GPUSimulation::GPUFemGrid2D> kdTree,
  GPUSimulation::GPUFemGrid2D grid,
  const float* uVelocity,
  const float* vVelocity,
  float* uVelocityOut,
  float* vVelocityOut,
  const float dt
) { 
  const unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= grid.getNodesCount()) return;
  const int elementSize = 6;
  const NSFem::Point2D position = grid.getNode(i);
  const NSFem::Point2D velocity(uVelocity[i], vVelocity[i]);
  const NSFem::Point2D start = position - velocity * dt;

  // If start is inside some element xi and eta will be the barrycentric coordinates
  // of start inside that element.
  float xi, eta;
  // If start does not lie in any triangle this will be the index of the nearest node to start
  int nearestNeighbour;
  const int element = kdTree.findElement(start, xi, eta, nearestNeighbour);
  if(element > -1) {
      // Start point lies in an element, interpolate it by using the shape functions.
      // This is possible because xi and eta do not change when the element is transformed
      // to the unit element where the shape functions are defined. 
      float uResult = 0, vResult = 0;
      const int* elementIndexes = grid.getElement(element);
      float interpolationCoefficients[elementSize];
      P2ShapeEval(xi, eta, interpolationCoefficients);
      for(int k = 0; k < elementSize; ++k) {
          const int nodeIndex = elementIndexes[k];
          uResult += interpolationCoefficients[k] * uVelocity[nodeIndex];
          vResult += interpolationCoefficients[k] * vVelocity[nodeIndex];
      }
      uVelocityOut[i] = uResult;
      vVelocityOut[i] = vResult;
  } else {
      // Start point does not lie in any element (probably it's outside the mesh)
      // Use the closest point in the mesh to approximate the velocity
      uVelocityOut[i] = uVelocity[nearestNeighbour];
      vVelocityOut[i] = vVelocity[nearestNeighbour];
  }
}
