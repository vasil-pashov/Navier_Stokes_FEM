#pragma once
#include "gpu_host_common.h"
#include "kd_tree_builder.h"
#include "gpu_sparse_matrix.h"
#include "sparse_matrix_math.h"
#include <cuda.h>

namespace EC {
    class ErrorCode;
}

namespace GPUSimulation {
    class GPUSimulationDevice : public GPU::GPUDeviceBase {
    friend class GPUSimulationDeviceManager;
    public:
        enum SimMatrix {
            pressureSitffness,
            velocityMass,
            diffusion,
            velocityDivergence,
            pressureDivergence,
            count
        };
        EC::ErrorCode loadModules(const char* advectionData, const char* sparseMatrixData);
        EC::ErrorCode uploadKDTree(const NSFem::KDTreeCPUOwner& cpuOwner);
        EC::ErrorCode initVelocityBuffers(const int numElements);
        EC::ErrorCode uploadMatrix(SimMatrix matrix, const SMM::TripletMatrix<float>& triplet);
        EC::ErrorCode advect(
            const int numElements,
            const float* uVelocity,
            const float* vVelocity,
            const float dt,
            float* uVelocityOut,
            float* vVelocityOut
        );

        /// Perform sparse matrix vector multiplication matrix * mult.
        /// @param[in] matrix Enum value of SimMatrix, stating which one of the device matrices will take
        /// part in the expresion.
        /// @param[in] mult The vector which multiples the matrix (should not overlap with res)
        /// @param[out] res The result of the vector matrix product (should not overlap with mult)
        EC::ErrorCode spRMult(
            SimMatrix matrix,
            const GPU::GPUBuffer& mult,
            GPU::GPUBuffer& res
        );

        /// Perform lhs - matrix * mult. Where lhs and mult are vectors.
        /// @param[in] matrix Enum value of SimMatrix, stating which one of the device matrices will take
        /// part in the expresion. 
        /// @param[in] lhs The vector from which A * rhs will be subtracted (can overlap with res)
        /// @param[in] mult The vector which multiples the matrix (should not overlap with res)
        /// @param[out] res The result of the vector matrix product (should not overlap with rhs, can overlap with lhs) 
        EC::ErrorCode spRMultSub(
            const SimMatrix matrix,
            const GPU::GPUBuffer& mult,
            const GPU::GPUBuffer& lhs,
            GPU::GPUBuffer& res
        );

        EC::ErrorCode conjugateGradient(
            const SimMatrix matrix,
            const float* const b,
            const float* const x0,
            float* const x,
            int maxIterations,
            float eps
        );
        /// Perform a * x + y where a is scalar, x and y are vectors. The result is stored in y
        /// @param[in] vectorLength The number of elemens in both x and y vectors
        /// @param[in] a The scalar which will multiply each element of x vector
        /// @param[in] x x vector from the equation y = a * x + y
        /// @param[in] y y vector from the equation y = a * x + y. The result is stored in this vector
        EC::ErrorCode saxpy(
            const int vectorLength,
            const float a,
            const GPU::GPUBuffer& x,
            GPU::GPUBuffer& y
        );
        
    private:
        EC::ErrorCode loadAdvectionModule(const char* data);
        EC::ErrorCode loadSparseMatrixModule(const char* data);

        GPUSimulation::GPUSparseMatrix matrices[SimMatrix::count];

        enum class Modules {
            advection,
            sparseMatrix,
            count
        };

        enum class AdvectionKernels {
            advect,
            count
        };

        enum class SparseMatrixKernels {
            spRMult,
            spRMultSub,
            saxpy,
            dotProduct,
            count
        };

        CUmodule modules[int(Modules::count)];
        CUfunction advectionKernels[int(AdvectionKernels::count)];
        CUfunction sparseMatrixKernels[int(SparseMatrixKernels::count)];

        NSFem::KDTreeGPUOwner kdTree;

        GPU::GPUBuffer uVelocityInBuffer;
        GPU::GPUBuffer vVelocityInBuffer;

        GPU::GPUBuffer uVelocityOutBuffer;
        GPU::GPUBuffer vVelocityOutBuffer;
    };

    class GPUSimulationDeviceManager : public GPU::GPUDeviceManagerBase<GPUSimulationDevice> {
    private:
        EC::ErrorCode loadModule(
            const char* filePath,
            const int numKernels,
            const char* kernelsToExtract[],
            CUmodule& moduleOut,
            CUfunction** kernelsOut
        );
    public:
        GPUSimulationDeviceManager();
        EC::ErrorCode init();
    };
};