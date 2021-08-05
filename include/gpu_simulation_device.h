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
            pressureSitffnes,
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
        EC::ErrorCode spRMult(
            SimMatrix matrix,
            const GPU::GPUBuffer& x,
            GPU::GPUBuffer& res
        );
        EC::ErrorCode spRMultSub(
            const SimMatrix matrix,
            const GPU::GPUBuffer& mult,
            const GPU::GPUBuffer& lhs,
            GPU::GPUBuffer& res
        );
    private:
        EC::ErrorCode loadAdvectionModule(const char* data);
        EC::ErrorCode loadSparseMatrixModule(const char* data);

        GPUSimulation::GPUSparseMatrix matrices[SimMatrix::count];

        CUmodule advectionModule;
        CUmodule sparseMatrixModule;

        CUfunction advection_kernel;
        CUfunction spRMult_kernel;
        CUfunction spRMultSub_kernel;

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