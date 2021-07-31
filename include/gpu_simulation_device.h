#pragma once
#include "gpu_host_common.h"
#include "kd_tree_builder.h"
#include <cuda.h>

namespace EC {
    class ErrorCode;
}

namespace GPUSimulation {
    class GPUSimulationDevice : public GPU::GPUDeviceBase {
    friend class GPUSimulationDeviceManager;
    public:
        EC::ErrorCode uploadKDTree(const NSFem::KDTreeCPUOwner& cpuOwner);
        EC::ErrorCode initVelocityBuffers(const int numElements);
        EC::ErrorCode advect(
            const int numElements,
            const float* uVelocity,
            const float* vVelocity,
            const float dt,
            float* uVelocityOut,
            float* vVelocityOut
        );
    private:
        CUmodule advectionModule;
        CUfunction advectionKernel;
        NSFem::KDTreeGPUOwner kdTree;

        GPU::GPUBuffer uVelocityInBuffer;
        GPU::GPUBuffer vVelocityInBuffer;

        GPU::GPUBuffer uVelocityOutBuffer;
        GPU::GPUBuffer vVelocityOutBuffer;
    };

    class GPUSimulationDeviceManager : public GPU::GPUDeviceManagerBase<GPUSimulationDevice> {
    public:
        GPUSimulationDeviceManager();
        EC::ErrorCode init(const NSFem::KDTreeCPUOwner& cpuOwner);
    };
};