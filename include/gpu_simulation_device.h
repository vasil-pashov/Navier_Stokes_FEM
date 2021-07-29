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
    private:
        CUmodule advectionModule;
        NSFem::KDTreeGPUOwner kdTree;
    };

    class GPUSimulationDeviceManager : public GPU::GPUDeviceManagerBase<GPUSimulationDevice> {
    public:
        GPUSimulationDeviceManager();
        EC::ErrorCode init(const NSFem::KDTreeCPUOwner& cpuOwner);
    };
};