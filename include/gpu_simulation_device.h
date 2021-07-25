#pragma once
#include "gpu_host_common.h"
#include <cuda.h>

namespace EC {
    class ErrorCode;
}

namespace GPUSimulation {
    class GPUSimulationDevice : public GPU::GPUDeviceBase {
    friend class GPUSimulationDeviceManager;
    public:
        EC::ErrorCode saxpyTest(float a, float* x, float* y, int size);
    private:
        CUmodule advectionModule;
        CUfunction saxpyTestKernel;
    };

    class GPUSimulationDeviceManager : public GPU::GPUDeviceManagerBase<GPUSimulationDevice> {
    public:
        GPUSimulationDeviceManager();
        EC::ErrorCode init();
    };
};