#include "gpu_simulation_device.h"
#include "error_code.h"
#include <array>
#include <memory>
#include <fstream>
namespace GPUSimulation {

EC::ErrorCode GPUSimulationDevice::saxpyTest(float alpha, float* x, float* y, int size) {
    GPU::ScopedGPUContext ctx(context);
    GPU::GPUBuffer xGpuBuffer, yGpuBuffer;

    EC::ErrorCode status = xGpuBuffer.init(size * sizeof(float));
    RETURN_ON_ERROR_CODE(status);
    status = xGpuBuffer.uploadBuffer(x, size * sizeof(float));
    RETURN_ON_ERROR_CODE(status);

    status = yGpuBuffer.init(size * sizeof(float));
    RETURN_ON_ERROR_CODE(status);
    status = yGpuBuffer.uploadBuffer(y, size * sizeof(float));

    const int blockSizeX = 256;
    const GPU::Dim3 gridSize((size + blockSizeX) / blockSizeX);
    const GPU::Dim3 blockSize(blockSizeX);
    void* kernelParams[] = {
        (void*)&size,
        (void*)&alpha,
        (void*)&xGpuBuffer.getHandle(),
        (void*)&yGpuBuffer.getHandle()
    };
    const GPU::KernelLaunchParams launchParams(
        gridSize,
        blockSize,
        kernelParams
    );
    callKernelSync(saxpyTestKernel, launchParams);
    status = yGpuBuffer.downloadBuffer(y);
    return status;
}

GPUSimulationDeviceManager::GPUSimulationDeviceManager() : 
    GPUDeviceManagerBase<GPUSimulationDevice>()
{

}

EC::ErrorCode GPUSimulationDeviceManager::init() {
    EC::ErrorCode status = GPUDeviceManagerBase<GPUSimulationDevice>::initDevices();
    if(status.hasError()) {
        return status;
    }
    const int numKernelsToLoad = 1;
    std::array<const char*, 1> kernelsToLoad = {
        "saxpy"
    };
    // TODO: Fix the hardcoded path
    const char* filePath = "/home/vasil/Documents/FMI/Магистратура/Дипломна/CPP/fem_solver/build/ptx/advection.ptx";
    std::ifstream file(filePath, std::ifstream::ate | std::ifstream::binary);
    if(file.fail()) {
        return EC::ErrorCode(errno, "%d: %s. Cannot open file: %s.", errno, strerror(errno), filePath);
    }
    const int64_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> data(new char[fileSize + 1]);
    data[fileSize] = '\0';
    file.read(data.get(), fileSize);
    for(GPUSimulationDevice& device : devices) {
        std::array<CUfunction*, numKernelsToLoad> kernelsOut {
            &device.saxpyTestKernel
        };
        status = device.loadModule(
            data.get(),
            kernelsToLoad.data(),
            kernelsToLoad.size(),
            device.advectionModule,
            kernelsOut.data()
        );
        if(status.hasError()) {
            return status;
        }
    }
    return EC::ErrorCode();
}

}