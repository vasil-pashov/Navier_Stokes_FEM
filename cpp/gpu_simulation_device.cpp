#include "gpu_simulation_device.h"
#include "error_code.h"
#include <array>
#include <memory>
#include <fstream>
namespace GPUSimulation {

EC::ErrorCode GPUSimulationDevice::uploadKDTree(const NSFem::KDTreeCPUOwner& cpuOwner) {
    GPU::ScopedGPUContext contextGuard(context);
    RETURN_ON_ERROR_CODE(cpuOwner.upload(kdTree));
    const int numNodes = kdTree.getGrid().getNodesCount();
    return initVelocityBuffers(numNodes);
}

EC::ErrorCode GPUSimulationDevice::initVelocityBuffers(const int numElements) {
    const int64_t size = numElements * sizeof(float);

    RETURN_ON_ERROR_CODE(uVelocityInBuffer.init(size));
    RETURN_ON_ERROR_CODE(vVelocityInBuffer.init(size));

    RETURN_ON_ERROR_CODE(uVelocityOutBuffer.init(size));
    RETURN_ON_ERROR_CODE(vVelocityOutBuffer.init(size));

    return EC::ErrorCode();
}

EC::ErrorCode GPUSimulationDevice::advect(
    int numElements,
    const float* uVelocity,
    const float* vVelocity,
    const float dt,
    float* uVelocityOut,
    float* vVelocityOut
) {
    GPU::ScopedGPUContext ctxGuard(context);
    const int64_t bufferSize = numElements * sizeof(float);
    assert(
        bufferSize <= uVelocityInBuffer.getByteSize() && bufferSize <= vVelocityInBuffer.getByteSize() &&
        bufferSize <= uVelocityOutBuffer.getByteSize() && bufferSize <= vVelocityOutBuffer.getByteSize()
    );
    RETURN_ON_ERROR_CODE(uVelocityInBuffer.uploadBuffer((const void*)uVelocity, bufferSize));
    RETURN_ON_ERROR_CODE(vVelocityInBuffer.uploadBuffer((const void*)vVelocity, bufferSize));
    GPUFemGrid2D grid = kdTree.getGrid();
    NSFem::KDTree<GPUFemGrid2D> tree = kdTree.getTree();
    
    const GPU::Dim3 blockSize(512);
    const GPU::Dim3 gridSize((grid.getNodesCount() + blockSize.x) / blockSize.x);
    void* kernelParams[] = {
        (void*)&tree,
        (void*)&grid,
        (void*)&uVelocityInBuffer.getHandle(),
        (void*)&vVelocityInBuffer.getHandle(),
        (void*)&uVelocityOutBuffer.getHandle(),
        (void*)&vVelocityOutBuffer.getHandle(),
        (void*)&dt

    };
    GPU::KernelLaunchParams params(
        blockSize,
        gridSize,
        kernelParams
    );
    RETURN_ON_ERROR_CODE(callKernelSync(advectionKernel, params));
    RETURN_ON_ERROR_CODE(uVelocityOutBuffer.downloadBuffer(uVelocityOut));
    RETURN_ON_ERROR_CODE(vVelocityOutBuffer.downloadBuffer(vVelocityOut));
    return EC::ErrorCode();
}

GPUSimulationDeviceManager::GPUSimulationDeviceManager() : 
    GPUDeviceManagerBase<GPUSimulationDevice>()
{

}

EC::ErrorCode GPUSimulationDeviceManager::init(const NSFem::KDTreeCPUOwner& cpuOwner) {
    RETURN_ON_ERROR_CODE(GPUDeviceManagerBase<GPUSimulationDevice>::initDevices());
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

    std::array<const char*, 1> kernelsToExtract = {
        "advect"
    };

    for(GPUSimulationDevice& device : devices) {
        std::array<CUfunction*, 1> kernelPointers = {
            &device.advectionKernel
        };
        static_assert(kernelsToExtract.size() == kernelPointers.size());
        RETURN_ON_ERROR_CODE(device.loadModule(
            data.get(),
            kernelsToExtract.data(),
            kernelsToExtract.size(),
            device.advectionModule,
            kernelPointers.data()
        ));
        RETURN_ON_ERROR_CODE(device.uploadKDTree(cpuOwner));
    }

    return EC::ErrorCode();
}

}