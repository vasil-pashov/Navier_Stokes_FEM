#include "gpu_simulation_device.h"
#include "error_code.h"
#include <array>
#include <memory>
#include <fstream>
namespace GPUSimulation {

EC::ErrorCode GPUSimulationDevice::uploadKDTree(const NSFem::KDTreeCPUOwner& cpuOwner) {
    const EC::ErrorCode status = cpuOwner.upload(kdTree);
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
    return EC::ErrorCode();
}

}