#pragma once
#include <cuda.h>
#include <vector>
#include <unordered_map>


namespace EC {
    class ErrorCode;
}

struct Dim3 {
    explicit Dim3(int x) : Dim3(x, 1, 1)
    {

    }
    Dim3(int x, int y) : Dim3(x, y, 1)
    {

    }
    Dim3(int x, int y, int z) : 
        x(x), y(y), z(z)
    {

    }
    int x;
    int y;
    int z;
};

struct KernelLaunchParams {
    KernelLaunchParams(Dim3 gridSize, Dim3 blockSize, void** kernelParams) :
        KernelLaunchParams(gridSize, blockSize, 0, nullptr, kernelParams, nullptr)
    {

    }
    KernelLaunchParams(Dim3 gridSize, Dim3 blockSize, int sharedMemSize, CUstream stream, void** kernelParams, void** extra) :
        gridSize(gridSize),
        blockSize(blockSize),
        stream(stream),
        kernelParams(kernelParams),
        extra(extra),
        sharedMemSize(sharedMemSize)
    {}
    Dim3 gridSize;
    Dim3 blockSize;
    CUstream stream;
    void** kernelParams;
    void** extra;
    int sharedMemSize;
};

namespace GPU {
class GPUDevice {
public:
    friend class GPUDeviceManager;
    GPUDevice() = default;
    EC::ErrorCode init(int index);
    EC::ErrorCode addModule(const char* moduleSource, char** kernelNames, int kernelCount, CUmodule* out = nullptr);
    EC::ErrorCode callKernelSync(const std::string& name, const KernelLaunchParams& params);
    ~GPUDevice();
private:
    void printDeviceInfo() const;
    CUdevice handle;
    CUcontext context;
    std::vector<CUmodule> modules;
    std::unordered_map<std::string, CUfunction> kernels;
};

class GPUDeviceManager {
public:
    GPUDeviceManager() = default;
    GPUDeviceManager(const GPUDeviceManager&) = delete;
    GPUDeviceManager& operator=(const GPUDeviceManager&) = delete;
    EC::ErrorCode init();
    EC::ErrorCode addModuleFromFile(const char* filePath, char** kernelNames, int kernelCount);
private:
    std::vector<GPUDevice> devices;
};
} // namespace GPU
