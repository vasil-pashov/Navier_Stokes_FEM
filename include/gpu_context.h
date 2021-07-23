#pragma once
#include <cuda.h>
#include <vector>
#include <unordered_map>


namespace EC {
    class ErrorCode;
}

namespace GPU {

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


class GPUDevice {
public:
    friend class GPUDeviceManager;
    GPUDevice() = default;
    EC::ErrorCode init(int index);
    EC::ErrorCode addModule(const char* moduleSource, const char* kernelNames[], int kernelCount, CUmodule* out = nullptr);
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
    EC::ErrorCode addModuleFromFile(const char* filePath, const char* kernelNames[], int kernelCount);
    GPUDevice& getDevice(int index) {
        return devices[index];
    }
private:
    std::vector<GPUDevice> devices;
};

/// Simple class to wrap around a buffer which lives on the GPU
/// It has functionalities to allocate, free, and upload data to the GPU
class GPUBuffer {
public:
    GPUBuffer() :
        handle(0),
        byteSize(0)
    { }

    /// Initialize the GPU memory by allocating byteSize bytes. It may fail silently
    /// if so byteSize and handle will be set to the same values as the default constructr.
    explicit GPUBuffer(int64_t byteSize);
        
    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;
    GPUBuffer(GPUBuffer&& other) :
        handle(other.handle),
        byteSize(other.byteSize)
    {
        other.handle = 0;
        other.byteSize = 0;
    }
    GPUBuffer& operator=(GPUBuffer&& other) {
        handle = other.handle;
        byteSize = other.byteSize;
        other.handle = 0;
        other.byteSize = 0;
        return *this;
    }

    /// Destructor. Will free all GPU memory allocated by the buffe.
    ~GPUBuffer();

    /// Initialize the GPU memory by allocating byteSize bytes. If the buffer was already allocated
    /// it will first deallocated the old memory.
    /// @param byteSize Number of bytes which will be allocated on the GPU
    /// @returns Status of operation.
    EC::ErrorCode init(int64_t byteSize);

    /// Copy uploadByteSize bytes for src to the GPU starting destOffset bytes into the GPU memory
    /// @param src CPU buffer which will be uploaded to the GPU
    /// @param uploadByteSize How many bytes from src will be uploaded to the GPU
    /// @param destOffset Offset into the GPU memory where src will be copied
    /// @returns Status of the operation.
    EC::ErrorCode uploadBuffer(const void* src, const int64_t uploadByteSize, const int64_t destOffset);

    /// Copy uploadByteSize bytes from src to the GPU starting from the begining of the GPU memory
    /// @param src CPU buffer which will be uploaded to the GPU
    /// @param uploadByteSize How many bytes from src will be uploaded to the GPU
    /// @returns Status of the task.
    EC::ErrorCode uploadBuffer(const void* src, const int64_t uploadByteSize);

    /// Copy the whole GPU buffer to a CPU buffer. The CPU buffer must be preallocated and must have
    /// byte size greater or equal to byteSize
    EC::ErrorCode downloadBuffer(void* dst);

    /// Copy donwloadByteSize bytes from GPU buffer to a CPU buffer, starting from the begining for the GPU buffer.
    /// The CPU buffer must be preallocated and must have byte size greater or equal to donwloadByteSize
    EC::ErrorCode downloadBuffer(void* dst, const int64_t donwloadByteSize);

    /// Copy donwloadByteSize bytes from GPU buffer to a CPU buffer, starting from srcOffset-th byte of the GPU buffer.
    /// The CPU buffer must be preallocated and must have byte size greater or equal to donwloadByteSize
    EC::ErrorCode downloadBuffer(void* dst, const int64_t donwloadByteSize, const int64_t srcOffset);

    /// Deallocate all GPU memory allocated by the buffer and set its size to 0
    EC::ErrorCode freeMem();

    /// Get an implementation specific handle which represents the buffer on the GPU
    CUdeviceptr getHandle() const {
        return handle;
    }

    /// Get the total memory allocated for this buffer on the GPU in bytes.
    int64_t getByteSize() const {
        return byteSize;
    }
private:
    /// CUDA handle representing the buffer. If it's 0 the buffer is not allocated
    CUdeviceptr handle;
    /// Size in bytes of the total memory allocated on the GPU for this buffer
    int64_t byteSize;
};

} // namespace GPU
