#pragma once
#include <cuda.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include "error_code.h"

namespace EC {
    class ErrorCode;
}

namespace GPU {

EC::ErrorCode checkCudaError(CUresult code, const char* file, const char* function, int line);

#ifdef _MSC_VER
    #define CUDAUTILS_CPP_FUNC_NAME __FUNCSIG__
#else
    #define CUDAUTILS_CPP_FUNC_NAME __PRETTY_FUNCTION__
#endif

#define RETURN_ON_CUDA_ERROR(f) \
{ \
    EC::ErrorCode res = checkCudaError(f, __FILE__, CUDAUTILS_CPP_FUNC_NAME, __LINE__); \
    if(res.hasError()) { \
        return res; \
    } \
}

#define CHECK_CUDA_ERROR(f) checkCudaError(f, __FILE__, CUDAUTILS_CPP_FUNC_NAME, __LINE__)

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

/// Simple class to wrap around CUDA GPU device. It has functions to load modules and
/// extract functions from them. It does not store any module/kernel data only extracts
/// them from a given binary. It sotres the device handle and the device contex. 
class GPUDeviceBase {
public:
    /// Create unitialized device
    GPUDeviceBase();
    /// Initialize the device from the device list returned by the API.
    /// The device contex will not be pushed when this is called.
    /// @param index The index of the device in the list returned by the API
    EC::ErrorCode init(int index);
    /// Load a module containing device code and kernels.
    /// @param[in] moduleSource The source code for the module which will be created. All kernels must be
    /// found in this source. If not an error is returned.
    /// @param[in] kernelNames List of kernel names which will be extracted from the modules created from moduleSource.
    /// Compiles mangle the names of the functions, so it's highly possible that a kernel won't be fuound unless
    /// it's declared as extern "C" in the source
    /// @param[in] kernelCount The count of the items in kernelNames
    /// @param[out] moduleOut The compiled module will be saved here
    /// @param[out] kernelsOut Preallocated array where the extracted kernel handles will be saved. It must have at least
    /// kernelCount number of elements preallocated.
    EC::ErrorCode loadModule(
        const char* moduleSource,
        const char* kernelNames[],
        int kernelCount,
        CUmodule& moduleOut,
        CUfunction* kernelsOut
    );
    /// Launch a GPU kernel and wait for it to finish
    /// @param[in] kernel Handle to the kernel which will be launched
    /// @param[in] kernelParams The parameters for the kernel 
    EC::ErrorCode callKernelSync(CUfunction kernel, const KernelLaunchParams& kernelParams);
    ~GPUDeviceBase();
protected:
    void printDeviceInfo() const;
    CUdevice deviceHandle;
    CUcontext context;
};

template<typename Device>
class GPUDeviceManagerBase {
public:
    GPUDeviceManagerBase() = default;
    GPUDeviceManagerBase(const GPUDeviceManagerBase&) = delete;
    GPUDeviceManagerBase& operator=(const GPUDeviceManagerBase&) = delete;
    EC::ErrorCode init();
    EC::ErrorCode addModuleFromFile(
        const char* filePath,
        const char* kernelNames[],
        int kernelCount,
        CUmodule modules[],
        CUfunction* kernels[]
    );
    Device& getDevice(int index);
    int getDeviceCount() const;
private:
    std::vector<Device> devices;
};

template<typename Device>
inline EC::ErrorCode GPUDeviceManagerBase<Device>::init() {
    RETURN_ON_CUDA_ERROR(cuInit(0));

    int deviceCount = 0;
    RETURN_ON_CUDA_ERROR(cuDeviceGetCount(&deviceCount));
    if(deviceCount == 0) {
        return EC::ErrorCode(-1, "Cannot find CUDA capable devices!");
    }
    printf("CUDA Devices found: %d\n", deviceCount);
    devices.resize(deviceCount);
    for(int i = 0; i < deviceCount; ++i) {
        const EC::ErrorCode ec = devices[i].init(i);
        if(ec.hasError()) {
            return ec;
        }
    }
    return EC::ErrorCode();
}

/// Base GPU device manager class, which stores devices of a specific concrete type.
/// It can initialize all initializes listed by the API and load modules and kernels
/// from sources for all devices.
/// @tparam Device Concrete type of GPU devices which will be stored in the manager
template<typename Device>
inline EC::ErrorCode GPUDeviceManagerBase<Device>::addModuleFromFile(
    const char* filepath,
    const char* kernelNames[],
    int kernelCount,
    CUmodule modules[],
    CUfunction* kernels[]
) {
    std::ifstream file(filepath, std::ifstream::ate | std::ifstream::binary);
    if(file.fail()) {
        return EC::ErrorCode(errno, "%d: %s. Cannot open file: %s.", errno, strerror(errno), filepath);
    }
    const int64_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> data(new char[fileSize + 1]);
    data[fileSize] = '\0';
    file.read(data.get(), fileSize);
    for(int i = 0; i < devices.count(); ++i) {
        const EC::ErrorCode ec = devices[i].loadModule(
            data.get(),
            kernelNames,
            kernelCount,
            modules[i],
            kernels[i]
        );
        if(ec.hasError()) {
            return ec;
        }
    }
    return EC::ErrorCode();
}

template<typename Device>
inline Device& GPUDeviceManagerBase<Device>::getDevice(int index) {
    return devices[index];
}

template<typename Device>
inline int GPUDeviceManagerBase<Device>::getDeviceCount() const {
    return devices.size();
}


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
