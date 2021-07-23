#include "gpu_context.h"
#include "error_code.h"
#include <fstream>
#include <memory>
#include <string>
#include <cassert>
#include <string>

namespace GPU {

struct ScopedGPUContext {
    explicit ScopedGPUContext(CUcontext ctx) : ctx(ctx) {
        cuCtxPushCurrent(ctx);
    }

    ScopedGPUContext(const ScopedGPUContext&) = delete;
    ScopedGPUContext& operator=(const ScopedGPUContext&) = delete;

    ScopedGPUContext(ScopedGPUContext&& other) = delete;
    ScopedGPUContext& operator=(ScopedGPUContext&&) = delete;

    ~ScopedGPUContext() {
        CUcontext popped;
        cuCtxPopCurrent(&popped);
        assert(ctx == popped);
    }
private:
    CUcontext ctx;
};

inline EC::ErrorCode checkCudaError(CUresult code, const char* file, const char* function, int line) {
    if(code != CUDA_SUCCESS) {
        EC::ErrorCode result;
        const char* errorName = nullptr;
        CUresult status = cuGetErrorName(code, &errorName);
        if(status == CUDA_SUCCESS) {
            const char* errorString = nullptr;
            status = cuGetErrorString(code, &errorString);
            if(status == CUDA_SUCCESS) {
                result = EC::ErrorCode(
                    code,
                    "CUDA error %d (%s). %s. File: %s, function: %s, line: %d",
                    code,
                    errorName,
                    errorString,
                    file,
                    function,
                    line);
            } else {
                result = EC::ErrorCode(
                    code,
                    "CUDA error %d (%s). File: %s, function: %s, line: %d",
                    code,
                    errorName,
                    file,
                    function,
                    line
                );
            }
        } else {
            result = EC::ErrorCode(
                code,
                "CUDA error: %d. File: %s, function: %s, line: %d",
                code,
                file,
                function,
                line
            );
        }
        assert(false);
        return result;
    }
    return EC::ErrorCode();
}

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

#define CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(attribute, device) \
    {\
        int result = 0; \
        CUresult status = cuDeviceGetAttribute(&result, attribute, device); \
        if(status == CUDA_SUCCESS) { \
            printf("%s: %d\n", #attribute, result); \
        } else { \
            printf("Could not query attribute: %s\n", #attribute); \
        } \
    }

EC::ErrorCode GPUDevice::init(int index) {
    RETURN_ON_CUDA_ERROR(cuDeviceGet(&handle, index));
    const int deviceNameLen = 256;
    char name[deviceNameLen];
    RETURN_ON_CUDA_ERROR(cuDeviceGetName(name, deviceNameLen, handle));
    printf("Initializing device: %s\n", name);
    printDeviceInfo();
    RETURN_ON_CUDA_ERROR(cuDevicePrimaryCtxRetain(&context, handle));
    return EC::ErrorCode();
}

EC::ErrorCode GPUDevice::addModule(const char* src, const char* kernelNames[], int kernelCount, CUmodule* out) {
    CUjit_option compilerOptions[] = {
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER
    };
    const int logBufferSize = 4096;
    char jitLogBuffer[logBufferSize];
    char jitErrorBuffer[logBufferSize];
    void* compilerOptionsValue[] = {
        (void*)logBufferSize,
        (void*)jitLogBuffer,
        (void*)logBufferSize,
        (void*)jitErrorBuffer
    };
    static_assert(
        std::size(compilerOptions) == std::size(compilerOptionsValue),
        "There must be one-to-one matching between compiler options and values"
    );

    ScopedGPUContext ctxGuard(context);
    CUmodule module;
    CUresult res = cuModuleLoadDataEx(&module, src, std::size(compilerOptions), compilerOptions, compilerOptionsValue);
    if(res != CUDA_SUCCESS) {
        const char* errorName = nullptr;
        cuGetErrorName(res, &errorName);
        const EC::ErrorCode err = EC::ErrorCode(
            res,
            "CUDA error: %d (%s). CUDA module failed to compile.\nInfo log: %s.\nError log: %s",
            res,
            errorName,
            jitLogBuffer,
            jitErrorBuffer
        );
        assert(false && "Cuda module failed to compile/link");
        return err;
    }
    modules.push_back(module);
    if(out) {
        *out = module;
    }
    for(int i = 0; i < kernelCount; ++i) {
        CUfunction kernel = nullptr;
        RETURN_ON_CUDA_ERROR(cuModuleGetFunction(&kernel, module, kernelNames[i]));
        const auto insertRes = kernels.insert_or_assign(kernelNames[i], kernel);
        if(insertRes.second == false) {
            return EC::ErrorCode(-1, "Kernel: %s already exists", kernelNames[i]);
        }
    }
    return EC::ErrorCode();
}

EC::ErrorCode GPUDevice::callKernelSync(const std::string& name, const KernelLaunchParams& launchParams) {
    auto it = kernels.find(name);
    if(it == kernels.end()) {
        return EC::ErrorCode(-1, "Kernel: %s does not exist or was not loaded \n");
    }
    RETURN_ON_CUDA_ERROR(cuLaunchKernel(
        it->second,
        launchParams.gridSize.x, launchParams.gridSize.y, launchParams.gridSize.z,
        launchParams.blockSize.x, launchParams.blockSize.y, launchParams.blockSize.z,
        launchParams.sharedMemSize,
        launchParams.stream,
        launchParams.kernelParams,
        launchParams.extra
    ));
    return EC::ErrorCode();
}

GPUDevice::~GPUDevice() {
    const EC::ErrorCode error = CHECK_CUDA_ERROR(cuDevicePrimaryCtxRelease(handle));
    if(error.hasError()) {
        fprintf(stderr, "[Error] %s\n", error.getMessage());
    }
}

EC::ErrorCode GPUDeviceManager::init() {
    RETURN_ON_CUDA_ERROR(cuInit(0));

    int deviceCount = 0;
    RETURN_ON_CUDA_ERROR(cuDeviceGetCount(&deviceCount));
    if(deviceCount == 0) {
        return EC::ErrorCode(-1, "Cannot find CUDA capable devices!");
    }
    printf("CUDA Devices found: %d\n", deviceCount);
    devices.resize(deviceCount);
    for(int i = 0; i < deviceCount; ++i) {
        devices[i].init(i);
    }
    return EC::ErrorCode();
}

EC::ErrorCode GPUDeviceManager::addModuleFromFile(const char* filepath, const char* kernelNames[], int kernelCount) {
    std::ifstream file(filepath, std::ifstream::ate | std::ifstream::binary);
    if(file.fail()) {
        return EC::ErrorCode(errno, "%d: %s. Cannot open file: %s.", errno, strerror(errno), filepath);
    }
    const int64_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::string data;
    data.resize(fileSize);
    file.read(data.data(), fileSize);
    for(GPUDevice& device : devices) {
        const EC::ErrorCode errorCode = device.addModule(data.c_str(), kernelNames, kernelCount);
        if(errorCode.hasError()) {
            return errorCode;
        }
    }
    return EC::ErrorCode();
}

void GPUDevice::printDeviceInfo() const {
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, handle); //= 1,              /**< Maximum number of threads per block */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, handle); // = 2,                    /**< Maximum block dimension X */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, handle); // = 3,                    /**< Maximum block dimension Y */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, handle); // = 4,                    /**< Maximum block dimension Z */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X , handle); //= 5,                     /**< Maximum grid dimension X */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y , handle); //= 6,                     /**< Maximum grid dimension Y */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z , handle); //= 7,                     /**< Maximum grid dimension Z */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, handle); // = 8,        /**< Maximum shared memory available per block in bytes */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, handle); // = 8,            /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, handle); // = 9,              /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_WARP_SIZE, handle); // = 10,                  /**< Warp size in threads */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_PITCH, handle); // = 11,                /**< Maximum pitch in bytes allowed by memory copies */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, handle); // = 12,           /**< Maximum number of 32-bit registers available per block */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, handle); // = 12,               /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, handle); // = 13,                        /**< Typical clock frequency in kilohertz */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, handle); // = 14,                 /**< Alignment requirement for textures */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, handle); // = 15,                       /**< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, handle); // = 16,              /**< Number of multiprocessors on device */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT , handle); //= 17,               /**< Specifies whether there is a run time limit on kernels */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_INTEGRATED, handle); // = 18,                        /**< Device is integrated with host memory */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, handle); // = 19,               /**< Device can map host memory into CUDA address space */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, handle); // = 20,                      /**< Compute mode (See ::CUcomputemode for details) */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH , handle); //= 21,           /**< Maximum 1D texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH , handle); //= 22,           /**< Maximum 2D texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, handle); // = 23,          /**< Maximum 2D texture height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH , handle); //= 24,           /**< Maximum 3D texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, handle); // = 25,          /**< Maximum 3D texture height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH , handle); //= 26,           /**< Maximum 3D texture depth */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH , handle); //= 27,   /**< Maximum 2D layered texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, handle); // = 28,  /**< Maximum 2D layered texture height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, handle); // = 29,  /**< Maximum layers in a 2D layered texture */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH , handle); //= 27,     /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT, handle); // = 28,    /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES, handle); // = 29, /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, handle); // = 30,                 /**< Alignment requirement for surfaces */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, handle); // = 31,                /**< Device can possibly execute multiple kernels concurrently */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_ECC_ENABLED, handle); // = 32,                       /**< Device has ECC support enabled */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID , handle); //= 33,                        /**< PCI bus ID of the device */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, handle); // = 34,                     /**< PCI device ID of the device */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_TCC_DRIVER, handle); // = 35,                        /**< Device is using TCC driver model */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, handle); // = 36,                 /**< Peak memory clock frequency in kilohertz */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, handle); // = 37,           /**< Global memory bus width in bits */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, handle); // = 38,                     /**< Size of L2 cache in bytes */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, handle); // = 39,    /**< Maximum resident threads per multiprocessor */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, handle); // = 40,                /**< Number of asynchronous engines */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, handle); // = 41,                /**< Device shares a unified address space with the host */    
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, handle); // = 42,   /**< Maximum 1D layered texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, handle); // = 43,  /**< Maximum layers in a 1D layered texture */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER, handle); // = 44,                  /**< Deprecated, do not use. */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH, handle); // = 45,    /**< Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT, handle); // = 46,   /**< Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE, handle); // = 47, /**< Alternate maximum 3D texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE, handle); // = 48,/**< Alternate maximum 3D texture height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE, handle); // = 49, /**< Alternate maximum 3D texture depth */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, handle); // = 50,                     /**< PCI domain ID of the device */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, handle); // = 51,           /**< Pitch alignment requirement for textures */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, handle); // = 52,      /**< Maximum cubemap texture width/height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH , handle); //= 53,  /**< Maximum cubemap layered texture width/height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, handle); // = 54, /**< Maximum layers in a cubemap layered texture */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH , handle); //= 55,           /**< Maximum 1D surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH , handle); //= 56,           /**< Maximum 2D surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT, handle); // = 57,          /**< Maximum 2D surface height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH , handle); //= 58,           /**< Maximum 3D surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT, handle); // = 59,          /**< Maximum 3D surface height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH , handle); //= 60,           /**< Maximum 3D surface depth */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH , handle); //= 61,   /**< Maximum 1D layered surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS, handle); // = 62,  /**< Maximum layers in a 1D layered surface */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH , handle); //= 63,   /**< Maximum 2D layered surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT, handle); // = 64,  /**< Maximum 2D layered surface height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS, handle); // = 65,  /**< Maximum layers in a 2D layered surface */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH, handle); // = 66,      /**< Maximum cubemap surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH , handle); //= 67,  /**< Maximum cubemap layered surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, handle); // = 68, /**< Maximum layers in a cubemap layered surface */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH , handle); //= 69,    /**< Maximum 1D linear texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH , handle); //= 70,    /**< Maximum 2D linear texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, handle); // = 71,   /**< Maximum 2D linear texture height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH , handle); //= 72,    /**< Maximum 2D linear texture pitch in bytes */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, handle); // = 73, /**< Maximum mipmapped 2D texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, handle); // = 74,/**< Maximum mipmapped 2D texture height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, handle); // = 75,          /**< Major compute capability version number */     
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, handle); // = 76,          /**< Minor compute capability version number */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, handle); // = 77, /**< Maximum mipmapped 1D texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, handle); // = 78,       /**< Device supports stream priorities */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, handle); // = 79,         /**< Device supports caching globals in L1 */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED , handle); //= 80,          /**< Device supports caching locals in L1 */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, handle); // = 81,  /**< Maximum shared memory available per multiprocessor in bytes */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, handle); // = 82,  /**< Maximum number of 32-bit registers available per multiprocessor */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY , handle); //= 83,                    /**< Device can allocate managed memory on this system */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, handle); // = 84,                    /**< Device is on a multi-GPU board */ 
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, handle); // = 85,           /**< Unique id for a group of devices on the same multi-GPU board */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, handle); // = 86,       /**< Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)*/
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, handle); // = 87,  /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, handle); // = 88,            /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, handle); // = 89,         /**< Device can coherently access managed memory concurrently with the CPU */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, handle); // = 90,      /**< Device supports compute preemption. */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, handle); // = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
}

GPUBuffer::GPUBuffer(int64_t byteSize) :
    byteSize(byteSize)
{
    EC::ErrorCode ec = CHECK_CUDA_ERROR(cuMemAlloc(&handle, byteSize));
    if(ec.hasError()) {
        fprintf(stderr, "%s\n", ec.getMessage());
        handle = 0;
        byteSize = 0;
    }
}

GPUBuffer::~GPUBuffer() {
    freeMem();
}

EC::ErrorCode GPUBuffer::init(int64_t byteSize) {
    if(handle != 0) {
        const EC::ErrorCode ec = freeMem();
        if(ec.hasError()) {
            return ec;
        }
    }
    RETURN_ON_CUDA_ERROR(cuMemAlloc(&handle, byteSize));
    return EC::ErrorCode();
}

EC::ErrorCode GPUBuffer::uploadBuffer(const void* src, const int64_t uploadByteSize, const int64_t destOffset) {
    assert(destOffset + uploadByteSize < byteSize && "Trying to use more space than there is allocated");
    RETURN_ON_CUDA_ERROR(cuMemcpyHtoD((CUdeviceptr)((char*)handle + destOffset), src, uploadByteSize));
    return EC::ErrorCode();
}

EC::ErrorCode GPUBuffer::uploadBuffer(const void* src, const int64_t uploadByteSize) {
    return uploadBuffer(src, uploadByteSize, 0);
}

EC::ErrorCode GPUBuffer::downloadBuffer(void* src) {
    return downloadBuffer(src, byteSize, 0);
}

EC::ErrorCode GPUBuffer::downloadBuffer(void* src, int64_t donwloadByteSize) {
    return downloadBuffer(src, donwloadByteSize, 0);
}

EC::ErrorCode GPUBuffer::downloadBuffer(void* src, int64_t downloadByteSize, const int64_t srcOffset) {
    RETURN_ON_CUDA_ERROR(cuMemcpyDtoH(src, (CUdeviceptr)((char*)handle + srcOffset), downloadByteSize));
    return EC::ErrorCode();
}


EC::ErrorCode GPUBuffer::freeMem() {
    EC::ErrorCode ec = CHECK_CUDA_ERROR(cuMemFree(handle));
    handle = 0;
    byteSize = 0;
    return ec;
}

}