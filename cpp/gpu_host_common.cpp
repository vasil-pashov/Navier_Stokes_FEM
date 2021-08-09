#include "gpu_host_common.h"
#include "error_code.h"
#include <fstream>
#include <memory>
#include <string>
#include <cassert>
#include <string>

namespace GPU {

EC::ErrorCode checkCudaError(CUresult code, const char* file, const char* function, int line) {
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

GPUDeviceBase::GPUDeviceBase() :
    deviceHandle(0),
    context(nullptr)
{}

EC::ErrorCode GPUDeviceBase::init(int index) {
    RETURN_ON_CUDA_ERROR(cuDeviceGet(&deviceHandle, index));
    const int deviceNameLen = 256;
    char name[deviceNameLen];
    RETURN_ON_CUDA_ERROR(cuDeviceGetName(name, deviceNameLen, deviceHandle));
    printf("Initializing device: %s\n", name);
    printDeviceInfo();
    RETURN_ON_CUDA_ERROR(cuDevicePrimaryCtxRetain(&context, deviceHandle));
    return EC::ErrorCode();
}

EC::ErrorCode GPUDeviceBase::loadModule(
    const char* moduleSource,
    const char* kernelNames[],
    int kernelCount,
    CUmodule& out,
    CUfunction** kernels
) {
    const int complierOptionsCount = 4;
    CUjit_option compilerOptions[complierOptionsCount] = {
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER
    };
    const int logBufferSize = 4096;
    char jitLogBuffer[logBufferSize];
    char jitErrorBuffer[logBufferSize];
    void* compilerOptionsValue[complierOptionsCount] = {
        (void*)logBufferSize,
        (void*)jitLogBuffer,
        (void*)logBufferSize,
        (void*)jitErrorBuffer
    };

    ScopedGPUContext ctxGuard(context);
    CUresult res = cuModuleLoadDataEx(&out, moduleSource, complierOptionsCount, compilerOptions, compilerOptionsValue);
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
    
    for(int i = 0; i < kernelCount; ++i) {
        RETURN_ON_CUDA_ERROR(cuModuleGetFunction(kernels[i], out, kernelNames[i]));
    }
    return EC::ErrorCode();
}

EC::ErrorCode GPUDeviceBase::callKernel(CUfunction kernel, const KernelLaunchParams& launchParams) {
    RETURN_ON_CUDA_ERROR(cuLaunchKernel(
        kernel,
        launchParams.gridSize.x, launchParams.gridSize.y, launchParams.gridSize.z,
        launchParams.blockSize.x, launchParams.blockSize.y, launchParams.blockSize.z,
        launchParams.sharedMemSize,
        launchParams.stream,
        launchParams.kernelParams,
        launchParams.extra
    ));
    return EC::ErrorCode();
}

GPUDeviceBase::~GPUDeviceBase() {
    const EC::ErrorCode error = CHECK_CUDA_ERROR(cuDevicePrimaryCtxRelease(deviceHandle));
    if(error.hasError()) {
        fprintf(stderr, "[Error] %s\n", error.getMessage());
    }
}

void GPUDeviceBase::printDeviceInfo() const {
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, deviceHandle); //= 1,              /**< Maximum number of threads per block */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, deviceHandle); // = 2,                    /**< Maximum block dimension X */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, deviceHandle); // = 3,                    /**< Maximum block dimension Y */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, deviceHandle); // = 4,                    /**< Maximum block dimension Z */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X , deviceHandle); //= 5,                     /**< Maximum grid dimension X */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y , deviceHandle); //= 6,                     /**< Maximum grid dimension Y */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z , deviceHandle); //= 7,                     /**< Maximum grid dimension Z */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, deviceHandle); // = 8,        /**< Maximum shared memory available per block in bytes */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, deviceHandle); // = 8,            /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, deviceHandle); // = 9,              /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_WARP_SIZE, deviceHandle); // = 10,                  /**< Warp size in threads */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_PITCH, deviceHandle); // = 11,                /**< Maximum pitch in bytes allowed by memory copies */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, deviceHandle); // = 12,           /**< Maximum number of 32-bit registers available per block */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, deviceHandle); // = 12,               /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, deviceHandle); // = 13,                        /**< Typical clock frequency in kilohertz */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, deviceHandle); // = 14,                 /**< Alignment requirement for textures */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, deviceHandle); // = 15,                       /**< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, deviceHandle); // = 16,              /**< Number of multiprocessors on device */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT , deviceHandle); //= 17,               /**< Specifies whether there is a run time limit on kernels */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_INTEGRATED, deviceHandle); // = 18,                        /**< Device is integrated with host memory */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, deviceHandle); // = 19,               /**< Device can map host memory into CUDA address space */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, deviceHandle); // = 20,                      /**< Compute mode (See ::CUcomputemode for details) */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH , deviceHandle); //= 21,           /**< Maximum 1D texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH , deviceHandle); //= 22,           /**< Maximum 2D texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, deviceHandle); // = 23,          /**< Maximum 2D texture height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH , deviceHandle); //= 24,           /**< Maximum 3D texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, deviceHandle); // = 25,          /**< Maximum 3D texture height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH , deviceHandle); //= 26,           /**< Maximum 3D texture depth */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH , deviceHandle); //= 27,   /**< Maximum 2D layered texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, deviceHandle); // = 28,  /**< Maximum 2D layered texture height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, deviceHandle); // = 29,  /**< Maximum layers in a 2D layered texture */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH , deviceHandle); //= 27,     /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT, deviceHandle); // = 28,    /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES, deviceHandle); // = 29, /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, deviceHandle); // = 30,                 /**< Alignment requirement for surfaces */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, deviceHandle); // = 31,                /**< Device can possibly execute multiple kernels concurrently */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_ECC_ENABLED, deviceHandle); // = 32,                       /**< Device has ECC support enabled */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID , deviceHandle); //= 33,                        /**< PCI bus ID of the device */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, deviceHandle); // = 34,                     /**< PCI device ID of the device */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_TCC_DRIVER, deviceHandle); // = 35,                        /**< Device is using TCC driver model */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, deviceHandle); // = 36,                 /**< Peak memory clock frequency in kilohertz */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, deviceHandle); // = 37,           /**< Global memory bus width in bits */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, deviceHandle); // = 38,                     /**< Size of L2 cache in bytes */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, deviceHandle); // = 39,    /**< Maximum resident threads per multiprocessor */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, deviceHandle); // = 40,                /**< Number of asynchronous engines */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, deviceHandle); // = 41,                /**< Device shares a unified address space with the host */    
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, deviceHandle); // = 42,   /**< Maximum 1D layered texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, deviceHandle); // = 43,  /**< Maximum layers in a 1D layered texture */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER, deviceHandle); // = 44,                  /**< Deprecated, do not use. */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH, deviceHandle); // = 45,    /**< Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT, deviceHandle); // = 46,   /**< Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE, deviceHandle); // = 47, /**< Alternate maximum 3D texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE, deviceHandle); // = 48,/**< Alternate maximum 3D texture height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE, deviceHandle); // = 49, /**< Alternate maximum 3D texture depth */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, deviceHandle); // = 50,                     /**< PCI domain ID of the device */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, deviceHandle); // = 51,           /**< Pitch alignment requirement for textures */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, deviceHandle); // = 52,      /**< Maximum cubemap texture width/height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH , deviceHandle); //= 53,  /**< Maximum cubemap layered texture width/height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, deviceHandle); // = 54, /**< Maximum layers in a cubemap layered texture */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH , deviceHandle); //= 55,           /**< Maximum 1D surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH , deviceHandle); //= 56,           /**< Maximum 2D surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT, deviceHandle); // = 57,          /**< Maximum 2D surface height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH , deviceHandle); //= 58,           /**< Maximum 3D surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT, deviceHandle); // = 59,          /**< Maximum 3D surface height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH , deviceHandle); //= 60,           /**< Maximum 3D surface depth */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH , deviceHandle); //= 61,   /**< Maximum 1D layered surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS, deviceHandle); // = 62,  /**< Maximum layers in a 1D layered surface */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH , deviceHandle); //= 63,   /**< Maximum 2D layered surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT, deviceHandle); // = 64,  /**< Maximum 2D layered surface height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS, deviceHandle); // = 65,  /**< Maximum layers in a 2D layered surface */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH, deviceHandle); // = 66,      /**< Maximum cubemap surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH , deviceHandle); //= 67,  /**< Maximum cubemap layered surface width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, deviceHandle); // = 68, /**< Maximum layers in a cubemap layered surface */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH , deviceHandle); //= 69,    /**< Maximum 1D linear texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH , deviceHandle); //= 70,    /**< Maximum 2D linear texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, deviceHandle); // = 71,   /**< Maximum 2D linear texture height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH , deviceHandle); //= 72,    /**< Maximum 2D linear texture pitch in bytes */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, deviceHandle); // = 73, /**< Maximum mipmapped 2D texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, deviceHandle); // = 74,/**< Maximum mipmapped 2D texture height */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, deviceHandle); // = 75,          /**< Major compute capability version number */     
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, deviceHandle); // = 76,          /**< Minor compute capability version number */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, deviceHandle); // = 77, /**< Maximum mipmapped 1D texture width */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, deviceHandle); // = 78,       /**< Device supports stream priorities */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, deviceHandle); // = 79,         /**< Device supports caching globals in L1 */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED , deviceHandle); //= 80,          /**< Device supports caching locals in L1 */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, deviceHandle); // = 81,  /**< Maximum shared memory available per multiprocessor in bytes */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, deviceHandle); // = 82,  /**< Maximum number of 32-bit registers available per multiprocessor */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY , deviceHandle); //= 83,                    /**< Device can allocate managed memory on this system */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, deviceHandle); // = 84,                    /**< Device is on a multi-GPU board */ 
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, deviceHandle); // = 85,           /**< Unique id for a group of devices on the same multi-GPU board */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, deviceHandle); // = 86,       /**< Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)*/
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, deviceHandle); // = 87,  /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, deviceHandle); // = 88,            /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, deviceHandle); // = 89,         /**< Device can coherently access managed memory concurrently with the CPU */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, deviceHandle); // = 90,      /**< Device supports compute preemption. */
    CUDAUTILS_PRINT_DEVICE_ATTRIBUTE(CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, deviceHandle); // = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
}

GPUBuffer::GPUBuffer(int64_t byteSize) :
    byteSize(byteSize)
{
    EC::ErrorCode ec = CHECK_CUDA_ERROR(cuMemAlloc(&handle, byteSize));
    if(ec.hasError()) {
        assert(false);
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
    this->byteSize = byteSize;
    return EC::ErrorCode();
}

EC::ErrorCode GPUBuffer::uploadBuffer(const void* src, const int64_t uploadByteSize, const int64_t destOffset) {
    assert(destOffset + uploadByteSize <= byteSize && "Trying to use more space than there is allocated");
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

EC::ErrorCode GPUBuffer::copyFrom(const GPUBuffer& source) {
    assert(source.byteSize == byteSize);
    return CHECK_CUDA_ERROR(cuMemcpyDtoD(handle, source.handle, byteSize));
}

EC::ErrorCode GPUBuffer::copyFromAsync(const GPUBuffer& source, CUstream stream) {
    assert(source.byteSize == byteSize);
    return CHECK_CUDA_ERROR(cuMemcpyDtoDAsync(handle, source.handle, byteSize, stream));
}


EC::ErrorCode GPUBuffer::freeMem() {
    EC::ErrorCode ec = CHECK_CUDA_ERROR(cuMemFree(handle));
    handle = 0;
    byteSize = 0;
    return ec;
}

CPUPinnedBuffer::CPUPinnedBuffer() :
    data(nullptr),
    byteSize(0)
{

}

CPUPinnedBuffer::CPUPinnedBuffer(CPUPinnedBuffer&& other) :
    data(other.data),
    byteSize(other.byteSize)
{
    other.byteSize = 0;
    other.data = nullptr;
}

CPUPinnedBuffer& CPUPinnedBuffer::operator=(CPUPinnedBuffer&& other) {
    freeMem();
    data = other.data;
    byteSize = other.byteSize;
    return *this;
}

CPUPinnedBuffer::~CPUPinnedBuffer() {
    freeMem();
}

EC::ErrorCode CPUPinnedBuffer::init(const int64_t byteSize) {
    RETURN_ON_CUDA_ERROR(cuMemAllocHost(&data, byteSize));
    this->byteSize = byteSize;
    return EC::ErrorCode();
}

void* CPUPinnedBuffer::getData() {
    return data;
}

void CPUPinnedBuffer::freeMem() {
    [[maybe_unused]]CUresult res = cuMemFreeHost(data);
    assert(res == CUDA_SUCCESS);
    data = nullptr;
    byteSize = 0;
}

int64_t CPUPinnedBuffer::getByteSize() const {
    return byteSize;
}

}