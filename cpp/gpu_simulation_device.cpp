#include "gpu_simulation_device.h"
#include "matrix_math_common.cuh"
#include "error_code.h"
#include <array>
#include <memory>
#include <fstream>
namespace GPUSimulation {

#define GET_PTX_FILE_PATH(ptxFileName) PTX_SOURCE_FOLDER ptxFileName

EC::ErrorCode GPUSimulationDevice::init(int index) {
    using namespace GPU;
    RETURN_ON_ERROR_CODE(GPUDeviceBase::init(index));
#if CUDA_VERSION >= 9000
    RETURN_ON_CUDA_ERROR(cuDeviceGetAttribute(&cudaCooperativeGroupsSupported, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, deviceHandle));
#else
    cudaCooperativeGroupsSupported = 0;
#endif
    return EC::ErrorCode();
}

EC::ErrorCode GPUSimulationDevice::loadModules(const char* advectionData, const char* sparseMatrixData) {
    RETURN_ON_ERROR_CODE(loadAdvectionModule(advectionData));
    return loadSparseMatrixModule(sparseMatrixData);
}

EC::ErrorCode GPUSimulationDevice::loadAdvectionModule(const char* data) {
    const int kernelCount = int(AdvectionKernels::count);
    std::array<const char*, kernelCount> kernelsToExtract = {
        "advect"
    };

    std::array<CUfunction*, kernelCount> kernelPointers = {
        &advectionKernels[int(AdvectionKernels::advect)]
    };

    return loadModule(
        data,
        kernelsToExtract.data(),
        kernelCount,
        modules[int(Modules::advection)],
        kernelPointers.data()
    );
}

EC::ErrorCode GPUSimulationDevice::loadSparseMatrixModule(const char* data) {
    const int kernelCount = int(SparseMatrixKernels::count);
    // Warrning order must match with the one in SparseMatrixKernels enum
    std::array<const char*, kernelCount> kernelsToExtract = {
        "spRMultKernel",
        "spRMultSubKernel",
        "saxpyKernel",
        "dotProductKernel",
        "saxpbyKernel",
        "conjugateGradientMegakernel"
    };

    std::array<CUfunction*, kernelCount> kernelPointers;
    for(int i = 0; i < kernelCount; ++i) {
        kernelPointers[i] = &sparseMatrixKernels[i];
    }

    return loadModule(
        data,
        kernelsToExtract.data(),
        kernelCount,
        modules[int(Modules::sparseMatrix)],
        kernelPointers.data()
    );
}

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
        gridSize,
        blockSize,
        kernelParams
    );
    RETURN_ON_ERROR_CODE(callKernel(advectionKernels[int(AdvectionKernels::advect)], params));
    RETURN_ON_ERROR_CODE(uVelocityOutBuffer.downloadBuffer(uVelocityOut));
    RETURN_ON_ERROR_CODE(vVelocityOutBuffer.downloadBuffer(vVelocityOut));
    return EC::ErrorCode();
}

EC::ErrorCode GPUSimulationDevice::uploadMatrix(
    SimMatrix matrix,
    const SMM::TripletMatrix<float>& triplet
) {
    GPU::ScopedGPUContext ctxGuard(context);
    assert(matrix < SimMatrix::count);
    return matrices[matrix].upload(triplet);
}

EC::ErrorCode GPUSimulationDevice::spRMult(
    const SimMatrix matrix,
    const GPU::GPUBuffer& x,
    GPU::GPUBuffer& res
) {
    GPUSimulation::GPUSparseMatrix& m = matrices[matrix];
    GPU::ScopedGPUContext ctxGuard(context);
    const GPU::Dim3 blockSize(512);
    const GPU::Dim3 gridSize((m.getDenseRowCount() + blockSize.x) / blockSize.x);
    const int rowCount = m.getDenseRowCount();
    void* kernelParams[] = {
        (void*)&rowCount,
        (void*)&m.getRowStartHandle(),
        (void*)&m.getColumnIndexHandle(),
        (void*)&m.getValuesHandle(),
        (void*)&x.getHandle(),
        (void*)&res.getHandle()
    };
    GPU::KernelLaunchParams params(
        gridSize,
        blockSize,
        kernelParams
    );
    return callKernel(sparseMatrixKernels[int(SparseMatrixKernels::spRMult)], params);
}

/// Perform operation: lhs - matrix * mult (multily matrix by vector mult and subtract this from lhs)
/// @param[in] matrix Enum value from GPUSimulationDevice::SimMatrix representing the matrix which will be multiplied
/// @param[in] lhs The vector from which matrix * mult will be subtracted
/// @param[in] mult The vector which will multiply the matrix (the vector being on the right hand side of the matrix)
/// @param[in] res The vector holding the result. It can overlap with lhs if needed.
EC::ErrorCode GPUSimulationDevice::spRMultSub(
    const SimMatrix matrix,
    const GPU::GPUBuffer& lhs,
    const GPU::GPUBuffer& mult,
    GPU::GPUBuffer& res
) {
    GPUSimulation::GPUSparseMatrix& m = matrices[matrix];
    GPU::ScopedGPUContext ctxGuard(context);
    const GPU::Dim3 blockSize(512);
    const GPU::Dim3 gridSize((m.getDenseRowCount() + blockSize.x) / blockSize.x);
    const int rowCount = m.getDenseRowCount();
    void* kernelParams[] = {
        (void*)&rowCount,
        (void*)&m.getRowStartHandle(),
        (void*)&m.getColumnIndexHandle(),
        (void*)&m.getValuesHandle(),
        (void*)&lhs.getHandle(),
        (void*)&mult.getHandle(),
        (void*)&res.getHandle()
    };
    GPU::KernelLaunchParams params(
        gridSize,
        blockSize,
        kernelParams
    );
    return callKernel(sparseMatrixKernels[int(SparseMatrixKernels::spRMultSub)], params);
}

EC::ErrorCode GPUSimulationDevice::saxpy(
    const int vectorLength,
    const float a,
    const GPU::GPUBuffer& x,
    GPU::GPUBuffer& y
) {
    GPU::ScopedGPUContext ctxGuard(context);
    const GPU::Dim3 blockSize(512);
    const GPU::Dim3 gridSize((vectorLength + blockSize.x) / blockSize.x);
    void* kernelParams[] = {
        (void*)&vectorLength,
        (void*)&a,
        (void*)&x.getHandle(),
        (void*)&y.getHandle()
    };
    GPU::KernelLaunchParams params(
        gridSize,
        blockSize,
        kernelParams
    );
    return callKernel(sparseMatrixKernels[int(SparseMatrixKernels::saxpy)], params);
}

EC::ErrorCode GPUSimulationDevice::saxpby(
    const int vectorLength,
    const float a,
    const float b,
    const GPU::GPUBuffer& x,
    const GPU::GPUBuffer& y,
    GPU::GPUBuffer& result
) {
    GPU::ScopedGPUContext ctxGuard(context);
    const GPU::Dim3 blockSize(512);
    const GPU::Dim3 gridSize((vectorLength + blockSize.x) / blockSize.x);
    void* kernelParams[] = {
        (void*)&vectorLength,
        (void*)&a,
        (void*)&b,
        (void*)&x.getHandle(),
        (void*)&y.getHandle(),
        (void*)&result.getHandle()
    };
    GPU::KernelLaunchParams params(
        gridSize,
        blockSize,
        kernelParams
    );
    return callKernel(sparseMatrixKernels[int(SparseMatrixKernels::saxpby)], params);   
}

EC::ErrorCode GPUSimulationDevice::dotProduct(
    const int vectorLength,
    const GPU::GPUBuffer& a,
    const GPU::GPUBuffer& b,
    GPU::GPUBuffer& result
) {
    return dotProductInternal(
        vectorLength,
        a.getHandle(),
        b.getHandle(),
        result.getHandle()
    );
}

EC::ErrorCode GPUSimulationDevice::dotProductInternal(
    const int vectorLength,
    CUdeviceptr a,
    CUdeviceptr b,
    CUdeviceptr result
) {
    GPU::ScopedGPUContext ctxGuard(context);
    // Important must be kept in sync with the size of the shared memory in dotProduct kernel
    const GPU::Dim3 blockSize(512);
    const GPU::Dim3 gridSize((vectorLength + blockSize.x) / blockSize.x);
    void* kernelParams[] = {
        (void*)&vectorLength,
        (void*)&a,
        (void*)&b,
        (void*)&result
    };
    const int dynamicSharedMemSize = blockSize.x * sizeof(float);
    GPU::KernelLaunchParams params(
        gridSize,
        blockSize,
        dynamicSharedMemSize,
        0,
        kernelParams,
        nullptr
    );
    return callKernel(sparseMatrixKernels[int(SparseMatrixKernels::dotProduct)], params);
}

EC::ErrorCode GPUSimulationDevice::conjugateGradient(
    const SimMatrix matrix,
    const float* const b,
    const float* const x0,
    float* const xOut,
    int maxIterations,
    float eps
) {
    using namespace GPU;
    GPU::ScopedGPUContext ctxGuard(context);
    GPUSparseMatrix& a = matrices[matrix];
    // The algorithm in pseudo code is as follows:
    // 1. r_0 = b - A.x_0
    // 2. p_0 = r_0
    // 3. for j = 0, j, ... until convergence/max iteratoions
    // 4.	alpha_i = (r_j, r_j) / (A.p_j, p_j)
    // 5.	x_{j+1} = x_j + alpha_j * p_j
    // 6.	r_{j+1} = r_j - alpha_j * A.p_j
    // 7. 	beta_j = (r_{j+1}, r_{j+1}) / (r_j, r_j)
    // 8.	p_{j+1} = r_{j+1} + beta_j * p_j
    const float epsSuared = eps * eps;
    const int rows = a.getDenseRowCount();
    const int64_t byteSize = rows * sizeof(float);
    GPU::GPUBuffer x, ap, p, r;
    // TODO: Merge these two into a structure with two elements
    GPU::MappedBuffer residualNormSquared, pAp;

    RETURN_ON_ERROR_CODE(ap.init(byteSize));
    RETURN_ON_ERROR_CODE(p.init(byteSize));
    RETURN_ON_ERROR_CODE(x.init(byteSize));
    RETURN_ON_ERROR_CODE(r.init(byteSize));

    RETURN_ON_ERROR_CODE(residualNormSquared.init(sizeof(float)));
    *static_cast<float*>(residualNormSquared.getCPUAddress()) = 0;

    RETURN_ON_ERROR_CODE(pAp.init(sizeof(float)));
    *static_cast<float*>(pAp.getCPUAddress()) = 0;

    RETURN_ON_ERROR_CODE(x.uploadBuffer(x0, byteSize));
    RETURN_ON_ERROR_CODE(p.uploadBuffer(b, byteSize));

    RETURN_ON_ERROR_CODE(spRMultSub(matrix, p, x, r));
    RETURN_ON_ERROR_CODE(p.copyFromAsync(r, 0));
    RETURN_ON_ERROR_CODE(dotProductInternal(
        rows,
        r.getHandle(),
        r.getHandle(),
        residualNormSquared.getGPUAddress()
    ));
    RETURN_ON_CUDA_ERROR(cuStreamSynchronize(0));

    if(epsSuared > *static_cast<float*>(residualNormSquared.getCPUAddress())) {
        return EC::ErrorCode();
    }
    if(maxIterations == -1) {
        maxIterations = rows;
    }

    for(int i = 0; i < maxIterations; ++i) {
        *static_cast<float*>(pAp.getCPUAddress()) = 0.0f;

        RETURN_ON_ERROR_CODE(spRMult(matrix, p, ap));
        RETURN_ON_ERROR_CODE(dotProductInternal(
            rows,
            ap.getHandle(),
            p.getHandle(),
            pAp.getGPUAddress()
        ));
        RETURN_ON_CUDA_ERROR(cuStreamSynchronize(0));

        assert(*static_cast<float*>(pAp.getCPUAddress()) != 0);
        const float oldResidualNormSquared = *static_cast<float*>(residualNormSquared.getCPUAddress());
        const float alpha = oldResidualNormSquared / *static_cast<float*>(pAp.getCPUAddress());
        *static_cast<float*>(residualNormSquared.getCPUAddress()) = 0.0f;
        RETURN_ON_ERROR_CODE(saxpy(rows, alpha, p, x));
        RETURN_ON_ERROR_CODE(saxpy(rows, -alpha, ap, r));
        RETURN_ON_ERROR_CODE(dotProductInternal(
            rows,
            r.getHandle(),
            r.getHandle(),
            residualNormSquared.getGPUAddress()
        ));
        RETURN_ON_CUDA_ERROR(cuStreamSynchronize(0));
        if(epsSuared > *static_cast<float*>(residualNormSquared.getCPUAddress())) {
            return x.downloadBuffer(xOut);
        }
        const float beta = *static_cast<float*>(residualNormSquared.getCPUAddress()) / oldResidualNormSquared;
        RETURN_ON_ERROR_CODE(saxpby(rows, 1, beta, r, p, p));
    }
    return EC::ErrorCode("Max iterations reached!");
}

EC::ErrorCode GPUSimulationDevice::conjugateGradientMegaKernel(
    const SimMatrix matrix,
    const float* const b,
    const float* const x0,
    float* const xOut,
    int maxIterations,
    float eps
) {
    using namespace GPU;
    GPU::ScopedGPUContext ctxGuard(context);
    GPUSparseMatrix& a = matrices[matrix];
    // The algorithm in pseudo code is as follows:
    // 1. r_0 = b - A.x_0
    // 2. p_0 = r_0
    // 3. for j = 0, j, ... until convergence/max iteratoions
    // 4.	alpha_i = (r_j, r_j) / (A.p_j, p_j)
    // 5.	x_{j+1} = x_j + alpha_j * p_j
    // 6.	r_{j+1} = r_j - alpha_j * A.p_j
    // 7. 	beta_j = (r_{j+1}, r_{j+1}) / (r_j, r_j)
    // 8.	p_{j+1} = r_{j+1} + beta_j * p_j
    const float epsSuared = eps * eps;
    const int rows = a.getDenseRowCount();
    const int64_t byteSize = rows * sizeof(float);
    GPU::GPUBuffer x, ap, p, r,pAp, barrier, generation, newResidualNormSquared, residualNormSquared;

    RETURN_ON_ERROR_CODE(ap.init(byteSize));
    RETURN_ON_ERROR_CODE(p.init(byteSize));
    RETURN_ON_ERROR_CODE(x.init(byteSize));
    RETURN_ON_ERROR_CODE(r.init(byteSize));

    RETURN_ON_ERROR_CODE(residualNormSquared.init(sizeof(float)));
     RETURN_ON_CUDA_ERROR(cuMemsetD32Async(residualNormSquared.getHandle(), 0, 1, 0));

    RETURN_ON_ERROR_CODE(pAp.init(sizeof(float)));
    RETURN_ON_CUDA_ERROR(cuMemsetD32Async(pAp.getHandle(), 0, 1, 0));

    RETURN_ON_ERROR_CODE(barrier.init(sizeof(unsigned int)));
    RETURN_ON_CUDA_ERROR(cuMemsetD32Async(barrier.getHandle(), 0, 1, 0));

    RETURN_ON_ERROR_CODE(generation.init(sizeof(unsigned int)));
    RETURN_ON_CUDA_ERROR(cuMemsetD32Async(generation.getHandle(), 0, 1, 0));

    RETURN_ON_ERROR_CODE(newResidualNormSquared.init(sizeof(float)));
    RETURN_ON_CUDA_ERROR(cuMemsetD32Async(newResidualNormSquared.getHandle(), 0, 1, 0));

    RETURN_ON_ERROR_CODE(x.uploadBuffer(x0, byteSize));
    RETURN_ON_ERROR_CODE(p.uploadBuffer(b, byteSize));

    RETURN_ON_ERROR_CODE(spRMultSub(matrix, p, x, r));
    RETURN_ON_ERROR_CODE(p.copyFromAsync(r, 0));
    RETURN_ON_ERROR_CODE(dotProductInternal(
        rows,
        r.getHandle(),
        r.getHandle(),
        residualNormSquared.getHandle()
    ));
    RETURN_ON_CUDA_ERROR(cuStreamSynchronize(0));

    if(maxIterations == -1) {
        maxIterations = rows;
    }

    Dim3 blockSize(128);
    int numBlocksPerSm = 0;
    cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm,
        sparseMatrixKernels[int(SparseMatrixKernels::conjugateGradientMegakernel)],
        128, 128 * 4
    );
    Dim3 gridSize(std::min(deviceSMCount * numBlocksPerSm, (rows + blockSize.x) / blockSize.x));
    CGParams cgparams;
    cgparams.rowStart = (int*)a.getRowStartHandle();
    cgparams.columnIndex = (int*)a.getColumnIndexHandle();
    cgparams.values = (float*)a.getValuesHandle();
    cgparams.x = (float*)x.getHandle();
    cgparams.p = (float*)p.getHandle();
    cgparams.ap = (float*)ap.getHandle();
    cgparams.r = (float*)r.getHandle();
    cgparams.residualNormSquared = (float*)residualNormSquared.getHandle();
    cgparams.newResidualNormSquared = (float*)newResidualNormSquared.getHandle();
    cgparams.pAp = (float*)pAp.getHandle();
    cgparams.barrier = (unsigned int*)barrier.getHandle();
    cgparams.generation = (unsigned int*)generation.getHandle();
    cgparams.rows = a.getDenseRowCount();
    cgparams.maxIterations = maxIterations;
    cgparams.epsSq = epsSuared;

    void* kernelParams[] = {
        (void*)&cgparams
    };
    // Used by the dot product function
    const int dynamicSharedMemSize = blockSize.x * sizeof(float);
    GPU::KernelLaunchParams params(
        gridSize,
        blockSize,
        dynamicSharedMemSize,
        0,
        kernelParams,
        nullptr
    );
#if CUDA_VERSION >= 9000
    if (cudaCooperativeGroupsSupported) {
        RETURN_ON_CUDA_ERROR(
            cuLaunchCooperativeKernel(
                sparseMatrixKernels[int(SparseMatrixKernels::conjugateGradientMegakernel)],
                gridSize.x, gridSize.y, gridSize.z,
                blockSize.x, blockSize.y, blockSize.z,
                dynamicSharedMemSize,
                0,
                kernelParams
            ))
    } else {
        RETURN_ON_ERROR_CODE(callKernel(sparseMatrixKernels[int(SparseMatrixKernels::conjugateGradientMegakernel)], params));
    }
#else
    RETURN_ON_ERROR_CODE(callKernel(sparseMatrixKernels[int(SparseMatrixKernels::conjugateGradientMegakernel)], params));
#endif
    float newResidual;
    newResidualNormSquared.downloadBuffer(&newResidual);
    if(newResidual < epsSuared) {
        return x.downloadBuffer(xOut);
    }
    return EC::ErrorCode("Max iterations reached!");
}


GPUSimulationDeviceManager::GPUSimulationDeviceManager() : 
    GPUDeviceManagerBase<GPUSimulationDevice>()
{

}

EC::ErrorCode GPUSimulationDeviceManager::init() {
    RETURN_ON_ERROR_CODE(GPUDeviceManagerBase<GPUSimulationDevice>::initDevices());
    auto loadModuleData = [](const char* filePath, std::vector<char>& data) -> EC::ErrorCode {
        std::ifstream file(filePath, std::ifstream::ate | std::ifstream::binary);
        if(file.fail()) {
            return EC::ErrorCode(errno, "%d: %s. Cannot open file: %s.", errno, strerror(errno), filePath);
        }
        const int64_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        data.resize(fileSize + 1);
        data[fileSize] = '\0';
        file.read(data.data(), fileSize);
        return EC::ErrorCode();
    };

    const char* advectionPtxPath = GET_PTX_FILE_PATH("advection.ptx");
    const char* sparseMatrixPtxPath = GET_PTX_FILE_PATH("matrix_math.ptx");

    std::vector<char> advectionPtxData;
    std::vector<char> sparseMatrixPtxData;
    RETURN_ON_ERROR_CODE(loadModuleData(advectionPtxPath, advectionPtxData));
    RETURN_ON_ERROR_CODE(loadModuleData(sparseMatrixPtxPath, sparseMatrixPtxData));
    for(GPUSimulationDevice& device : devices) {
        RETURN_ON_ERROR_CODE(device.loadModules(advectionPtxData.data(), sparseMatrixPtxData.data()));
    }
    return EC::ErrorCode();
}

}