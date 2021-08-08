#include "gpu_simulation_device.h"
#include "error_code.h"
#include <array>
#include <memory>
#include <fstream>
namespace GPUSimulation {

#define GET_PTX_FILE_PATH(ptxFileName) PTX_SOURCE_FOLDER ptxFileName

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
        "spRMult",
        "spRMultSub",
        "saxpy",
        "dotProduct"
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
        blockSize,
        gridSize,
        kernelParams
    );
    RETURN_ON_ERROR_CODE(callKernelSync(advectionKernels[int(AdvectionKernels::advect)], params));
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
        blockSize,
        gridSize,
        kernelParams
    );
    return callKernelSync(sparseMatrixKernels[int(SparseMatrixKernels::spRMult)], params);
}

/// Perform operation: lhs - matrix * mult (multily matrix by vector mult and subtract this from lhs)
/// @param[in] matrix Enum value from GPUSimulationDevice::SimMatrix representing the matrix which will be multiplied
/// @param[in] mult The vector which will multiply the matrix (the vector being on the right hand side of the matrix)
/// @param[in] lhs The vector from which matrix * mult will be subtracted
/// @param[in] res The vector holding the result. It can overlap with lhs if needed.
EC::ErrorCode GPUSimulationDevice::spRMultSub(
    const SimMatrix matrix,
    const GPU::GPUBuffer& mult,
    const GPU::GPUBuffer& lhs,
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
        blockSize,
        gridSize,
        kernelParams
    );
    return callKernelSync(sparseMatrixKernels[int(SparseMatrixKernels::spRMultSub)], params);
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
        blockSize,
        gridSize,
        kernelParams
    );
    return callKernelSync(sparseMatrixKernels[int(SparseMatrixKernels::saxpy)], params);
}

EC::ErrorCode GPUSimulationDevice::conjugateGradient(
    const SimMatrix matrix,
    const float* const b,
    const float* const x0,
    float* const x,
    int maxIterations,
    float eps
) {
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
    const int rows = a.getDenseRowCount();
    const int64_t byteSize = rows * sizeof(float);
    GPU::GPUBuffer pDev, apDev, bDev;

    RETURN_ON_ERROR_CODE(pDev.init(byteSize));
    RETURN_ON_ERROR_CODE(apDev.init(byteSize));
    RETURN_ON_ERROR_CODE(bDev.init(byteSize));

    RETURN_ON_ERROR_CODE(apDev.uploadBuffer(x0, byteSize));

    RETURN_ON_ERROR_CODE(bDev.uploadBuffer(b, byteSize));

    const float epsSuared = eps * eps;
    SMM::Vector<float> r(rows, 0);
    RETURN_ON_ERROR_CODE(spRMultSub(matrix, apDev, bDev, pDev));
    RETURN_ON_ERROR_CODE(pDev.downloadBuffer(r.begin()));
    // a.rMultSub(b, x0, r);

    SMM::Vector<float> p(rows), Ap(rows, 0);
    float residualNormSquared = 0;
    for(int i = 0; i < rows; ++i) {
        p[i] = r[i];
        residualNormSquared += r[i] * r[i];
    }
    if(epsSuared > residualNormSquared) {
        return 0;
    }
    if(maxIterations == -1) {
        maxIterations = rows;
    }
    // We have initial condition different than the output vector on the first iteration when we compute
    // x = x + alpha * p, we must have the x on the right hand side to be the initial condition x. And on all
    // next iterations it must be the output vector.
    const float* currentX = x0;
    for(int i = 0; i < maxIterations; ++i) {
        RETURN_ON_ERROR_CODE(spRMult(matrix, pDev, apDev));
        RETURN_ON_ERROR_CODE(apDev.downloadBuffer(Ap.begin()));
        //a.rMult(p, Ap);
        const float pAp = Ap * p;
        // If the denominator is 0 we have a lucky breakdown. The residual at the previous step must be 0.
        assert(pAp != 0);
        // alpha = (r_i, r_i) / (Ap, p)
        const float alpha = residualNormSquared / pAp;
        // x = x + alpha * p
        // r = r - alpha * Ap
        float newResidualNormSquared = 0;
        for(int j = 0; j < rows; ++j) {
            x[j] = alpha * p[j] + currentX[j];
            r[j] = -alpha * Ap[j] + r[j];
            newResidualNormSquared += r[j] * r[j];
        }
        if(epsSuared > newResidualNormSquared) {
            return EC::ErrorCode();
        }
        // beta = (r_{i+1}, r_(i+1)) / (r_i, r_i)
        const float beta = newResidualNormSquared / residualNormSquared;
        residualNormSquared = newResidualNormSquared;
        // p = r + beta * p
        for(int j = 0; j < rows; ++j) {
            p[j] = beta * p[j] + r[j];
        }
        RETURN_ON_ERROR_CODE(pDev.uploadBuffer(p.begin(), byteSize));
        currentX = x;
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