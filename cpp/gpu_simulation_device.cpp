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
    const int kernelCount = 1;
    std::array<const char*, kernelCount> kernelsToExtract = {
        "advect"
    };

    std::array<CUfunction*, kernelCount> kernelPointers = {
        &advection_kernel
    };

    return loadModule(
        data,
        kernelsToExtract.data(),
        kernelCount,
        advectionModule,
        kernelPointers.data()
    );
}

EC::ErrorCode GPUSimulationDevice::loadSparseMatrixModule(const char* data) {
    const int kernelCount = 2;
    std::array<const char*, kernelCount> kernelsToExtract = {
        "spRMult",
        "spRMultSub"
    };

    std::array<CUfunction*, kernelCount> kernelPointers = {
        &spRMult_kernel,
        &spRMultSub_kernel
    };

    return loadModule(
        data,
        kernelsToExtract.data(),
        kernelCount,
        sparseMatrixModule,
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
    RETURN_ON_ERROR_CODE(callKernelSync(advection_kernel, params));
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
    return callKernelSync(spRMult_kernel, params);
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
    return callKernelSync(spRMultSub_kernel, params);
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