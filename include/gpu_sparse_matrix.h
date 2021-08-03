#pragma once
#include "gpu_host_common.h"
#include "sparse_matrix_math.h"
namespace EC {
    class ErrorCode;
};

namespace GPUSimulation {
class GPUSparseMatrix {
public:

    EC::ErrorCode upload(const SMM::TripletMatrix<float>& triplet);
    const CUdeviceptr& getRowStartHandle();
    const CUdeviceptr& getColumnIndexHandle();
    const CUdeviceptr& getValuesHandle(); 
    int getDenseRowCount() const;
private:
    GPU::GPUBuffer rowStartBuffer;
    GPU::GPUBuffer columnIndexBuffer;
    GPU::GPUBuffer valuesBuffer;
    int rowCount;
};
}
