#include "gpu_sparse_matrix.h"
#include "error_code.h"
#include <vector>

namespace GPUSimulation {

EC::ErrorCode GPUSparseMatrix::upload(const SMM::TripletMatrix<float>& triplet) {
    const int n = triplet.getDenseRowCount();
    std::vector<int> count(n);
    rowCount = n;
    for (const auto& el : triplet) {
        count[el.getRow()]++;
    }

    std::vector<int> start(n+1);
    start[0] = 0;
    for (int i = 0; i < n; ++i) {
        start[i + 1] = start[i] + count[i];
    }

    const int nnz = triplet.getNonZeroCount();
    std::vector<int> positions(nnz);
    std::vector<float> values(nnz);
    for (const auto& el : triplet) {
        const int row = el.getRow();
        const int currentCount = count[row];
        const int position = start[row + 1] - currentCount;
        // Columns in each row are sorted in increasing order.
        assert(position == start[row] || positions[position - 1] < el.getCol());
        positions[position] = el.getCol();
        values[position] = el.getValue();
        count[row]--;
    }
		


    const int64_t rowStartByteSize = start.size() * sizeof(int);
    const int64_t columnIndexByteSize = nnz * sizeof(int);
    const int64_t valuesByteSize = nnz * sizeof(float);

    RETURN_ON_ERROR_CODE(rowStartBuffer.init(rowStartByteSize));
    RETURN_ON_ERROR_CODE(columnIndexBuffer.init(columnIndexByteSize));
    RETURN_ON_ERROR_CODE(valuesBuffer.init(valuesByteSize));

    RETURN_ON_ERROR_CODE(rowStartBuffer.uploadBuffer(start.data(), rowStartByteSize));
    RETURN_ON_ERROR_CODE(columnIndexBuffer.uploadBuffer(positions.data(), columnIndexByteSize));
    RETURN_ON_ERROR_CODE(valuesBuffer.uploadBuffer(values.data(), valuesByteSize));
    return EC::ErrorCode();
}

const CUdeviceptr& GPUSparseMatrix::getColumnIndexHandle() {
    return columnIndexBuffer.getHandle();
}

const CUdeviceptr& GPUSparseMatrix::getRowStartHandle() {
    return rowStartBuffer.getHandle();
}

const CUdeviceptr& GPUSparseMatrix::getValuesHandle() {
    return valuesBuffer.getHandle();
}

int GPUSparseMatrix::getDenseRowCount() const {
    return rowCount;
}

}