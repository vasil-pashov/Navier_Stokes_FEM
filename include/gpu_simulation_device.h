#pragma once
#include "gpu_host_common.h"
#include "kd_tree_builder.h"
#include "gpu_sparse_matrix.h"
#include "sparse_matrix_math.h"
#include <cuda.h>

namespace EC {
    class ErrorCode;
}

namespace GPUSimulation {
    class GPUSimulationDevice : public GPU::GPUDeviceBase {
    friend class GPUSimulationDeviceManager;
    public:
        enum SimMatrix {
            pressureSitffness,
            velocityMass,
            diffusion,
            velocityDivergence,
            pressureDivergence,
            count
        };
        EC::ErrorCode loadModules(const char* advectionData, const char* sparseMatrixData);
        EC::ErrorCode uploadKDTree(const NSFem::KDTreeCPUOwner& cpuOwner);
        EC::ErrorCode initVelocityBuffers(const int numElements);
        EC::ErrorCode uploadMatrix(SimMatrix matrix, const SMM::TripletMatrix<float>& triplet);
        EC::ErrorCode advect(
            const int numElements,
            const float* uVelocity,
            const float* vVelocity,
            const float dt,
            float* uVelocityOut,
            float* vVelocityOut
        );
        EC::ErrorCode spRMult(
            SimMatrix matrix,
            const GPU::GPUBuffer& x,
            GPU::GPUBuffer& res
        );
        EC::ErrorCode spRMultSub(
            const SimMatrix matrix,
            const GPU::GPUBuffer& mult,
            const GPU::GPUBuffer& lhs,
            GPU::GPUBuffer& res
        );
        EC::ErrorCode conjugateGradient(
            const SimMatrix matrix,
            const float* const b,
            const float* const x0,
            float* const x,
            int maxIterations,
            float eps
        ) {
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
            {
                GPU::ScopedGPUContext ctxGuard(context);
                RETURN_ON_ERROR_CODE(pDev.init(byteSize));
                RETURN_ON_ERROR_CODE(apDev.init(byteSize));
                RETURN_ON_ERROR_CODE(apDev.uploadBuffer(x0, byteSize));
                RETURN_ON_ERROR_CODE(bDev.init(byteSize));
                RETURN_ON_ERROR_CODE(bDev.uploadBuffer(b, byteSize));
            }
            const float epsSuared = eps * eps;
            SMM::Vector<float> r(rows, 0);
            RETURN_ON_ERROR_CODE(spRMultSub(matrix, bDev, apDev, pDev));
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
    private:
        EC::ErrorCode loadAdvectionModule(const char* data);
        EC::ErrorCode loadSparseMatrixModule(const char* data);

        GPUSimulation::GPUSparseMatrix matrices[SimMatrix::count];

        CUmodule advectionModule;
        CUmodule sparseMatrixModule;

        CUfunction advection_kernel;
        CUfunction spRMult_kernel;
        CUfunction spRMultSub_kernel;

        NSFem::KDTreeGPUOwner kdTree;

        GPU::GPUBuffer uVelocityInBuffer;
        GPU::GPUBuffer vVelocityInBuffer;

        GPU::GPUBuffer uVelocityOutBuffer;
        GPU::GPUBuffer vVelocityOutBuffer;
    };

    class GPUSimulationDeviceManager : public GPU::GPUDeviceManagerBase<GPUSimulationDevice> {
    private:
        EC::ErrorCode loadModule(
            const char* filePath,
            const int numKernels,
            const char* kernelsToExtract[],
            CUmodule& moduleOut,
            CUfunction** kernelsOut
        );
    public:
        GPUSimulationDeviceManager();
        EC::ErrorCode init();
    };
};