#include <cassert>
#include "assembly.h"
#include "grid.h"

namespace NSFem {
    void NavierStokesAssembly::assembleMassMatrix() {
        const int p2Size = 6;
        float p2Integrated[p2Size];
        integrateOverTriangle<p2Size>(p2Shape, p2Integrated);
        float p2Local[p2Size][p2Size];
        for(int i = 0; i < p2Size; ++i) {
            for(int j = 0; j < p2Size; ++j) {
                p2Local[i][j] = p2Integrated[i] * p2Integrated[j];
            }
        }

        const auto localMass = [&p2Local, p2Size](float* elementNodes, float* localMatrixOut) -> void {
            const float jDetAbs = std::abs(linTriangleTmJacobian(
                elementNodes[0], elementNodes[1],
                elementNodes[2], elementNodes[3],
                elementNodes[4], elementNodes[5]
            ));
            for(int i = 0; i < p2Size; ++i) {
                for(int j = 0; j < p2Size; ++j) {
                    const int index = i * p2Size + j;
                    localMatrixOut[index] = p2Local[i][j] * jDetAbs;
                }
            }
        };

        assembleMatrix<decltype(localMass), p2Size, p2Size>(localMass, velocityMassMatrix);
    }
}