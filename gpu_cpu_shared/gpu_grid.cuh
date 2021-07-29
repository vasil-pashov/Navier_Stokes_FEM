#ifndef GPU_GRID_H
#define GPU_GRID_H
#include "defines_common.cuh"
#include "misc_common.cuh"

namespace GPUSimulation {
class GPUFemGrid2D {
public:
    DEVICE GPUFemGrid2D() :
        nodes(nullptr),
        elements(nullptr)
    {}
    DEVICE GPUFemGrid2D(
        const float* nodes,
        const int* elements,
        int nodesCount
    ) : 
        nodes(nodes),
        elements(elements),
        nodesCount(nodesCount)
    {}
    DEVICE void getElement(const int elementIndex, int* outElement, NSFem::real* outNodes) const {
        for(int i = 0; i < elementSize; ++i) {
            outElement[i] = (elements + elementSize * elementIndex)[i];
            outNodes[2 * i] =  nodes[outElement[i] * 2];
            outNodes[2 * i + 1] =  nodes[outElement[i] * 2 + 1];
        }
    }
    DEVICE const int* getElement(const int elementIndex) const {
        return elements + elementSize * elementIndex;
    }
    DEVICE int getElementSize() const {
        return elementSize;
    }
    DEVICE NSFem::Point2D getNode(const int nodeIndex) const {
        return NSFem::Point2D(nodes[2 * nodeIndex], nodes[2 * nodeIndex + 1]);
    }
    DEVICE int getNodesCount() const {
        return nodesCount;
    }
private:
    const float* nodes;
    const int* elements;
    int elementSize = 6;
    int nodesCount;
};
}
#endif
