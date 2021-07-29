#ifndef GPU_GRID_H
#define GPU_GRID_H
#include "defines_common.cuh"
#include "misc_common.cuh"

namespace GPUSimulation {
class GPUFemGrid2D {
public:
    device GPUFemGrid2D() :
        nodes(nullptr),
        elements(nullptr)
    {}
    device GPUFemGrid2D(
        const float* nodes,
        const int* elements,
        int nodesCount
    ) : 
        nodes(nodes),
        elements(elements),
        nodesCount(nodesCount)
    {}
    device void getElement(const int elementIndex, int* outElement, NSFem::real* outNodes) const {
        for(int i = 0; i < elementSize; ++i) {
            outElement[i] = (elements + elementSize * elementIndex)[i];
            outNodes[2 * i] =  nodes[outElement[i] * 2];
            outNodes[2 * i + 1] =  nodes[outElement[i] * 2 + 1];
        }
    }
    device const int* getElement(const int elementIndex) const {
        return elements + elementSize * elementIndex;
    }
    device int getElementSize() const {
        return elementSize;
    }
    device NSFem::Point2D getNode(const int nodeIndex) const {
        return NSFem::Point2D(nodes[2 * nodeIndex], nodes[2 * nodeIndex + 1]);
    }
    device int getNodesCount() const {
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