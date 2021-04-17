#pragma once
#include <vector>
#include <cinttypes>
#include <functional>
#include <vector>
#include <cstring>

namespace NSFem {


/// Class to represent 2D FEM grid. It will host the coordinates of all nodes, all elements as index buffers into the nodes
/// All boundary nodes, with the conditions which will be imposed on the nodes.
class FemGrid2D {
public:
    /// Return the number of nodes in the mesh
    const int getNodesCount() const;
    /// Return the number of elements in the mesh
    const int getElementsCount() const;
    /// Return the total number of indexes used to describe the elements.
    /// This number will be equal to number_of_nodes_per_element * elements_count
    /// This is needed as getElements returnes linear structure where the indexes
    /// of each elements are put one after another.
    const int getElementsBufferSize() const;
    /// Return linear structure of the nodes for the given mesh, where the first 2 elements
    /// are the (x, y) coordinates of the first point and so on.
    const float* getNodesBuffer() const;
    /// Return index buffer for the elements of the mesh. Each integer here is index in the nodes array
    /// for the node.
    const int* getElementsBuffer() const;
    /// Extract the i-th element and write it in outElement
    /// @param[in] elementIndex The index of the element to be extracted
    /// @param[out] outElement Array of size at least elementSize where node indices for the array will be extracted
    void getElement(const int elementIndex, int* outElement, float* outNodes) const;
protected:
    /// List containing 2D coordinates for each node in the grid.
    std::vector<float> nodes;
    /// Index buffer into nodes array. Each consecutive elementSize will form an elements.
    std::vector<int> elements;
    /// Cached number of elements. Must be equal to elements.size() / elementSize.
    int elementsCount;
    /// Number of nodes which represent a specific element.
    int elementSize;
};

inline const int FemGrid2D::getElementsCount() const {
    return elementsCount;
}

inline void FemGrid2D::getElement(const int elementIndex, int* outElement, float* outNodes) const {
    memcpy(outElement, elements.data() + elementSize * elementIndex, sizeof(int) * elementSize);
    for(int i = 0; i < elementSize; ++i) {
        outNodes[2 * i] =  nodes[outElement[i] * 2];
        outNodes[2 * i + 1] =  nodes[outElement[i] * 2 + 1];
    }
}

}