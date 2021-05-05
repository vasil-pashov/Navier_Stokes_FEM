#pragma once
#include <vector>
#include <cinttypes>
#include <vector>
#include <cstring>
#include "expression.h"

namespace NSFem {

using real = double;

/// Class to represent 2D FEM grid. It will host the coordinates of all nodes, all elements as index buffers into the nodes
/// All boundary nodes, with the conditions which will be imposed on the nodes.
class FemGrid2D {
public:
    class VelocityDirichlet {
    public:
        friend class FemGrid2D;
        VelocityDirichlet() = default;
        /// Evaluate the boundary condition for the i-th node on the boundary- 1
        /// @param[in] vars List of variables which take part in the expression for the boundary condition
        /// @param[out] outU The result for the u component of the velocity on the boundary
        /// @param[out] outv The result for the v component of the velocity on the boundary
        void eval(const std::unordered_map<char, float>* vars, float& outU, float& outV) const;
        /// Get the number of nodes which are included in the boundary
        int getSize() const;
        /// Gives a list with the indexes of all nodes which are on the boundary
        const int* getNodeIndexes() const;
    private:
        /// Indexes in in FemGrid2D::nodes to the nodes which are on the boundary
        std::vector<int> nodeIndexes;
        /// The boundary condition for the u component of the velocity
        Expression u;
        /// The boundary condition for the v component of the velocity
        Expression v;
    };

    struct PressureDirichlet {
    public:
        friend class FemGrid2D;
        PressureDirichlet() = default;
        /// Evaluate the boundary condition for the i-th node on the boundary
        /// @param[in] vars List of variables which take part in the expression for the boundary condition
        /// The coordinates of the nodd (x, y) will be passed by the function internaly
        /// @param[out] outP The result for the pressure on the bondary
        void eval(const std::unordered_map<char, float>* vars, float& outP) const;
        /// Get the number of nodes which are included in the boundary
        int getSize() const;
        /// Gives a list with the indexes of all nodes which are on the boundary
        const int* getNodeIndexes() const;
    private:
        /// Indexes in in FemGrid2D::nodes to the nodes which are on the boundary
        std::vector<int> nodeIndexes;
        /// The boundary condition for the pressure
        Expression p;
    };
    /// Return the number of nodes in the mesh
    int getNodesCount() const;
    /// Return the number of elements in the mesh
    int getElementsCount() const;
    /// Return the total number of indexes used to describe the elements.
    /// This number will be equal to number_of_nodes_per_element * elements_count
    /// This is needed as getElements returnes linear structure where the indexes
    /// of each elements are put one after another.
    int getElementsBufferSize() const;
    /// Return linear structure of the nodes for the given mesh, where the first 2 elements
    /// are the (x, y) coordinates of the first point and so on.
    const real* getNodesBuffer() const;
    /// Return index buffer for the elements of the mesh. Each integer here is index in the nodes array
    /// for the node.
    const int* getElementsBuffer() const;
    /// Extract the i-th element and write it in outElement
    /// @param[in] elementIndex The index of the element to be extracted
    /// @param[out] outElement Array of size at least elementSize where node indices for the array will be extracted
    void getElement(const int elementIndex, int* outElement, real* outNodes) const;
    /// Load mesh written in JSON file format
    /// @param[in] filePath Path the the mesh file
    /// @returns Status code: 0 on success
    int loadJSON(const char* filePath);
    using VelocityDirichletConstIt = const VelocityDirichlet*;
    /// Get the number of different boundaries where Dirichlet condition is imposed for the velocity
    int getVelocityDirichletSize() const;
    /// Get iterator to all different boundaries where Dirichlet condition is imposed for the velocity
    VelocityDirichletConstIt getVelocityDirichlet() const;

    using PressureDirichletConstIt = const PressureDirichlet*;
    /// Get the number of different boundaries where Dirichlet condition is imposed for the pressure
    int getPressureDirichletSize() const;
    /// Get iterator to all different boundaries where Dirichlet condition is imposed for the pressure
    PressureDirichletConstIt getPressureDirichlet() const;
protected:
    /// List containing 2D coordinates for each node in the grid.
    std::vector<real> nodes;
    /// Index buffer into nodes array. Each consecutive elementSize will form an elements.
    std::vector<int> elements;
    /// Cached number of elements. Must be equal to elements.size() / elementSize.
    int elementsCount;
    /// Number of nodes which represent a specific element.
    int elementSize;
    /// The nuber of nodes in the mesh
    int nodesCount;

    std::vector<VelocityDirichlet> velocityDirichlet;
    std::vector<PressureDirichlet> pressureDirichlet;
};

inline int FemGrid2D::getVelocityDirichletSize() const {
    return velocityDirichlet.size();
}

inline FemGrid2D::VelocityDirichletConstIt FemGrid2D::getVelocityDirichlet() const {
    return velocityDirichlet.data();
}

inline int FemGrid2D::getPressureDirichletSize() const {
    return pressureDirichlet.size();
}
/// Get iterator to all different boundaries where Dirichlet condition is imposed for the velocity
inline FemGrid2D::PressureDirichletConstIt FemGrid2D::getPressureDirichlet() const {
    return pressureDirichlet.data();
}

inline int FemGrid2D::getNodesCount() const {
    return nodesCount;
}

inline const real* FemGrid2D::getNodesBuffer() const {
    return nodes.data();
}

inline int FemGrid2D::getElementsCount() const {
    return elementsCount;
}

inline void FemGrid2D::getElement(const int elementIndex, int* outElement, real* outNodes) const {
    memcpy(outElement, elements.data() + elementSize * elementIndex, sizeof(int) * elementSize);
    for(int i = 0; i < elementSize; ++i) {
        outNodes[2 * i] =  nodes[outElement[i] * 2];
        outNodes[2 * i + 1] =  nodes[outElement[i] * 2 + 1];
    }
}

inline int FemGrid2D::VelocityDirichlet::getSize() const {
    return nodeIndexes.size();
}

inline const int* FemGrid2D::VelocityDirichlet::getNodeIndexes() const {
    return nodeIndexes.data();
}

inline int FemGrid2D::PressureDirichlet::getSize() const {
    return nodeIndexes.size();
}

inline const int* FemGrid2D::PressureDirichlet::getNodeIndexes() const {
    return nodeIndexes.data();
}

}