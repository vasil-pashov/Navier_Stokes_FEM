#pragma once
#include <vector>
#include <cinttypes>
#include <vector>
#include <cstring>
#include "expression.h"

namespace NSFem {

using real = float;

struct Point2D {
    Point2D() : x(0), y(0) {}
    Point2D(real x, real y) : x(x), y(y) {}
    Point2D operator+(const Point2D& other) const {
        return Point2D(x + other.x, y + other.y);
    }
    Point2D operator-(const Point2D& other) const {
        return Point2D(x - other.x, y - other.y);
    }
    Point2D operator*(const real scalar) const {
        return Point2D(x * scalar, y * scalar);
    }
    /// Find the squared distance to another 2D point
    real distToSq(const Point2D& other) const {
        return (x - other.x)*(x - other.x) + (y - other.y)*(y - other.y);
    }
    real operator[](const int idx) const {
        assert(idx < 2);
        return (&x)[idx];
    }
    real& operator[](const int idx) {
        assert(idx < 2);
        return (&x)[idx];
    }
    real x, y;
};

class BBox2D {
public:
    BBox2D() :
        min(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()),
        max(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity())
    {
        static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
    }

    BBox2D(const Point2D& min, const Point2D& max) : 
        min(min), max(max)
    {}

    void reset() {
        min.x = std::numeric_limits<float>::infinity();
        min.y = std::numeric_limits<float>::infinity();
        max.x = -std::numeric_limits<float>::infinity();
        max.y = -std::numeric_limits<float>::infinity();
    }

    void expand(const Point2D& point) {
        min.x = std::min(point.x, min.x);
        min.y = std::min(point.y, min.y);

        max.x = std::max(point.x, point.x);
        max.y = std::max(point.y, point.y);
    }

    bool isInside(const Point2D& point) {
        return (min.x <= point.x && point.x <= max.x) && (min.y <= point.y && point.y <= max.y);
    }

    const Point2D getMax() const {
        return max;
    }

    const Point2D getMin() const {
        return min;
    }
private:
    Point2D min;
    Point2D max;
};



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
    /// Return the number of pressure nodes in the mesh
    int getPressureNodesCount() const;
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
    const int* getElementsBuffer() const {
        return elements.data();
    }
    const int* getElement(const int elementIndex) const {
        return elements.data() + elementIndex * elementSize;
    }
    const Point2D& getNode(const int nodeIndex) const {
        return *reinterpret_cast<const Point2D*>(nodes.data() + 2 * nodeIndex);
    }
    /// Extract the i-th element and write it in outElement
    /// @param[in] elementIndex The index of the element to be extracted
    /// @param[out] outElement Array of size at least elementSize where node indices for the array will be extracted
    void getElement(const int elementIndex, int* outElement, real* outNodes) const;
    /// Return the number of nodes in each element.
    int getElementSize() const;
    /// Load mesh written in JSON file format
    /// @param[in] filePath Path the the mesh file
    /// @returns Status code: 0 on success
    EC::ErrorCode loadJSON(const char* filePath);

    using VelocityDirichletConstIt = std::vector<VelocityDirichlet>::const_iterator;
    /// Get the number of different boundaries where Dirichlet condition is imposed for the velocity
    int getVelocityDirichletSize() const;
    /// Get iterator to all different boundaries where Dirichlet condition is imposed for the velocity
    VelocityDirichletConstIt getVelocityDirichletBegin() const;
    VelocityDirichletConstIt getVelocityDirichletEnd() const;

    using PressureDirichletConstIt = std::vector<PressureDirichlet>::const_iterator;
    /// Get the number of different boundaries where Dirichlet condition is imposed for the pressure
    int getPressureDirichletSize() const;
    /// Get iterator to all different boundaries where Dirichlet condition is imposed for the pressure
    PressureDirichletConstIt getPressureDirichletBegin() const;
    /// Iterator one past the last pressure Dirichlet boundary. Should not be dereferenced.
    PressureDirichletConstIt getPressureDirichletEnd() const;

    const BBox2D& getBBox() const {
        return bbox;
    }
protected:
    /// List containing 2D coordinates for each node in the grid.
    std::vector<real> nodes;
    /// Index buffer into nodes array. Each consecutive elementSize will form an elements.
    std::vector<int> elements;
    /// Cached number of elements. Must be equal to elements.size() / elementSize.
    int elementsCount;
    /// Number of nodes which represent a specific element.
    int elementSize;
    /// The number of velocity nodes in the mesh. This is the same as the total number of nodes in the mesh
    int velocityNodesCount;
    /// The number of pressure nodes in the mesh. Pressure nodes are a subset of the velocity nodes.
    int pressureNodesCount;


    std::vector<VelocityDirichlet> velocityDirichlet;
    std::vector<PressureDirichlet> pressureDirichlet;

    BBox2D bbox;
};

inline int FemGrid2D::getVelocityDirichletSize() const {
    return velocityDirichlet.size();
}

inline FemGrid2D::VelocityDirichletConstIt FemGrid2D::getVelocityDirichletBegin() const {
    return velocityDirichlet.begin();
}

inline FemGrid2D::VelocityDirichletConstIt FemGrid2D::getVelocityDirichletEnd() const {
    return velocityDirichlet.end();
}

inline int FemGrid2D::getPressureDirichletSize() const {
    return pressureDirichlet.size();
}
/// Get iterator to all different boundaries where Dirichlet condition is imposed for the velocity
inline FemGrid2D::PressureDirichletConstIt FemGrid2D::getPressureDirichletBegin() const {
    return pressureDirichlet.begin();
}

inline FemGrid2D::PressureDirichletConstIt FemGrid2D::getPressureDirichletEnd() const {
    return pressureDirichlet.end();
}

inline int FemGrid2D::getNodesCount() const {
    return velocityNodesCount;
}

inline int FemGrid2D::getPressureNodesCount() const {
    return pressureNodesCount;
}

inline const real* FemGrid2D::getNodesBuffer() const {
    return nodes.data();
}

inline int FemGrid2D::getElementsCount() const {
    return elementsCount;
}

inline int FemGrid2D::getElementSize() const {
    return elementSize;
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
