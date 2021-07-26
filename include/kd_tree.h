#pragma once
#include <vector>
#include <limits>
#include "grid.h"

namespace NSFem {

/// Check if a 2D point p lies inside the triangle formed by point A, B and C
/// @param[in] P The point which is goint to be tested
/// @param[in] A First vertex of the triangle
/// @param[in] B Second vertex of the triangle
/// @param[in] C Third vertex of the triangle
/// @param[out] xi First barrycentric coordinate of the point inside the triangle
/// @param[out] eta Second barrycentric coordinate of the point inside the triangle
/// @retval true if the point lies in the triangle, false othwerwise
bool isPointInTriagle(
    const Point2D& p,
    const Point2D& A,
    const Point2D& B,
    const Point2D& C,
    real& xi,
    real& eta
);

class TriangleKDTree {
public:
    friend class TriangleKDTreeBuilder;
    /// @brief Default constrict the tree with empty bounding box and no nodes.
    TriangleKDTree();
    TriangleKDTree(TriangleKDTree&&) = default;
    TriangleKDTree& operator=(TriangleKDTree&&) = default;
    TriangleKDTree(const TriangleKDTree&) = delete;
    TriangleKDTree& operator=(const TriangleKDTree&) = delete;
    /// @brief Checks if a point lies inside any of the triangles in the grid.
    /// If so xi and eta will be set to the barrycentric coordinates of the point inside the triangle
    /// @param[in] point 2D point in world space which will be tested against the tree
    /// @param[out] xi First barrycentric coordinate of the point inside the triangle
    /// @param[out] eta Second barrycentric coordinate of the point inside the triangle
    /// @param[out] closestFEMNodeIndex If point does not lie inside any element, this will hold the index
    /// of the closest point of the grid to the given one
    /// @retval -1 if the point does not lie inside a triangle, otherwise the index of the element where
    /// the point lies 
    int findElement(const Point2D& point, real& xi, real& eta, int& closestFEMNodeIndex) const;
private:
	struct Node {
	public:
		Node();
        /// Setup existing node to be an internal one. The left child index
        /// is implicit and is the one after the current one (in the array of nodes).
        /// @param[in] axis The axis which split this node to form the left and right children
        /// @param[in] rightChildIndex The index of the right child in the array of nodes
        /// @param[in] splitPoint Coordinates along the axis where the node was split into left and right children
		void makeInternal(unsigned int axis, unsigned int rightChildIndex, float splitPoint);
        /// Create a leaf node, which contains element indices
        /// @param[in] triangleOffset Index in the global array of triangle indices where the elements for
        /// this leaf start
        /// @param[in] numTriangles Number of elements into this leaf
        /// @returns Leaf node for the KDTree
		static Node makeLeaf(unsigned int triangleOffset, unsigned int numTriangles);
        /// Retrieve the axis which splits this node into left and right children
		int getAxis() const;
        /// Check if the node is a leaf node
		bool isLeaf() const;
        /// @brief Retrieve the number of elements in a leaf.
        /// @note This function should be called only if the node is a leaf.
		int getNumTrianges() const;
        /// Retrieve index into the global array of nodes where the right child is held
		int getRightChildIndex() const;
        /// Retrieve the coordinate along the split axis where this node was split into left and right
		float getSplitPoint() const;
        /// @brief Retrieve index into the global array of triangle indices where the elements for a leaf start
        /// @note This function should be called only if the node is a leaf
		float getTriangleOffset() const;
	private:
        /// Set the index into the global array of nodes where the right child of this node is
		void setRightChildIndex(unsigned int index);
		// the lower two bits define the split axis x-0, y-1 or if the node is leaf-2
		// If the node is leaf the upper 30 bits represent the number of triangles in the leaf
		// If the node is not a leaf, the upper 30 bits represent index in the array of nodes
		// where the node containing the part above the split poisiton is contained
		// Do not use union for this as reading the member which was not written last is UB and we dont know
		// from which member of the union to read the lower 2 bits
		unsigned int flags;
		union {
			float splitPoint;
			unsigned int triangleOffset;
		};
	};

    int getRootIndex() const {
        return 0;
    }

    /// Compare all of the points in the leaf and check if any of the points has distance to the
    /// given point less than minDistSq. If so update minDistSq and the index of the femNode in the mesh
    void nearestNeghbourProcessLeaf(const Point2D& point, const Node& node, real& minDistSq, int& closestFEMNodeIndex) const;

    struct TraversalStackEntry {
        TraversalStackEntry() : node(0), count(0) {}
        TraversalStackEntry(int node, int count) :
            node(node),
            count(count)
        {}
        void descend() {
            count++;
        }
        int getVisitCount() const {
            return count;
        }
        bool isExhausted() const {
            return count == 2;
        }
        int getNode() const {
            return node;
        }
    private:
        int node;
        /// Count how many times this node was visited. Since this is a binary tree the node
        /// can be visited at most twice: frist when we go down the left child and second time
        /// when we go down the right child.
        int count;
    };

    std::vector<Node> nodes;
    std::vector<int> leafTriangleIndexes;
    BBox2D bbox;
    FemGrid2D *grid;
};


inline TriangleKDTree::Node::Node() : 
	flags(0),
	triangleOffset(0)
{ }

inline void TriangleKDTree::Node::makeInternal(unsigned int axis, unsigned int rightChildIndex, float splitPoint) {
	assert(axis < 2);
	this->flags = axis;
    setRightChildIndex(rightChildIndex);
	this->splitPoint = splitPoint;
}

inline TriangleKDTree::Node TriangleKDTree::Node::makeLeaf(unsigned int triangleOffset, unsigned int numTriangles) {
	Node res;
	res.flags = static_cast<unsigned int>(2);
	res.flags |= (numTriangles << 2);
	res.triangleOffset = triangleOffset;
	return res;
}

inline int TriangleKDTree::Node::getAxis() const {
	return static_cast<unsigned int>(3) & flags;
}

inline bool TriangleKDTree::Node::isLeaf() const {
	return (getAxis() == 2);
}

inline int TriangleKDTree::Node::getNumTrianges() const {
	return flags >> 2;
}

inline int TriangleKDTree::Node::getRightChildIndex() const {
	return flags >> 2;
}

inline float TriangleKDTree::Node::getSplitPoint() const {
	return splitPoint;
}

inline float TriangleKDTree::Node::getTriangleOffset() const {
	return triangleOffset;
}

inline void TriangleKDTree::Node::setRightChildIndex(unsigned int index) {
	flags |= (index << 2);
}

class TriangleKDTreeBuilder {
public:
    TriangleKDTreeBuilder();
    TriangleKDTreeBuilder(int maxDepth, int minLeafSize);
    TriangleKDTree build(FemGrid2D* grid);
private:
    int build(
        FemGrid2D* grid,
        std::vector<int>& leafTriangleIndexes,
        std::vector<TriangleKDTree::Node>& nodes,
        std::vector<int>& indices,
        const BBox2D& subtreeBBox,
        int axis,
        int level
    );
    int maxDepth;
    int minLeafSize;
};


};