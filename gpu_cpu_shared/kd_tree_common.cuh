#ifndef KD_TREE_COMMON_H
#define KD_TREE_COMMON_H

#include <cassert>

#ifdef __CUDACC__
    #define device __device__
#else
    #define device
#endif

namespace NSFem {
    struct KDNode {
public:
    device KDNode();
    /// Setup existing node to be an internal one. The left child index
    /// is implicit and is the one after the current one (in the array of nodes).
    /// @param[in] axis The axis which split this node to form the left and right children
    /// @param[in] rightChildIndex The index of the right child in the array of nodes
    /// @param[in] splitPoint Coordinates along the axis where the node was split into left and right children
    device void makeInternal(unsigned int axis, unsigned int rightChildIndex, float splitPoint);
    /// Create a leaf node, which contains element indices
    /// @param[in] triangleOffset Index in the global array of triangle indices where the elements for
    /// this leaf start
    /// @param[in] numTriangles Number of elements into this leaf
    /// @returns Leaf node for the KDTree
    device static KDNode makeLeaf(unsigned int triangleOffset, unsigned int numTriangles);
    /// Retrieve the axis which splits this node into left and right children
    device int getAxis() const;
    /// Check if the node is a leaf node
    device bool isLeaf() const;
    /// @brief Retrieve the number of elements in a leaf.
    /// @note This function should be called only if the node is a leaf.
    device int getNumTrianges() const;
    /// Retrieve index into the global array of nodes where the right child is held
    device int getRightChildIndex() const;
    /// Retrieve the coordinate along the split axis where this node was split into left and right
    device float getSplitPoint() const;
    /// @brief Retrieve index into the global array of triangle indices where the elements for a leaf start
    /// @note This function should be called only if the node is a leaf
    device float getTriangleOffset() const;
private:
    /// Set the index into the global array of nodes where the right child of this node is
    device void setRightChildIndex(unsigned int index);
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

device inline KDNode::KDNode() : 
	flags(0),
	triangleOffset(0)
{ }

device inline void KDNode::makeInternal(unsigned int axis, unsigned int rightChildIndex, float splitPoint) {
	assert(axis < 2);
	this->flags = axis;
    setRightChildIndex(rightChildIndex);
	this->splitPoint = splitPoint;
}

device inline KDNode KDNode::makeLeaf(unsigned int triangleOffset, unsigned int numTriangles) {
	KDNode res;
	res.flags = static_cast<unsigned int>(2);
	res.flags |= (numTriangles << 2);
	res.triangleOffset = triangleOffset;
	return res;
}

device inline int KDNode::getAxis() const {
	return static_cast<unsigned int>(3) & flags;
}

device inline bool KDNode::isLeaf() const {
	return (getAxis() == 2);
}

device inline int KDNode::getNumTrianges() const {
	return flags >> 2;
}

device inline int KDNode::getRightChildIndex() const {
	return flags >> 2;
}

device inline float KDNode::getSplitPoint() const {
	return splitPoint;
}

device inline float KDNode::getTriangleOffset() const {
	return triangleOffset;
}

device inline void KDNode::setRightChildIndex(unsigned int index) {
	flags |= (index << 2);
}

struct TraversalStackEntry {
    device TraversalStackEntry() : node(0), count(0) {}
    device TraversalStackEntry(int node, int count) :
        node(node),
        count(count)
    {}
    device void descend() {
        count++;
    }
    device int getVisitCount() const {
        return count;
    }
    device bool isExhausted() const {
        return count == 2;
    }
    device int getNode() const {
        return node;
    }
private:
    /// Index into an array of nodes of the node being processed
    int node;
    /// Count how many times this node was visited. Since this is a binary tree the node
    /// can be visited at most twice: frist when we go down the left child and second time
    /// when we go down the right child.
    int count;
};
}
#endif