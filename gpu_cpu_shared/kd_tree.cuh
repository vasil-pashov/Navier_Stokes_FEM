#pragma once
#include <limits>
#include <cmath>
#include <numeric>
#include "kd_tree_common.cuh"
#include "small_vector.cuh"
#include "defines_common.cuh"
namespace NSFem {

class FemGrid2D;

template<typename T>
device inline T square(T a) {
    return a * a;
}


template<typename Grid>
class KDTree {
public:
    friend class KDTreeBuilder;
    /// @brief Default constrict the tree with empty bounding box and no nodes.
    KDTree();
    KDTree(
        BBox2D bbox,
        const KDNode* nodes,
        const int* leafTriangleIndexes,
        const Grid* grid
    );
    KDTree(KDTree&&) = default;
    KDTree& operator=(KDTree&&) = default;
    KDTree(const KDTree&) = delete;
    KDTree& operator=(const KDTree&) = delete;
    /// @brief Checks if a point lies inside any of the triangles in the grid.
    /// If so xi and eta will be set to the barrycentric coordinates of the point inside the triangle
    /// @param[in] point 2D point in world space which will be tested against the tree
    /// @param[out] xi First barrycentric coordinate of the point inside the triangle
    /// @param[out] eta Second barrycentric coordinate of the point inside the triangle
    /// @param[out] closestFEMNodeIndex If point does not lie inside any element, this will hold the index
    /// of the closest point of the grid to the given one
    /// @retval -1 if the point does not lie inside a triangle, otherwise the index of the element where
    /// the point lies 
    device int findElement(const Point2D& point, real& xi, real& eta, int& closestFEMNodeIndex) const {
        int currentNodeIndex = getRootIndex();
        KDNode currentNode = nodes[currentNodeIndex];
        SmallVector<TraversalStackEntry, 32> traversalStack;
        // Search the element which contains the given point (if there is such). Not that unlike
        // nearest neighbour (or raytracing) we should descend only to the nearest side of the splitting plane
        // and there is no need to descend to the "far" child
        while(!currentNode.isLeaf()) {
            const int axis = currentNode.getAxis();
            const real splitPoint = currentNode.getSplitPoint();
            const int childIndex = (point[axis] <= splitPoint) * (currentNodeIndex + 1) +
                (point[axis] > splitPoint) * currentNode.getRightChildIndex();
            traversalStack.emplaceBack(currentNodeIndex, 1);
            currentNodeIndex = childIndex;
            currentNode = nodes[currentNodeIndex];
        }
        traversalStack.emplaceBack(currentNodeIndex, 0);
        assert(currentNode.isLeaf());
        const int elementOffset = currentNode.getTriangleOffset();
        for(int i = 0; i < currentNode.getNumTrianges(); ++i) {
            const int elementIndex = leafTriangleIndexes[elementOffset + i];
            const int* element = grid->getElement(elementIndex);
            const Point2D& A = grid->getNode(element[0]);
            const Point2D& B = grid->getNode(element[1]);
            const Point2D& C = grid->getNode(element[2]);
            if(isPointInTriagle(point, A, B, C, xi, eta)) {
                return elementIndex;
            }
        }
    
        // If we reach here, this means that the point does not lie in any triangle of the mesh
        // We must proceed with finding the nearest neighbour, by unwinding the recursion
        // 1) Find the closest point point in all elements which were at the leaf. Note that there can be
        // a point which is in another node and is closer to the one in this leaf.
        // 2) Go upwards the recursion stack. For each node check if the distance between the point and the
        // splitting plane is less than the minimal distance found. If so the other node must traversed too.
        real minDistSq = std::numeric_limits<real>::infinity();
        assert(grid->getElementSize() == 6);
        while(!traversalStack.empty()) {
            const TraversalStackEntry& stackEntry = traversalStack.back();
            const int currentNodeIndex = stackEntry.getNode();
            const KDNode& currentNode = nodes[currentNodeIndex];
            if(currentNode.isLeaf()) {
                // When we are at a leaf we will examine all points in the leaf and compare the minimal distance
                nearestNeghbourProcessLeaf(point, currentNode, minDistSq, closestFEMNodeIndex);
                traversalStack.popBack();
            } else {
                // We can descend from each node exactly two times, so we need to cleare the stack
                // of all nodes which had both their children visited
                while(!traversalStack.empty() && traversalStack.back().isExhausted()) {
                    traversalStack.popBack();
                }
                if(traversalStack.empty()) {
                    // This means that the root was popped in the while loop on the previous step
                    break;
                }
                const int activeNodeIndex = traversalStack.back().getNode();
                const KDNode& activeNode = nodes[activeNodeIndex];
                const int axis = activeNode.getAxis();
                const real splitPoint = activeNode.getSplitPoint();
                if(traversalStack.back().getVisitCount() == 0) {
                    // The node on the top of the stack did not descent to neighter of its children
                    // We must always go down to the child which is on the same side of the splitting plane
                    // as the point we are searching.
    
                    // Check if the point is to the left or to the right of the splitting plane.
                    // If it's to the left (point[axis] <= splitPoint) will be 1 (point[axis] > splitPoint) will be 0
                    // so the next index will be the left child.
                    const int nextIndex = (point[axis] <= splitPoint) * (activeNodeIndex + 1) + 
                        (point[axis] > splitPoint) * activeNode.getRightChildIndex();
    
                    // Mark that the current node has one of its children traversed (the one we will add)
                    traversalStack.back().descend();
    
                    // Add the new child to the stack. When children are added we assume that they were not used by now
                    traversalStack.emplaceBack(nextIndex, 0);
                } else if(traversalStack.back().getVisitCount() == 1) {
                    // If the current node has one of its children traversed we need to check if the other must
                    // be traversed too. We need to traverse "the far" child if the distance between the splitting
                    // plane is less than the current minimal distance.
    
                    const real distToSplitSq = square(point[axis] - splitPoint);
                    if(distToSplitSq < minDistSq) {
                        // We must traverse the "far" child. In order to find it we check of which side of the splitting
                        // plane our point is and take the other.
                        const int farChildIndex = (point[axis] <= splitPoint) * activeNode.getRightChildIndex() + 
                            (point[axis] > splitPoint) * (activeNodeIndex + 1);
                        traversalStack.back().descend();
                        assert(traversalStack.back().isExhausted());
                        traversalStack.emplaceBack(farChildIndex, 0);
                    } else {
                        // This distance to the splitting plane is greater than the minimal distance found, no need to
                        // traverse the other side. Just pop the current node.
                        traversalStack.popBack();
                    }
                } else {
                    assert(false);
                }
            }
        }
        return -1;
    }
private:
    device int getRootIndex() const {
        return 0;
    }

    /// Compare all of the points in the leaf and check if any of the points has distance to the
    /// given point less than minDistSq. If so update minDistSq and the index of the femNode in the mesh
    device void nearestNeghbourProcessLeaf(
        const Point2D& point,
        const KDNode& currentNode,
        real& minDistSq,
        int& closestFEMNodeIndex
    ) const {
        Point2D femNodes[6];
        int femIndices[6];
        const int elementOffset = currentNode.getTriangleOffset();
        for(int i = 0; i < currentNode.getNumTrianges(); ++i) {
            const int finiteElementIndex = leafTriangleIndexes[elementOffset + i];
            grid->getElement(finiteElementIndex, femIndices, reinterpret_cast<real*>(femNodes));
            for(int j = 0; j < 6; ++j) {
                const real newDistSq = point.distToSq(femNodes[j]);
                if(newDistSq < minDistSq) {
                    minDistSq = newDistSq;
                    closestFEMNodeIndex = femIndices[j];
                }
            }
        }
    } 

    BBox2D bbox;
    const KDNode* nodes;
    const int* leafTriangleIndexes;
    const Grid *grid;
};

template<typename Grid>
KDTree<Grid>::KDTree() :
    grid(nullptr)
{}

template<typename Grid>
KDTree<Grid>::KDTree(
    BBox2D bbox,
    const KDNode* nodes,
    const int* leafTriangleIndexes,
    const Grid* grid
) :
    bbox(bbox),
    nodes(nodes),
    leafTriangleIndexes(leafTriangleIndexes),
    grid(grid)
{}

};