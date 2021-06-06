#include "kd_tree.h"
#include <cmath>
#include <vector>
#include<numeric>

namespace NSFem {

    template<typename T>
    static inline T square(T a) {
        return a * a;
    }

    bool isPointInTriagle(
        const Point2D& P,
        const Point2D& A,
        const Point2D& B,
        const Point2D& C,
        real& xi,
        real& eta
    ) {
        // This function uses the barrycentric coordinates to check if a point lies in a triangle.
        // Since we are in 2D we can describe point p as a linear combination of two vectors, we choose
        // them to be AB and AC. We want (0, 0) in this new space to be the point A
        // p = A + l1*AB + l2*AC
        // |px|  = |Ax + l1(Bx - Ax) + l2(Cx - Ax)|
        // |py|  = |Ay + l1(By - Ay) + l2(Cy - Ay)|
        //
        // |px - ax| = |Bx - Ax  Cx - Ax||l1|
        // |py - ay| = |By - Ay  Cy - Ay||l2|
        //
        // |l1| = 1/D * |Cy - Ay  Ax - Cx||px - ax|
        // |l2| = 1/D * |Ay - By  Bx - Ax||py - ay|
        const Point2D AB = B - A;
        const Point2D AC = C - A;
        const Point2D AP = P - A;
        const real D = (AB.x * AC.y) - (AB.y * AC.x);
        assert(D != 0);
        const int DSign = D > 0 ? 1 : -1;
        // We do not want to divide by the determinant, so when we solve the system we get the result
        // scaled by the determinant.
        const real scaledBarry1 =  (AP.x * AC.y - AP.y * AC.x) * DSign;
        if(scaledBarry1 < 0 || scaledBarry1 > D) {
            return false;
        }
        const real scaledBarry2 = (AP.y * AB.x - AP.x * AB.y) * DSign;
        if(scaledBarry2 < 0 || scaledBarry1 > D - scaledBarry2) {
            return false;
        }
            
        // When we reach this point we know that the point "start" is inside the triangle
        // We need to scale down the barrycentroc coords and we'll get the point inside the triangle
        const real dInv = 1 / std::abs(D);
        xi = scaledBarry1 * dInv;
        eta = scaledBarry2 * dInv;
        return true;
    }

    TriangleKDTree::TriangleKDTree(int maxDepth, int minLeafSize) :
        maxDepth(maxDepth),
        minLeafSize(minLeafSize)
    {}
    TriangleKDTree::TriangleKDTree() :
        maxDepth(-1),
        minLeafSize(16)
    {}
    void TriangleKDTree::init(FemGrid2D* grid) {
        this->grid = grid;
        this->bbox = grid->getBBox();
        maxDepth = maxDepth > -1 ? maxDepth : std::log(grid->getElementsCount());
        assert(grid->getElementSize() == 6);
        std::vector<int> indices(grid->getElementsCount());
        std::iota(indices.begin(), indices.end(), 0);
        build(indices, bbox, 0, 0);
    }

    int TriangleKDTree::build(
    	std::vector<int>& indices,
    	const BBox2D& boundingBox,
    	int axis,
    	int level
    ) {
    	if (level >= maxDepth || indices.size() <= minLeafSize) {
    		const int numTriangles = indices.size();
    		const int trianglesOffset = leafTriangleIndexes.size();
    		std::move(indices.begin(), indices.end(), std::back_inserter(leafTriangleIndexes));
    		const Node leaf = Node::makeLeaf(trianglesOffset, numTriangles);
    		nodes.push_back(leaf);
    		return nodes.size();
    	}

    	const float splitPoint = (bbox.getMin()[axis] + bbox.getMax()[axis]) * 0.5f;
    	Point2D leftBBMax = boundingBox.getMax();
    	leftBBMax[axis] = splitPoint;
    	BBox2D leftBoundingBox(boundingBox.getMin(), leftBBMax);

    	Point2D rightBBMin = boundingBox.getMin();
    	rightBBMin[axis] = splitPoint;
    	BBox2D rightBoundingBox(rightBBMin, boundingBox.getMax());

    	const int newAxis = (axis + 1) % 2;

    	std::vector<int> leftIndexes, rightIndexes;
    	for (const int elementIndex : indices) {
            const int nodeIndices[3] {
                grid->getElementsBuffer()[6 * elementIndex],
                grid->getElementsBuffer()[6 * elementIndex + 1],
                grid->getElementsBuffer()[6 * elementIndex + 2],
            };
            const Point2D& A = grid->getNode(nodeIndices[0]);
            const Point2D& B = grid->getNode(nodeIndices[1]);
            const Point2D& C = grid->getNode(nodeIndices[2]);

    		if (A[axis] <= splitPoint || B[axis] <= splitPoint || C[axis] <= splitPoint) {
    			leftIndexes.push_back(elementIndex);
    		}
    
    		if (A[axis] >= splitPoint || B[axis] >= splitPoint || C[axis] >= splitPoint) {
    			rightIndexes.push_back(elementIndex);
    		}
    	}

    	const int currentNodeIndex = nodes.size();
    	nodes.emplace_back();
    	const int rightNodeIndex = build(leftIndexes, leftBoundingBox, newAxis, level + 1);
    	nodes[currentNodeIndex].makeInternal(axis, rightNodeIndex, splitPoint);
    	return build(rightIndexes, rightBoundingBox, newAxis, level + 1);
    }

    int TriangleKDTree::findElement(const Point2D& point, real& xi, real& eta, int& closestFEMNodeIndex) {
        int currentNodeIndex = getRootIndex();
        Node currentNode = nodes[currentNodeIndex];
        std::vector<TraversalStackEntry> traversalStack;
        traversalStack.reserve(maxDepth);
        // Search the element which contains the given point (if there is such). Not that unlike
        // nearest neighbour (or raytracing) we should descend only to the nearest side of the splitting plane
        // and there is no need to descend to the "far" child
        while(!currentNode.isLeaf()) {
            const int axis = currentNode.getAxis();
            const real splitPoint = currentNode.getSplitPoint();
            const int childIndex = (point[axis] <= splitPoint) * (currentNodeIndex + 1) +
                (point[axis] > splitPoint) * currentNode.getRightChildIndex();
            traversalStack.emplace_back(currentNodeIndex, 1);
            currentNodeIndex = childIndex;
            currentNode = nodes[currentNodeIndex];
        }
        traversalStack.emplace_back(currentNodeIndex, 0);
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
            const Node& currentNode = nodes[currentNodeIndex];
            if(currentNode.isLeaf()) {
                // When we are at a leaf we will examine all points in the leaf and compare the minimal distance
                nearestNeghbourProcessLeaf(point, currentNode, minDistSq, closestFEMNodeIndex);
                traversalStack.pop_back();
            } else {
                // We can descend from each node exactly two times, so we need to cleare the stack
                // of all nodes which had both their children visited
                while(!traversalStack.empty() && traversalStack.back().isExhausted()) {
                    traversalStack.pop_back();
                }
                if(traversalStack.empty()) {
                    // This means that the root was popped in the while loop on the previous step
                    break;
                }
                const int activeNodeIndex = traversalStack.back().getNode();
                const Node& activeNode = nodes[activeNodeIndex];
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
                    traversalStack.emplace_back(nextIndex, 0);
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
                        traversalStack.emplace_back(farChildIndex, 0);
                    } else {
                        // This distance to the splitting plane is greater than the minimal distance found, no need to
                        // traverse the other side. Just pop the current node.
                        traversalStack.pop_back();
                    }
                } else {
                    assert(false);
                }
            }
        }
        return -1;
    }

    void TriangleKDTree::nearestNeghbourProcessLeaf(
        const Point2D& point,
        const Node& currentNode,
        real& minDistSq,
        int& closestFEMNodeIndex
    ) {
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
};