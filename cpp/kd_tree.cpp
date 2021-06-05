#include "kd_tree.h"
#include <cmath>

namespace NSFem {

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
        // |l2| = 1/D * |Ay - Cy  Bx - Ax||py - ay|
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
        const real dInv = 1 / D;
        xi = scaledBarry1 * dInv;
        eta = scaledBarry2 * dInv;
        return true;
    }

    TriangleKDTree::TriangleKDTree(int maxDepth, int maxLeafSize) :
        maxDepth(maxDepth),
        maxLeafSize(maxLeafSize)
    {}
    TriangleKDTree::TriangleKDTree() :
        maxDepth(-1),
        maxLeafSize(16)
    {}
    void TriangleKDTree::init(FemGrid2D* grid) {
        this->grid = grid;
        this->bbox = grid->getBBox();
        maxDepth = maxDepth > -1 ? maxDepth : std::log(grid->getElementsCount());
        assert(grid->getElementSize() == 6);
    }

    int TriangleKDTree::build(
    	std::vector<int>& indices,
    	const BBox2D& boundingBox,
    	int axis,
    	int level
    ) {
    	if (level > maxDepth || indices.size() <= maxLeafSize) {
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
            const Point2D A(grid->getNodesBuffer()[nodeIndices[0]], grid->getNodesBuffer()[nodeIndices[0] + 1]);
            const Point2D B(grid->getNodesBuffer()[nodeIndices[1]], grid->getNodesBuffer()[nodeIndices[1] + 1]);
            const Point2D C(grid->getNodesBuffer()[nodeIndices[2]], grid->getNodesBuffer()[nodeIndices[2] + 1]);

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

    int TriangleKDTree::findElement(const Point2D& point, real& xi, real& eta) {
        if(!bbox.isInside(point)) {
            return -1;
        }
        int currentNodeIndex = getRootIndex();
        Node currentNode = nodes[currentNodeIndex];
        while(!currentNode.isLeaf()) {
            const int axis = currentNode.getAxis();
            const bool goLeft = point[axis] <= currentNode.getSplitPoint(); 
            currentNodeIndex = goLeft ? currentNodeIndex + 1 : currentNode.getRightChildIndex();
            currentNode = nodes[currentNodeIndex];
        }
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
        return -1;
    }
 
};