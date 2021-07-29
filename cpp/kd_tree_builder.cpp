#include "kd_tree_builder.h"
#include "grid.h"
#include "kd_tree.cuh"

namespace NSFem {

KDTreeBuilder::KDTreeBuilder() :
    maxDepth(-1),
    minLeafSize(16)
{}

KDTreeBuilder::KDTreeBuilder(int maxDepth, int minLeafSize) :
    maxDepth(std::min(maxDepth, 29)),
    minLeafSize(minLeafSize)
{}

KDTreeCPUOwner KDTreeBuilder::buildCPUOwner(FemGrid2D* grid) {
    assert(grid != nullptr);
    KDTreeCPUOwner owner;
    build(grid, owner.leafTriangleIndexes, owner.nodes);
    owner.grid = grid;
    owner.treeBBox = grid->getBBox();
    owner.grid = grid;
    return owner;
}

KDTree<FemGrid2D> KDTreeCPUOwner::getTree() const {
    return KDTree<FemGrid2D>(
        treeBBox,
        nodes.data(),
        leafTriangleIndexes.data(),
        grid
    );
}

EC::ErrorCode KDTreeCPUOwner::upload(KDTreeGPUOwner& ownerOut) const {
    assert(grid->getElementSize() == 6);
    const int64_t nodesSize = sizeof(decltype(nodes)::value_type) * nodes.size();
    const int64_t indicesSize = sizeof(decltype(leafTriangleIndexes)::value_type) * leafTriangleIndexes.size();
    const int64_t gridNodesBufferSize = sizeof(float) * 2 * grid->getNodesCount();
    const int64_t gridElementsBufferSize = sizeof(int) * grid->getElementSize() * grid->getElementsCount();

    ownerOut.treeBBox = treeBBox;

    RETURN_ON_ERROR_CODE(ownerOut.nodes.init(nodesSize));
    RETURN_ON_ERROR_CODE(ownerOut.nodes.uploadBuffer(nodes.data(), nodesSize));

    RETURN_ON_ERROR_CODE(ownerOut.leafTriangleIndexes.init(indicesSize));
    RETURN_ON_ERROR_CODE(ownerOut.leafTriangleIndexes.uploadBuffer(leafTriangleIndexes.data(), indicesSize));

    RETURN_ON_ERROR_CODE(ownerOut.gridNodes.init(gridNodesBufferSize));
    RETURN_ON_ERROR_CODE(ownerOut.gridNodes.uploadBuffer(grid->getNodesBuffer(), gridNodesBufferSize));

    RETURN_ON_ERROR_CODE(ownerOut.gridElements.init(gridElementsBufferSize));
    RETURN_ON_ERROR_CODE(ownerOut.gridElements.uploadBuffer(grid->getElementsBuffer(), gridElementsBufferSize));

    ownerOut.grid = GPUSimulation::GPUFemGrid2D(
        (float*)ownerOut.nodes.getHandle(),
        (int*)ownerOut.gridElements.getHandle(),
        grid->getNodesCount()
    );
    return EC::ErrorCode();
}

KDTree<GPUSimulation::GPUFemGrid2D> KDTreeGPUOwner::getTree() const {
    
    return KDTree<GPUSimulation::GPUFemGrid2D>(
        treeBBox,
        (KDNode*)&nodes.getHandle(),
        (int*)&leafTriangleIndexes.getHandle(),
        &grid
    );
}

GPUSimulation::GPUFemGrid2D KDTreeGPUOwner::getGrid() const {
    return grid;
}

void KDTreeBuilder::build(
    FemGrid2D* grid,
    std::vector<int>& leafTriangleIndexes,
    std::vector<KDNode>& nodes
) {
    // The formula for depth is taken from pbrt
    maxDepth = maxDepth > -1 ? maxDepth : std::min(29, (int)std::round(8 + 1.3f * std::log(grid->getElementsCount())));
    assert(grid->getElementSize() == 6);
    std::vector<int> indices(grid->getElementsCount());
    std::iota(indices.begin(), indices.end(), 0);
    build(grid, leafTriangleIndexes, nodes, indices, grid->getBBox(), 0, 0);
}

int KDTreeBuilder::build(
    FemGrid2D* grid,
    std::vector<int>& leafTriangleIndexes,
    std::vector<KDNode>& nodes,
    std::vector<int>& indices,
    const BBox2D& subtreeBBox,
    int axis,
    int level
) {
    if (level >= maxDepth || indices.size() <= minLeafSize) {
        const int numTriangles = indices.size();
        const int trianglesOffset = leafTriangleIndexes.size();
        std::move(indices.begin(), indices.end(), std::back_inserter(leafTriangleIndexes));
        const KDNode leaf = KDNode::makeLeaf(trianglesOffset, numTriangles);
        nodes.push_back(leaf);
        return nodes.size();
    }

    const float splitPoint = (subtreeBBox.getMin()[axis] + subtreeBBox.getMax()[axis]) * 0.5f;
    Point2D leftBBMax = subtreeBBox.getMax();
    leftBBMax[axis] = splitPoint;
    BBox2D leftBoundingBox(subtreeBBox.getMin(), leftBBMax);

    Point2D rightBBMin = subtreeBBox.getMin();
    rightBBMin[axis] = splitPoint;
    BBox2D rightBoundingBox(rightBBMin, subtreeBBox.getMax());

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
    const int rightNodeIndex = build(grid, leafTriangleIndexes, nodes, leftIndexes, leftBoundingBox, newAxis, level + 1);
    nodes[currentNodeIndex].makeInternal(axis, rightNodeIndex, splitPoint);
    return build(grid, leafTriangleIndexes, nodes, rightIndexes, rightBoundingBox, newAxis, level + 1);
}

};