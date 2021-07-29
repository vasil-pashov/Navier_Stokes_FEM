#pragma once
#include <vector>
#include <limits>
#include "gpu_host_common.h"
#include "kd_tree_common.cuh"

namespace NSFem {

class FemGrid2D;

class KDTree {
public:
    friend class KDTreeBuilder;
    /// @brief Default constrict the tree with empty bounding box and no nodes.
    KDTree();
    KDTree(
        BBox2D bbox,
        const KDNode* nodes,
        const int* leafTriangleIndexes,
        const FemGrid2D* grid
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
    int findElement(const Point2D& point, real& xi, real& eta, int& closestFEMNodeIndex) const;
private:
    int getRootIndex() const {
        return 0;
    }

    /// Compare all of the points in the leaf and check if any of the points has distance to the
    /// given point less than minDistSq. If so update minDistSq and the index of the femNode in the mesh
    void nearestNeghbourProcessLeaf(
        const Point2D& point,
        const KDNode& node,
        real& minDistSq,
        int& closestFEMNodeIndex
    ) const;

    BBox2D bbox;
    const KDNode* nodes;
    const int* leafTriangleIndexes;
    const FemGrid2D *grid;
};

class KDTreeGPUOwner {
public:
    friend class KDTreeCPUOwner;
    KDTreeGPUOwner() = default;
    KDTreeGPUOwner(KDTreeGPUOwner&&) = default;
    KDTreeGPUOwner& operator=(KDTreeGPUOwner&&) = default;
    KDTreeGPUOwner(const KDTreeGPUOwner&) = delete;
    KDTreeGPUOwner& operator=(const KDTreeGPUOwner&) = delete;
private:
    GPU::GPUBuffer nodes;
    GPU::GPUBuffer leafTriangleIndexes;
    GPU::GPUBuffer gridNodes;
    GPU::GPUBuffer gridElements;
};

class KDTreeCPUOwner {
public:
    friend class KDTreeBuilder;
    KDTreeCPUOwner() = default;
    KDTreeCPUOwner(KDTreeCPUOwner&&) = default;
    KDTreeCPUOwner& operator=(KDTreeCPUOwner&&) = default;
    KDTreeCPUOwner(const KDTreeCPUOwner&) = delete;
    KDTreeCPUOwner& operator=(const KDTreeCPUOwner&) = delete;
    KDTree getTree() const;
    EC::ErrorCode upload(KDTreeGPUOwner& ownerOut);
protected:
    std::vector<KDNode> nodes;
    std::vector<int> leafTriangleIndexes;
    FemGrid2D* grid;
    BBox2D treeBBox;
};

class KDTreeBuilder {
public:
    KDTreeBuilder();
    KDTreeBuilder(int maxDepth, int minLeafSize);
    KDTreeCPUOwner buildCPUOwner(FemGrid2D* grid);
private:
    void build(
        FemGrid2D* grid,
        std::vector<int>& leafTriangleIndexes,
        std::vector<KDNode>& nodes
    );
    int build(
        FemGrid2D* grid,
        std::vector<int>& leafTriangleIndexes,
        std::vector<KDNode>& nodes,
        std::vector<int>& indices,
        const BBox2D& subtreeBBox,
        int axis,
        int level
    );
    int maxDepth;
    int minLeafSize;
};


};