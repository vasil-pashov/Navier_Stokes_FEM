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

};