#pragma once
#include <vector>
#include <limits>
#include "grid.h"
#include "kd_tree_common.cuh"

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
    int getRootIndex() const {
        return 0;
    }

    /// Compare all of the points in the leaf and check if any of the points has distance to the
    /// given point less than minDistSq. If so update minDistSq and the index of the femNode in the mesh
    void nearestNeghbourProcessLeaf(const Point2D& point, const KDNode& node, real& minDistSq, int& closestFEMNodeIndex) const;

    std::vector<KDNode> nodes;
    std::vector<int> leafTriangleIndexes;
    BBox2D bbox;
    FemGrid2D *grid;
};

class TriangleKDTreeBuilder {
public:
    TriangleKDTreeBuilder();
    TriangleKDTreeBuilder(int maxDepth, int minLeafSize);
    TriangleKDTree build(FemGrid2D* grid);
private:
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