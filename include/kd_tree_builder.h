#pragma once
#include "gpu_host_common.h"
#include "kd_tree_common.cuh"
#include "kd_tree.cuh"
#include "gpu_grid.cuh"

namespace NSFem {

class FemGrid2D;

class KDTreeGPUOwner {
public:
    friend class KDTreeCPUOwner;
    KDTreeGPUOwner() = default;
    KDTreeGPUOwner(KDTreeGPUOwner&&) = default;
    KDTreeGPUOwner& operator=(KDTreeGPUOwner&&) = default;
    KDTreeGPUOwner(const KDTreeGPUOwner&) = delete;
    KDTreeGPUOwner& operator=(const KDTreeGPUOwner&) = delete;
    KDTree<GPUSimulation::GPUFemGrid2D> getTree() const;
    GPUSimulation::GPUFemGrid2D getGrid() const;
private:
    GPU::GPUBuffer nodes;
    GPU::GPUBuffer leafTriangleIndexes;
    GPU::GPUBuffer gridNodes;
    GPU::GPUBuffer gridElements;
    GPUSimulation::GPUFemGrid2D grid;
    BBox2D treeBBox;
};

class KDTreeCPUOwner {
public:
    friend class KDTreeBuilder;
    KDTreeCPUOwner() = default;
    KDTreeCPUOwner(KDTreeCPUOwner&&) = default;
    KDTreeCPUOwner& operator=(KDTreeCPUOwner&&) = default;
    KDTreeCPUOwner(const KDTreeCPUOwner&) = delete;
    KDTreeCPUOwner& operator=(const KDTreeCPUOwner&) = delete;
    KDTree<FemGrid2D> getTree() const;
    EC::ErrorCode upload(KDTreeGPUOwner& ownerOut) const;
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

}