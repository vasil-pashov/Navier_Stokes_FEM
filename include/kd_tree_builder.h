#pragma once
#include "gpu_host_common.h"
#include "kd_tree_common.cuh"

namespace NSFem {

class FemGrid2D;
class KDTree;

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

}