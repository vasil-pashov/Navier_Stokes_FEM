#include <nlohmann/json.hpp>
#include <fstream>>
#include "assembly.h"

int parseScene(const char* filePath, NSFem::NavierStokesAssembly& ns) {
    std::ifstream jsonFile(filePath);
    nlohmann::basic_json data = nlohmann::json::parse(jsonFile, nullptr, false);
    if(data.is_discarded()) {
        return 1;
    }
    ns.grid.elementsCount = data["elementsCount"];
    ns.grid.nodesCount = data["nodesCount"];
    ns.grid.elementSize = data["elementSize"];
    ns.grid.elements.reserve(ns.grid.elementsCount * ns.grid.elementSize);
    ns.grid.nodes.reserve(ns.grid.nodesCount * 2);
    for(int i = 0; i < ns.grid.nodesCount; ++i) {
        assert(data["nodes"][i].size() == 2 && "Expected 2D grid. All nodes must be of (x, y) pairs of size 2");
        nodes.insert(nodes.end(), data["nodes"][i].begin(), data["nodes"][i].end());
    }
    for(int i = 0; i < ns.grid.elementsCount; ++i) {
        assert(ns.grid.elementSize == data["elements"][i].size() && "Mismatch between declared element size and actual element size");
        ns.grid.elements.insert(ns.grid.elements.end(), data["elements"][i].begin(), data["elements"][i].end());
    }
    assert(ns.grid.nodes.size() == nodesCount * 2);
    assert(ns.grid.elements.size() == elementSize * elementsCount);
    return 0;
}