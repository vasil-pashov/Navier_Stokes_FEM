#include "grid.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cassert>

namespace NSFem {

int FemGrid2D::loadJSON(const char* filePath) {
    std::ifstream jsonFile(filePath);
    nlohmann::basic_json data = nlohmann::json::parse(jsonFile, nullptr, false);
    if(data.is_discarded()) {
        return 1;
    }
    elementsCount = data["elementsCount"];
    nodesCount = data["nodesCount"];
    elementSize = data["elementSize"];
    elements.reserve(elementsCount * elementSize);
    nodes.reserve(nodesCount * 2);
    for(int i = 0; i < nodesCount; ++i) {
        assert(data["nodes"][i].size() == 2 && "Expected 2D grid. All nodes must be of (x, y) pairs of size 2");
        nodes.insert(nodes.end(), data["nodes"][i].begin(), data["nodes"][i].end());
    }
    for(int i = 0; i < elementsCount; ++i) {
        assert(elementSize == data["elements"][i].size() && "Mismatch between declared element size and actual element size");
        elements.insert(elements.end(), data["elements"][i].begin(), data["elements"][i].end());
    }
    assert(nodes.size() == nodesCount * 2);
    assert(elements.size() == elementSize * elementsCount);
    return 0;
}

}