#include <grid.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cassert>
#include "error_code.h"

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
        assert(elementSize == static_cast<int>(data["elements"][i].size()) && "Mismatch between declared element size and actual element size");
        elements.insert(elements.end(), data["elements"][i].begin(), data["elements"][i].end());
    }
    assert(static_cast<int>(nodes.size()) == nodesCount * 2);
    assert(static_cast<int>(elements.size()) == elementSize * elementsCount);

    const int velocityDirichletSize = data["uDirichlet"].size();
    velocityDirichlet.resize(velocityDirichletSize);
    EC::ErrorCode error;
    for(int i = 0; i < velocityDirichletSize; ++i) {
        velocityDirichlet[i].nodeIndexes.insert(
            velocityDirichlet[i].nodeIndexes.end(),
            data["uDirichlet"][i]["nodes"].begin(),
            data["uDirichlet"][i]["nodes"].end()
        );
        const std::string& u = data["uDirichlet"][i]["u"];
        const std::string& v = data["uDirichlet"][i]["v"];
        error = velocityDirichlet[i].u.init(u.c_str());
        if(error.hasError()) {
            assert(false);
            return 1;
        }
        error = velocityDirichlet[i].v.init(v.c_str());
        if(error.hasError()) {
            assert(false);
            return 1;
        }
    }

    const int pressureDirichletSize = data["pDirichlet"].size();
    pressureDirichlet.resize(pressureDirichletSize);
    for(int i = 0; i < pressureDirichletSize; ++i) {
        pressureDirichlet[i].nodeIndexes.insert(
            pressureDirichlet[i].nodeIndexes.begin(),
            data["pDirichlet"][i]["nodes"].begin(),
            data["pDirichlet"][i]["nodes"].end()
        );
        const std::string& p = data["pDirichlet"][i]["p"];
        error = pressureDirichlet[i].p.init(p.c_str());
        if(error.hasError()) {
            assert(false);
            return 1;
        }
    }

    return 0;
}

void FemGrid2D::VelocityDirichlet::eval(
    const std::unordered_map<char, float>* variables,
    float& outU,
    float& outV
) const {
    u.evaluate(variables, outU);
    v.evaluate(variables, outV);
}

void FemGrid2D::PressureDirichlet::eval(
    const std::unordered_map<char, float>* variables,
    float& outP
) const {
    p.evaluate(variables, outP);
}

}