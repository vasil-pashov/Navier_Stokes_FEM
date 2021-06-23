#include <grid.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cassert>
#include "error_code.h"

namespace NSFem {

EC::ErrorCode FemGrid2D::loadJSON(const char* filePath) {
    bbox.reset();
    std::ifstream jsonFile(filePath);
    if(!jsonFile.is_open()) {
        return EC::ErrorCode(1, "Cannot open JSON file: %s", filePath);
    }
    auto data = nlohmann::json::parse(jsonFile, nullptr, false);
    if(data.is_discarded()) {
        return EC::ErrorCode(2, "Cannot parse JSON file: %s", filePath);
    }
    elementsCount = data["elementsCount"];
    velocityNodesCount = data["velocityNodesCount"];
    assert(velocityNodesCount > 0);
    pressureNodesCount = data["pressureNodesCount"];
    assert(pressureNodesCount > 0);
    elementSize = data["elementSize"];
    elements.reserve(elementsCount * elementSize);
    nodes.reserve(velocityNodesCount * 2);
    for(int i = 0; i < velocityNodesCount; ++i) {
        assert(data["nodes"][i].size() == 2 && "Expected 2D grid. All nodes must be of (x, y) pairs of size 2");
        Point2D node(data["nodes"][i][0], data["nodes"][i][1]);
        bbox.expand(node);
        nodes.push_back(node.x);
        nodes.push_back(node.y);
    }
    for(int i = 0; i < elementsCount; ++i) {
        assert(elementSize == static_cast<int>(data["elements"][i].size()) && "Mismatch between declared element size and actual element size");
        elements.insert(elements.end(), data["elements"][i].begin(), data["elements"][i].end());
    }
    assert(static_cast<int>(nodes.size()) == velocityNodesCount * 2);
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
            return error;
        }
        error = velocityDirichlet[i].v.init(v.c_str());
        if(error.hasError()) {
            return error;
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
            return error; 
        }
    }

    return EC::ErrorCode();
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