#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include "assembly.h"
#include "grid.h"
#include "error_code.h"
#include "expression.h"

int main() {
    // NSFem::FemGrid2D grid;
    // grid.loadJSON("/home/vasil/Documents/FMI/Магистратура/Дипломна/CPP/Assests/mesh_small.json");
    // NSFem::NavierStokesAssembly assembler(std::move(grid), 0.01, 0.001);
    // assembler.assemble();
    const char* exprStr = "3 + 2 * 5 * -1";
    NSFem::Expression expr;
    EC::ErrorCode ec = expr.init(exprStr);
    if(ec.hasError()) {
        assert(false);
    }
}