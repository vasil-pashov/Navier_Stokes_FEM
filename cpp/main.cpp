#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include "assembly.h"
#include "grid.h"
#include "error_code.h"
#include "expression.h"

int main() {
    NSFem::FemGrid2D grid;
    grid.loadJSON("/home/vasil/Documents/FMI/Магистратура/Дипломна/CPP/Assests/mesh_small.json");
    NSFem::NavierStokesAssembly<NSFem::P2, NSFem::P1> assembler(std::move(grid), 0.01, 0.001);
    assembler.solve(1.0f);
}
