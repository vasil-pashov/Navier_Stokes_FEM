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
    const std::string outFolder("/home/vasil/Documents/FMI/Магистратура/Дипломна/CPP/Assests/mesh_small_out");
    NSFem::NavierStokesAssembly<NSFem::P2, NSFem::P1> assembler(std::move(grid), 0.01, 0.001, outFolder);
    assembler.solve(0.1);
}
