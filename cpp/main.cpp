#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include "assembly.h"
#include "grid.h"

int main() {
    NSFem::FemGrid2D grid;
    grid.loadJSON("/home/vasil/Documents/FMI/Магистратура/Дипломна/CPP/Assests/DFG_NS.json");
    NSFem::NavierStokesAssembly assembler(std::move(grid), 0.01, 0.001);
    assembler.assemble();
}