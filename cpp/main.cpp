#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include "assembly.h"
#include "grid.h"
#include "error_code.h"
#include "expression.h"

int main(int nargs, char** cargs) {
    if(nargs == 1) {
        printf("Missing path to a FEM grid.");
        return 1;
    } else if(nargs == 2) {
        printf("Missing output folder path.");
        return 1;
    } else {
        NSFem::FemGrid2D grid;
        grid.loadJSON(cargs[1]);
        NSFem::NavierStokesAssembly<NSFem::P2, NSFem::P1> assembler(std::move(grid), 0.001, 0.001, cargs[2]);
        // assembler.solve(1.f);
        // assembler.semiLagrangianSolve(10.f);
    }

}
