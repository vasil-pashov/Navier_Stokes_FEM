#include <cstdio>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <filesystem>
#include "assembly.h"
#include "grid.h"
#include "error_code.h"
#include "expression.h"
#include "cmd_line_parser.h"
#include <cpp_tm/cpp_tm.h>

enum SolverMethod {
    FEM,
    SemiLagrangianFEM
};

template<typename VeloctyShape, typename PressureShape>
void rungeEstimates(
    NSFem::NavierStokesAssembly<VeloctyShape, PressureShape>& solver,
    std::string baseFolder,
    SolverMethod method,
    bool solve
) {
    baseFolder += method == FEM ? "FEM" : "SemiLagrangian";
    constexpr double dt[3] = {0.01, 0.001, 0.0001};
    constexpr double totalTime = 1;
    constexpr int totalTimeSteps = totalTime / dt[0];
    constexpr int numNestedGrids = 3;
    std::string outPaths[numNestedGrids];
    for(int i = 0; i < numNestedGrids; ++i) {
        outPaths[i] = baseFolder + "_" + std::to_string(i);
        if(solve) {
            std::filesystem::create_directory(outPaths[i]);
            solver.setTimeStep(dt[i]);
            solver.setOutputDir(outPaths[i]);
            if(method == SolverMethod::FEM) {
                solver.solve(totalTime);
            } else if(method == SolverMethod::SemiLagrangianFEM) {
                solver.semiLagrangianSolve(totalTime);
            }
        }
    }

    std::vector<double> commonTimeVelocity[3][totalTimeSteps];
    int factor = 1;
    for(int i = 0; i < numNestedGrids; ++i) {
        for(int j = 0, k = 0; k < totalTimeSteps; j+=factor, k++) {
            const std::string cachePath(outPaths[i] + std::string("/out_") + std::to_string(j) + std::string(".json"));
            std::fstream cacheStream(cachePath);
            assert(cacheStream.is_open());
            nlohmann::basic_json cacheJSON;
            cacheStream >> cacheJSON;
            const int numNodes = cacheJSON["u"].size();
            commonTimeVelocity[i][k].resize(numNodes * 2);
            std::copy_n(cacheJSON["u"].begin(), numNodes, commonTimeVelocity[i][k].begin());
            std::copy_n(cacheJSON["v"].begin(), numNodes, commonTimeVelocity[i][k].begin() + numNodes);
        }
        factor *= 10;
    }
    
    std::vector<double> convergenceOrder(totalTimeSteps);
    for(int i = 0; i < totalTimeSteps; ++i) {
        double normVV2 = 0, normV2V3 = 0;
        for(int j = 0; j < commonTimeVelocity[0][i].size(); ++j) {
            const double diff1 = commonTimeVelocity[0][i][j] - commonTimeVelocity[1][i][j];
            const double diff2 = commonTimeVelocity[1][i][j] - commonTimeVelocity[2][i][j]; 
            normVV2 += diff1 * diff1;
            normV2V3 += diff2 * diff2;
        }
        normVV2 = std::sqrt(normVV2);
        normV2V3 = std::sqrt(normV2V3);
        
        if(normVV2 == 0 || normVV2 == 0) {
            continue;
        }

        convergenceOrder[i] = std::log10(std::abs(normVV2 / normV2V3));
        std::cout<<"Convergence: "<<convergenceOrder[i]<<" error: "<<std::pow(10, convergenceOrder[i]) * normVV2 / (std::pow(10, convergenceOrder[i])-1)<<"\n";
    }
}

int main(int nargs, char** cargs) {
    CMD::CommandLineArgs argParse;
    argParse.addParam(
        "sceneFile",
        "Path to the file describing the simulation",
        CMD::CommandLineArgs::Type::String,
        true
    );
    argParse.addParam(
        "numThreads",
        "The number of threads which the simulator should use. All threads by default.",
        CMD::CommandLineArgs::Type::Int,
        false
    );
    argParse.addParam(
        "outPath",
        "Path to a folder where the resulting caches and images will be saved",
        CMD::CommandLineArgs::Type::String,
        false
    );

    if(nargs == 2 && strcmp(cargs[1], "-help") == 0) {
        argParse.print(stdout);
        return 0;
    }

    EC::ErrorCode error = argParse.parse(nargs, cargs);
    if(error.hasError()) {
        fprintf(stderr, "[Error] %s\n", error.getMessage());
        fprintf(stderr, "Supported options\n");
        argParse.print(stderr);
        return 1;
    }
    unsigned int numThreads = [&]() -> unsigned int {
        const int* threads = argParse.getIntVal("numThreads");
        if(threads != nullptr) {
            return *threads;
        } else {
            return std::thread::hardware_concurrency() - 1;
        }
    }();

    CPPTM::ThreadManager tm(numThreads);

    NSFem::NavierStokesAssembly<NSFem::P2, NSFem::P1> assembler(tm);
    error = assembler.init(
        argParse.getStringVal("sceneFile"),
        argParse.getStringVal("outPath")
    );
    if(error.hasError()) {
        fprintf(stderr, "[Error] %s\n", error.getMessage());
        return 1;
    }
    assembler.semiLagrangianSolve();
    return 0;

}
