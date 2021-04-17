cmake_minimum_required(VERSION 3.19.0)
project(
	NSFemSolver
	VERSION 0.1.0
	DESCRIPTION "Finite element method solver for the Navier-Stokes equations." 
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(sparse_matrix_math REQUIRED CONFIG)

set(CPP
	cpp/main.cpp
	cpp/grid.cpp
	cpp/assembly.cpp
)

set(HEADERS
	include/grid.h
	include/assembly.h
)

add_executable(${PROJECT_NAME} ${CPP} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} INTERFACE sparse_matrix_math::sparse_matrix_math)
