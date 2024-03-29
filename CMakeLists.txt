cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(light2d_cuda LANGUAGES CXX CUDA)

find_package(CUDA 9.0 REQUIRED)

# ompiling options
set(CMAKE_CXX_STANDARD 17)
if(MSVC)
	set(EMBREE_MAX_ISA "SSE4.2" CACHE STRING "Selects highest ISA to support.")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox /openmp -d2SSAOptimizer-") # -d2SSAOptimizer- for MSVC bug
else()
	set(CXX_WARNINGS "-Wall -Wextra -Wpointer-arith -Wcast-align -fstrict-aliasing")
	if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
		set(CXX_WARNINGS "${CXX_WARNINGS} -Wno-maybe-uninitialized")
	endif()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_WARNINGS} -fvisibility-inlines-hidden -O3 -fopenmp")

	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
endif()

# CUDA flags
set(CMAKE_CUDA_STANDARD 11)

include_directories(
	src/thirdparty
	src/thirdparty/glew-2.1.0/include
	src/thirdparty/glfw-3.2.1/include
	src/thirdparty/glm-0.9.8.5
	${CUDA_INCLUDE_DIRS}
)

set(LIBRARIES
	glfw
	glew_s
	${CUDA_curand_LIBRARY}
	${CUDA_LIBRARIES}
)

add_executable(basic src/basic.cu)
target_link_libraries(basic ${LIBRARIES})

add_executable(basic_interactive src/basic_interactive.cu)
target_link_libraries(basic_interactive ${LIBRARIES})

add_executable(csg src/csg.cu)
target_link_libraries(csg ${LIBRARIES})

add_executable(csg_interactive src/csg_interactive.cu)
target_link_libraries(csg_interactive ${LIBRARIES})

add_executable(shapes src/shapes.cu)
target_link_libraries(shapes ${LIBRARIES})

add_executable(shapes_interactive src/shapes_interactive.cu)
target_link_libraries(shapes_interactive ${LIBRARIES})



option(BUILD_UTILS "utilities" OFF)
add_subdirectory(src/thirdparty/glew-2.1.0/build/cmake)

option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
add_subdirectory(src/thirdparty/glfw-3.2.1)
