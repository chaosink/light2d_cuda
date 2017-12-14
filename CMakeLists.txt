cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(light2d_cuda LANGUAGES CXX CUDA)

find_package(CUDA 9.0 REQUIRED)
find_package(OpenCV REQUIRED)

# C++ flags
if(MSVC)
	add_definitions(-DCONSTEXPR=const -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
else()
	add_definitions(-DCONSTEXPR=constexpr)
	set(CXX_WARNINGS "-Wall -Wextra -Wpointer-arith -Wcast-align -fstrict-aliasing -Wno-unused-local-typedefs -Wno-misleading-indentation -Wno-maybe-uninitialized -Wno-int-in-bool-context -Wno-implicit-fallthrough")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_WARNINGS} -fvisibility-inlines-hidden -O3")
endif()

# CUDA flags
set(CMAKE_CUDA_STANDARD 11)
string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_30,code=sm_30 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_32,code=sm_32 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_35,code=sm_35 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_37,code=sm_37 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_52,code=sm_52 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_53,code=sm_53 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=sm_60 ")
string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_62,code=sm_62 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70 ")

include_directories(
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
	${OPENGL_gl_LIBRARY}
	${OpenCV_LIBS})

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
