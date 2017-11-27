cmake_minimum_required(VERSION 3.5.1)
project(light2d_cuda)

find_package(CUDA)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV)

# C++ list
set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -Wall")

# CUDA list
list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_30,code=sm_30)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_50,code=sm_50)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=sm_52)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=compute_52)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_60,code=sm_60)
list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_61,code=sm_61)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_62,code=sm_62)

set(LIBRARIES
	glfw
	GLEW
	${CUDA_curand_LIBRARY}
	${CUDA_LIBRARIES}
	${OPENGL_gl_LIBRARY}
	${OpenCV_LIBS})

cuda_add_executable(basic src/basic.cu)
target_link_libraries(basic ${LIBRARIES})

cuda_add_executable(basic_interactive src/basic_interactive.cu)
target_link_libraries(basic_interactive ${LIBRARIES})

cuda_add_executable(csg src/csg.cu)
target_link_libraries(csg ${LIBRARIES})

cuda_add_executable(csg_interactive src/csg_interactive.cu)
target_link_libraries(csg_interactive ${LIBRARIES})

cuda_add_executable(shapes src/shapes.cu)
target_link_libraries(shapes ${LIBRARIES})

cuda_add_executable(shapes_interactive src/shapes_interactive.cu)
target_link_libraries(shapes_interactive ${LIBRARIES})
