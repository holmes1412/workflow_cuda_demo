cmake_minimum_required(VERSION 3.10)

project(cuda_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)
find_package(Workflow REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS} ${WORKFLOW_INCLUDE_DIR})

set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80)

set(SOURCES matrix_mul_demo.cu)

add_executable(cuda_demo ${SOURCES})

set_target_properties(cuda_demo PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	LINKER_LANGUAGE CUDA
)

target_link_libraries(cuda_demo ${CUDAToolkit_LIBRARIES} ${WORKFLOW_LIB_DIR}/libworkflow.a)

