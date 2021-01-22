cmake_minimum_required(VERSION 2.8)
project(face)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)

SET(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")  #c++11
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++0x.") #c++0x
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# include .h headers
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/stb)

# Ceres solver
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
set(EIGEN3_INCLUDE_DIRS /usr/local/include/eigen3/)
include_directories(${EIGEN3_INCLUDE_DIR})

# trimesh2
set(Trimesh2_DIR ${PROJECT_SOURCE_DIR}/3rdparty/trimesh2)
set(Trimesh2_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/3rdparty/trimesh2/include/)
set(Trimesh2_LIB_DIRS ${PROJECT_SOURCE_DIR}/3rdparty/trimesh2/lib.Linux64/libtrimesh.a)
include_directories(${Trimesh2_INCLUDE_DIRS})


add_library(${PROJECT_NAME} STATIC 
3rdparty/stb/stb_image.c
render_utils.cpp
cook_torrance.cpp
data_loader.cpp
spherical_harmonics.cpp
visualizer.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS} 
${Trimesh2_LIB_DIRS} 
${CERES_LIBRARIES} 
)

add_executable(main main.cpp)
target_link_libraries(main ${PROJECT_NAME})

#add_executable(test_ceres test_ceres.cpp)
#target_link_libraries(test_ceres ${PROJECT_NAME})

#add_executable(test_sh test_sh.cpp)
#target_link_libraries(test_sh ${PROJECT_NAME})

#add_executable(test_voronoi test_voronoi.cpp)
#target_link_libraries(test_voronoi ${PROJECT_NAME})
