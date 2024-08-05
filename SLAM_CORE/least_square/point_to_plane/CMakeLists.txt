cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)
project(slam_core)
add_executable(slam_core  slam_core.cpp)

find_package(PCL 1.10 REQUIRED )
set(PCL_INCLUDE_DIRS /usr/include/pcl-1.10)  #PCL 库

add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")  #eigen3 库

target_link_libraries (slam_core ${PCL_LIBRARIES})
install(TARGETS slam_core RUNTIME DESTINATION bin)

