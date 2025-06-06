cmake_minimum_required(VERSION 2.8.3)
project(alego)

set(CMAKE_BUILD_TYPE "Release")

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  nodelet
  tf
  pcl_conversions
  message_generation
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/deep/running_ws/devel/share/stable_gicp")

find_package(stable_gicp REQUIRED)

include_directories(${stable_gicp_INCLUDE_DIRS})


message("====================stable_gicp_INCLUDE_DIRS[1]===================== :  ${stable_gicp_INCLUDE_DIRS}")
message("====================stable_gicp_LIBRARIES[1]====================== :  ${stable_gicp_LIBRARIES}")

add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
)

catkin_package(INCLUDE_DIRS include)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
)

#add_library(imageProjection src/imageProjection.cpp)
#target_link_libraries(imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_library(laserOdometry src/laserOdometry.cpp)
#target_link_libraries(laserOdometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

#add_library(laserMapping src/laserMapping.cpp)
#target_link_libraries(laserMapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} gtsam)

#add_executable(IP src/IP.cpp)
#target_link_libraries(IP ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable(LO src/LO.cpp)
#target_link_libraries(LO ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(LM src/LM.cpp include/Scancontext/Scancontext.cpp)
target_link_libraries(LM ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS}  ${stable_gicp_LIBRARIES} rt gtsam)

