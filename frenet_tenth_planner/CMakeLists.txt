cmake_minimum_required(VERSION 3.10)
project(frenet_tenth_planner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(frenet_core
  src/cubic_spline_planner.cpp
  src/polynomials.cpp
  src/frenet_optimal_trajectory.cpp
  src/track_loader.cpp
)

target_include_directories(frenet_core PUBLIC include)
target_link_libraries(frenet_core yaml-cpp)

add_executable(frenet_planner_node src/frenet_planner_node.cpp)
target_link_libraries(frenet_planner_node frenet_core)
ament_target_dependencies(frenet_planner_node
  rclcpp nav_msgs geometry_msgs visualization_msgs tf2 tf2_geometry_msgs)

install(TARGETS
  frenet_core
  frenet_planner_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch config scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
