cmake_minimum_required(VERSION 3.8)
project(snap_map_icp)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(angles REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(laser_geometry REQUIRED)



add_executable(snap_map src/snap_map_icp.cpp)
ament_target_dependencies(snap_map geometry_msgs laser_geometry pcl_ros pcl_conversions std_msgs std_srvs nav_msgs nav2_msgs tf2 sensor_msgs rclcpp Eigen3 angles tf2_ros tf2_geometry_msgs)
#allow target to access header files defined in include
target_include_directories( snap_map
 PUBLIC
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
   $<INSTALL_INTERFACE:include>)

install(TARGETS
  snap_map
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
   find_package(ament_lint_auto REQUIRED)
   set(ament_cmake_copyright_FOUND TRUE)
   ament_lint_auto_find_test_dependencies()
 endif()

ament_package()
