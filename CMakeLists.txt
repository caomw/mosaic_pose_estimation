# http://ros.org/doc/groovy/api/catkin/html/user_guide/supposed.html
cmake_minimum_required(VERSION 2.8.3)
project(mosaic_pose_estimation)
# Load catkin and all dependencies required for this package
# TODO: remove all from COMPONENTS that are not catkin packages.
find_package(catkin REQUIRED COMPONENTS 
	roscpp 
	rospy 
	roslib 
	sensor_msgs	
	image_transport 
	image_geometry 
	image_proc
	cv_bridge 
	tf 
	nav_msgs 
	bag_tools )


find_package(OpenCV REQUIRED)

if (UNIX)
  find_library(YAMLCPP_LIBRARY NAMES yaml-cpp)
  find_file(YAMLCPP_HAVE_H yaml-cpp/yaml.h )
  find_path(YAMLCPP_H_INCLUDE_DIR yaml-cpp/yaml.h )
  SET(YAMLCPP_INCLUDE_DIR ${YAMLCPP_H_INCLUDE_DIR}/yaml-cpp)
else()
  find_library(
    YAMLCPP_LIBRARY
    NAMES yamlcpp.lib
    NO_DEFAULT_PATH 
    PATHS 
    $ENV{YAMLCPP_ROOT}/lib
  )
  find_file(YAMLCPP_HAVE_H include/yaml.h PATHS $ENV{YAMLCPP_ROOT} NO_DEFAULT_PATH)
  find_path(YAMLCPP_H_INCLUDE_DIR include/yaml.h PATHS $ENV{YAMLCPP_ROOT} NO_DEFAULT_PATH)
  SET(YAMLCPP_INCLUDE_DIR ${YAMLCPP_H_INCLUDE_DIR}/include)
endif(UNIX)

SET(YAMLCPP_FOUND FALSE)
if (YAMLCPP_HAVE_H)
  if (YAMLCPP_LIBRARY)
    SET(YAMLCPP_FOUND TRUE)
  endif (YAMLCPP_LIBRARY)
endif (YAMLCPP_HAVE_H)

if(NOT YAMLCPP_FOUND)
  message(STATUS "YamlCpp Can't be found")
  break()
else()
  message(STATUS "YamlCpp Library: " ${YAMLCPP_LIBRARY} )
  message(STATUS "YamlCpp Header: " ${YAMLCPP_INCLUDE_DIR} )
endif(NOT YAMLCPP_FOUND)

catkin_package()

catkin_python_setup()

include_directories(${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${YAMLCPP_INCLUDE_DIR} include)

add_executable(mosaic_processor_node src/mosaic_processor_node.cpp)
add_executable(process_bag src/process_bag.cpp)
add_library(mosaic_processor src/mosaic_processor.cpp)

target_link_libraries(mosaic_processor ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
target_link_libraries(mosaic_processor_node mosaic_processor ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
target_link_libraries(process_bag mosaic_processor ${YAMLCPP_LIBRARY} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

install(PROGRAMS scripts/getOdometryDatFile.py
         DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
