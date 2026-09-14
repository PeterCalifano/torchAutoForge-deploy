cmake_minimum_required(VERSION 3.16)
project(ptafdeploy_ros)

option(PTAFDEPLOY_ENABLE_CUDA "Accepted for workspace-wide build helper compatibility" OFF)

# Resolve the core inference package and ROS 2 lifecycle/component dependencies.
find_package(ament_cmake REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(ptafdeploy REQUIRED)
find_package(ptafdeploy_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# Locate the imported ONNX Runtime library so installed ROS executables can find it at runtime.
set(PTAFDEPLOY_ROS_ORT_RUNTIME_DIR "")
foreach(
  location_property
  IMPORTED_LOCATION
  IMPORTED_LOCATION_RELWITHDEBINFO
  IMPORTED_LOCATION_RELEASE
  IMPORTED_LOCATION_DEBUG)
  get_target_property(onnxruntime_location onnxruntime::onnxruntime ${location_property})
  if(onnxruntime_location AND NOT onnxruntime_location MATCHES "-NOTFOUND$")
    get_filename_component(PTAFDEPLOY_ROS_ORT_RUNTIME_DIR "${onnxruntime_location}" DIRECTORY)
    break()
  endif()
endforeach()

# Build wrapper-safe tensor conversions independently for reuse by nodes and tests.
add_library(ptafdeploy_ros_conversions SHARED src/conversions.cpp)

target_compile_features(ptafdeploy_ros_conversions PUBLIC cxx_std_20)

target_include_directories(ptafdeploy_ros_conversions PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

ament_target_dependencies(ptafdeploy_ros_conversions ptafdeploy_interfaces)

target_link_libraries(ptafdeploy_ros_conversions autoforge_deploy::autoforge_deploy)

# Build and register the reusable lifecycle component around CModelFacade.
add_library(ptafdeploy_ros_component SHARED src/CInferenceLifecycleNode.cpp)

target_compile_features(ptafdeploy_ros_component PUBLIC cxx_std_20)

target_include_directories(ptafdeploy_ros_component PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

ament_target_dependencies(ptafdeploy_ros_component
  lifecycle_msgs
  rclcpp
  rclcpp_components
  rclcpp_lifecycle
  ptafdeploy_interfaces)

target_link_libraries(ptafdeploy_ros_component
  ptafdeploy_ros_conversions
  autoforge_deploy::autoforge_deploy)

if(PTAFDEPLOY_ROS_ORT_RUNTIME_DIR)
  set_target_properties(ptafdeploy_ros_component PROPERTIES
    INSTALL_RPATH "${PTAFDEPLOY_ROS_ORT_RUNTIME_DIR}")
endif()

rclcpp_components_register_nodes(ptafdeploy_ros_component
  "ptafdeploy_ros::CInferenceLifecycleNode")

# Provide the standalone entry point while retaining the component for composed launch.
add_executable(ptafdeploy_node src/ptafdeploy_node_main.cpp)

target_compile_features(ptafdeploy_node PRIVATE cxx_std_20)

ament_target_dependencies(ptafdeploy_node rclcpp)

target_link_libraries(ptafdeploy_node ptafdeploy_ros_component)

if(PTAFDEPLOY_ROS_ORT_RUNTIME_DIR)
  set_target_properties(ptafdeploy_node PROPERTIES
    INSTALL_RPATH "${PTAFDEPLOY_ROS_ORT_RUNTIME_DIR}")
endif()

# Keep tests focused on bridge/lifecycle behavior owned by this ROS 2 package.
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # Resolve target-owned fixture paths once for compile-time test configuration.
  set(PTAFDEPLOY_EXAMPLE_CONFIG_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/model_configs")
  set(PTAFDEPLOY_TEST_MODEL_CONFIG_PATH
    "${PTAFDEPLOY_EXAMPLE_CONFIG_DIR}/centroiding_traced_sample.ptafmodel")
  set(PTAFDEPLOY_TEST_YOLO_CONFIG_PATH
    "${PTAFDEPLOY_EXAMPLE_CONFIG_DIR}/yolov7_640x640.ptafmodel")
  set(PTAFDEPLOY_TEST_YOLO_MODEL_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../../models/onnx/yolov7_640x640.onnx")
  set(PTAFDEPLOY_TEST_YOLO_IMAGE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/object_detection_yoloV7/sample_data/horses.jpg")

  # Verify the wrapper-safe message/tensor conversion boundary.
  ament_add_gtest(test_conversions test/test_conversions.cpp)

  if(TARGET test_conversions)
    target_include_directories(test_conversions PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

    target_link_libraries(test_conversions ptafdeploy_ros_conversions)
  endif()

  # Verify lifecycle gating, configuration, inference, and failure propagation.
  ament_add_gtest(test_inference_lifecycle_node test/test_inference_lifecycle_node.cpp)

  if(TARGET test_inference_lifecycle_node)
    target_compile_definitions(test_inference_lifecycle_node PRIVATE
      "PTAFDEPLOY_TEST_MODEL_CONFIG=\"${PTAFDEPLOY_TEST_MODEL_CONFIG_PATH}\"")

    target_include_directories(test_inference_lifecycle_node PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include")

    ament_target_dependencies(test_inference_lifecycle_node
      lifecycle_msgs
      rclcpp
      rclcpp_lifecycle
      ptafdeploy_interfaces)

    target_link_libraries(test_inference_lifecycle_node ptafdeploy_ros_component)
  endif()

  # Add real YOLO lifecycle coverage only when its integration dependency is available.
  find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
  if(OpenCV_FOUND)
    ament_add_gtest(test_yolo_inference_lifecycle_node
      test/test_yolo_inference_lifecycle_node.cpp)

    if(TARGET test_yolo_inference_lifecycle_node)
      target_compile_definitions(test_yolo_inference_lifecycle_node PRIVATE
        "PTAFDEPLOY_TEST_YOLO_CONFIG=\"${PTAFDEPLOY_TEST_YOLO_CONFIG_PATH}\""
        "PTAFDEPLOY_TEST_YOLO_MODEL=\"${PTAFDEPLOY_TEST_YOLO_MODEL_PATH}\""
        "PTAFDEPLOY_TEST_YOLO_IMAGE=\"${PTAFDEPLOY_TEST_YOLO_IMAGE_PATH}\"")

      target_include_directories(test_yolo_inference_lifecycle_node PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${OpenCV_INCLUDE_DIRS})

      ament_target_dependencies(test_yolo_inference_lifecycle_node
        rclcpp
        rclcpp_lifecycle
        ptafdeploy_interfaces)

      target_link_libraries(test_yolo_inference_lifecycle_node
        ptafdeploy_ros_component
        ptafdeploy_ros_conversions
        autoforge_deploy::autoforge_deploy
        ${OpenCV_LIBS})
    endif()
  else()
    message(STATUS "OpenCV unavailable: skipping the optional ROS 2 YOLO integration test")
  endif()
endif()

# Install the public bridge API and all standalone/component runtime targets.
install(DIRECTORY include/ DESTINATION include)

install(TARGETS
  ptafdeploy_ros_conversions
  ptafdeploy_ros_component
  ptafdeploy_node
  EXPORT ptafdeploy_ros_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include)

# Export the same dependency surface required by downstream ROS 2 consumers.
ament_export_include_directories(include)

ament_export_targets(ptafdeploy_ros_targets HAS_LIBRARY_TARGET)

ament_export_dependencies(lifecycle_msgs
  ptafdeploy
  ptafdeploy_interfaces
  rclcpp
  rclcpp_components
  rclcpp_lifecycle)
ament_package()
