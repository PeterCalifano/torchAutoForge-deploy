# VERSION CONTROL 
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

#set(CMAKE_C_COMPILER gcc-11)
#set(CMAKE_CXX_COMPILER g++-11)
#set(CMAKE_CUDA_COMPILER nvcc)

# PROJECT DEFINITION
set(project_name "object_detection_yoloV7")
set(CMAKE_CXX_STANDARD 20)

# Assumes one has built the template_project with install in the root
set(DEPENDENCY_ROOT_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../../")

# Check if it exists 
if (NOT EXISTS "${DEPENDENCY_ROOT_PREFIX}/install")
    # Build and install it
    execute_process(
        COMMAND bash -c "cd ${DEPENDENCY_ROOT_PREFIX} && cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./install && cmake --build build --target install"
    )
endif()

if(NOT EXISTS "${DEPENDENCY_ROOT_PREFIX}/install/lib/cmake/")
    message(FATAL_ERROR "Please build and install the template_project first.")
endif()

project(${project_name} VERSION 0.1.0 LANGUAGES CXX)

# Tell cmake where to find the autoforge-deploy
set(autoforge-deploy_DIR "${DEPENDENCY_ROOT_PREFIX}/install/lib/cmake/autoforge-deploy")
find_package( autoforge-deploy CONFIG REQUIRED)
find_package( CUDAToolkit 12.0 REQUIRED) 
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn videoio)

#####
# Temporary for development. This should be set by the user or as environment variable.
set(HOME $ENV{HOME})
set(onnxruntime_DIR ${HOME}/devDir/ML-repos/onnxruntime/install/lib/cmake/onnxruntime)
set(onnxruntime_INCLUDE_DIRS ${HOME}/devDir/ML-repos/onnxruntime/install/include/onnxruntime) # DOUBT why the hell the cmake target does not include the include directory...?
find_package(onnxruntime REQUIRED)
#####

# Handle CUDA and setup
add_library(cuda_compile_settings INTERFACE)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")

    # Include directories
    target_include_directories(cuda_compile_settings INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

    set_property(TARGET cuda_compile_settings 
                 PROPERTY CUDA_ARCHITECTURES ${CUDAToolkit_ARCHITECTURES})

    target_compile_definitions(cuda_compile_settings INTERFACE __CUDA_ENABLED__=1)
    
    # Select CUDA libraries to link
    set(CUDA_LINK_LIBRARIES CUDA::cublas CUDA::cudart CUDA::curand CUDA::cuda_driver)

    set_target_properties(cuda_compile_settings PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # Enable separable compilation

    # Check if empty
    if (CUDA_LINK_LIBRARIES STREQUAL "")
      message(FATAL_ERROR "CUDA libraries not found.")
    else()
      message(STATUS "CUDA interface target linked against CUDA libraries: ${CUDA_LINK_LIBRARIES}")
    endif()

    #target_link_libraries(cuda_compile_settings INTERFACE ${CUDA_LINK_LIBRARIES})  # CUDA

endif()

# Alternatives: 
# 1. find_package(autoforge-deploy REQUIRED CONFIG PATHS "${DEPENDENCY_ROOT_PREFIX}/install/lib/cmake/")
# 2. When configuring the project: cmake -DCMAKE_PREFIX_PATH=/path/to/autoforge-deploy_install (...)
# 3. Set CMAKE_PREFIX_PATH in the environment: export CMAKE_PREFIX_PATH=/path/to/template_project_install

# Get source and header files
file(GLOB SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

# Add executable target
add_executable(${project_name} ${SRC_FILES} ${HEADER_FILES})

# Link target
target_link_libraries(${project_name} PRIVATE autoforge-deploy::autoforge-deploy cuda_compile_settings ${OpenCV_LIBS})
# Works: target_link_libraries(${project_name} PRIVATE autoforge-deploy::autoforge-deploy onnxruntime::onnxruntime cuda_compile_settings)
#target_include_directories(${project_name} PRIVATE ${template_project_INCLUDE_DIRS})

#message(WARNING "Note that template_project_INCLUDE_DIRS directories is empty because by default it is not built: ${template_project_INCLUDE_DIRS}\nInstead, includes directly transitions through the linekd target.")
#get_target_property(INC autoforge-deploy::autoforge-deploy INTERFACE_INCLUDE_DIRECTORIES)
#message(STATUS "Include directories: ${INC}\n")
