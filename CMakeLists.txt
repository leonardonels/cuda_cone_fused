cmake_minimum_required(VERSION 3.5)
project(cuda_cone_fused)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                      $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
                      $<$<COMPILE_LANGUAGE:CXX>:-march=native>
                      $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

# ----------------------------------------------------------------------------
# Optional GPU (CUDA) EKF backend. ON by Default.
# ----------------------------------------------------------------------------
option(USE_CUDA "Build the resident-state GPU EKF backend (needs CUDA toolkit)" ON)
set(CUDA_CONE_FUSED_CUDA_ARCH "87" CACHE STRING "CUDA arch for the GPU backend (Orin = 87)")

# ----------------------------------------------------------------------------
# Optional debug outputs: prints and topics. OFF by Default.
# ----------------------------------------------------------------------------
option(DEBUG "Compile in the debug-only cone publishers/topics" OFF)
if(DEBUG)
  message(STATUS "cuda_cone_fused: DEBUG publishers ENABLED (CONE_FUSED_DEBUG)")
else()
  message(STATUS "cuda_cone_fused: DEBUG publishers disabled")
endif()

if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  # Default arch for every target so CMP0104 doesn't complain about the
  # (non-CUDA) executable target that links the CUDA-enabled library.
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_CONE_FUSED_CUDA_ARCH}")
  message(STATUS "cuda_cone_fused: GPU backend ENABLED (CUDA arch ${CUDA_CONE_FUSED_CUDA_ARCH})")
else()
  message(STATUS "cuda_cone_fused: GPU backend disabled (CPU-only build)")
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(mmr_base REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
  include
)

# Build Node class
set(CUDA_CONE_FUSED_SOURCES
  src/${PROJECT_NAME}.cpp
  src/ekf_odom.cpp
  src/output/cone_map_publisher.cpp
  src/backend/cpu_backend.cpp
  src/backend/ekf_backend.cpp
)
if(USE_CUDA)
  list(APPEND CUDA_CONE_FUSED_SOURCES src/backend/cuda_backend.cu)
endif()

add_library(${PROJECT_NAME} ${CUDA_CONE_FUSED_SOURCES})

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_ros
  nav_msgs
  visualization_msgs
  mmr_base
)

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)

if(DEBUG)
  target_compile_definitions(${PROJECT_NAME} PUBLIC CONE_FUSED_DEBUG)
endif()

if(USE_CUDA)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_CONE_FUSED_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
  )
  target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CUDA)
  target_link_libraries(${PROJECT_NAME}
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
  )
endif()

add_executable(${PROJECT_NAME}_node src/${PROJECT_NAME}_node.cpp)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  mmr_base
)

target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}
  Eigen3::Eigen
)

# Install target
install(TARGETS
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch directory
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config directory
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
