# CMake configuration for C++ tests
cmake_minimum_required(VERSION 3.18)
project(PlanarCUDA_Tests CUDA CXX)

# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Include project source directories
include_directories(${CMAKE_SOURCE_DIR}/../../src)

# Test source files
set(TEST_SOURCES
    test_k_value_calculator.cpp
    test_cost_functions.cpp
    # test_sa_optimizer.cpp
    # test_gpu_memory_manager.cpp
)

# Create test executable
add_executable(run_tests ${TEST_SOURCES})

# Link libraries
target_link_libraries(run_tests
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    ${CUDA_LIBRARIES}
    pthread
)

# Enable C++14
set_target_properties(run_tests PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

# Add test
enable_testing()
add_test(NAME AllTests COMMAND run_tests)

# Print configuration
message(STATUS "Google Test: ${GTEST_LIBRARIES}")
message(STATUS "CUDA: ${CUDA_LIBRARIES}")
