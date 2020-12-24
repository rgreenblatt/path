set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_QT_COMPILER clang++)
set(CMAKE_CUDA_COMPILER clang++)

cmake_policy(SET CMP0048 NEW)
project(path VERSION 0.1 LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.18)

set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

add_compile_options(-Wall -Wextra -fsized-deallocation -g)

# use libc++ for this project and submodules
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -stdlib=libc++")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

# unfortunately required with c++20
add_compile_options(-isystem ${CMAKE_SOURCE_DIR}/system/include/)

# required for Eigen
add_compile_options(-Wno-deprecated-copy)
add_compile_options(-Wno-deprecated-anon-enum-enum-conversion)

add_compile_options(-fconstexpr-steps=80000000 -fbracket-depth=512)
# I think -ffast-math is enabled by -Ofast, but the documentation isn't
# perfectly clear (certainly most options of -ffast-math are enabled by
# -Ofast, but I'm not sure about the cuda specific ones)
set(RELEASE_ARGS "-Ofast;-ffast-math;-march=native;-DNDEBUG")
# set(RELEASE_ARGS "-O3;-UNDEBUG")
add_compile_options("$<$<CONFIG:RELEASE>:${RELEASE_ARGS}>")
add_compile_options("$<$<CONFIG:RELWITHDEBINFO>:${RELEASE_ARGS}>")
add_compile_options("$<$<CONFIG:DEBUG>:-ftrapv>")

option (FORCE_COLORED_OUTPUT
  "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if (${FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
       add_compile_options (-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
       add_compile_options (-fcolor-diagnostics)
    endif ()
endif ()

# TODO: convert to generator...
# disable ptxas warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcuda-ptxas '-w' -g")

# theoretical this should reduce compile times, but it doesn't
# set(CUDA_EXTRA_DEBUG_ARGS
#   "-O0 -Xcuda-ptxas '--allow-expensive-optimizations=false'")

# set(CUDA_EXTRA_DEBUG_ARGS
#   "${CUDA_EXTRA_DEBUG_ARGS} -fsanitize=address,undefined")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} ${CUDA_EXTRA_DEBUG_ARGS}")

# set(CMAKE_CXX_FLAGS_DEBUG
#   "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_QT_STANDARD 20)
set(CMAKE_QT_STANDARD_REQUIRED ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Qt5 COMPONENTS Widgets Xml REQUIRED)
find_package(OpenMP REQUIRED)
# find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/docopt.cpp")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/yaml-cpp")

set(BUILD_SHARED_LIBS OFF)

option(SYNTAX_ONLY "Only check syntax" OFF)
option(CPU_ONLY "don't build for the gpu" OFF)

if(CPU_ONLY)
  add_definitions(-DCPU_ONLY)
endif()

# qt5_wrap_cpp(UI_MAINWINDOW "${PROJECT_SOURCE_DIR}/include/ui/mainwindow.h")
# qt5_wrap_cpp(UI_VIEW "${PROJECT_SOURCE_DIR}/include/ui/canvas.h")

add_definitions(-DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int
  -DEIGEN_MAX_ALIGN_BYTES=0)

macro(target_add_standard_include TARGET_NAME)
  target_include_directories(${TARGET_NAME} PUBLIC
    "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(${TARGET_NAME} SYSTEM PUBLIC
    "${PROJECT_SOURCE_DIR}/system/include/"
    "${PROJECT_SOURCE_DIR}/extern/eigen/"
    "${PROJECT_SOURCE_DIR}/extern/include/"
    "/opt/cuda/include/"
    "${QT5_INCLUDE_DIR}")
endmacro()

add_library(
  intersect
  STATIC
  "${PROJECT_SOURCE_DIR}/src/intersect/accel/dir_tree/dir_tree.cu"
  "${PROJECT_SOURCE_DIR}/src/intersect/accel/kdtree/kdtree.cu"
  "${PROJECT_SOURCE_DIR}/src/intersect/accel/sphere_partition.cpp"

  "${PROJECT_SOURCE_DIR}/src/intersect/accel/kdtree/generator.cu"
  )
target_add_standard_include(intersect)
target_link_libraries(intersect PUBLIC yaml-cpp PRIVATE OpenMP::OpenMP_CXX)

add_library(
  renderer
  STATIC

  "${PROJECT_SOURCE_DIR}/src/render/renderer.cu"

  "${PROJECT_SOURCE_DIR}/src/integrate/light_sampler/random_triangle/random_triangle.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/tone_map.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/integrate_image.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/integrate_image.cpp"
  "${PROJECT_SOURCE_DIR}/src/render/detail/renderer_impl.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/divide_work.cpp"
  )
target_add_standard_include(renderer)
target_link_libraries(renderer PUBLIC yaml-cpp PRIVATE OpenMP::OpenMP_CXX
  Qt5::Widgets -lcurand intersect)

add_library(
  scenefile_compat
  STATIC

  "${PROJECT_SOURCE_DIR}/src/scene/camera.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/scenefile_compat/CS123XmlSceneParser.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/scenefile_compat/scenefile_loader.cpp"
  )
target_add_standard_include(scenefile_compat)
target_link_libraries(scenefile_compat PRIVATE Qt5::Xml yaml-cpp)

if(BUILD_TESTING OR CMAKE_EXPORT_COMPILE_COMMANDS)
  enable_testing()

  include(GoogleTest)

  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest")

  macro(package_add_test TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME} PRIVATE gtest gmock gtest_main)
    gtest_discover_tests(${TEST_NAME}
      WORKING_DIRECTORY ${PROJECT_DIR}
      PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
      )
    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    target_add_standard_include(${TEST_NAME})
    list(APPEND TEST_LIST ${TEST_NAME})
  endmacro()

  package_add_test(bitset "${PROJECT_SOURCE_DIR}/tests/src/bitset.cu")
  package_add_test(reduce "${PROJECT_SOURCE_DIR}/tests/src/reduce.cu")
  package_add_test(group "${PROJECT_SOURCE_DIR}/tests/src/group.cpp")
  package_add_test(device_vector
    "${PROJECT_SOURCE_DIR}/tests/src/device_vector.cu")
  package_add_test(lambda_check
    "${PROJECT_SOURCE_DIR}/tests/src/lambda_check.cu")
  # package_add_test(sphere_partition
  #   "${PROJECT_SOURCE_DIR}/tests/src/sphere_partition.cpp")
  # target_link_libraries(sphere_partition PRIVATE renderer)
  package_add_test(binary_search
    "${PROJECT_SOURCE_DIR}/tests/src/binary_search.cpp")
  package_add_test(projection "${PROJECT_SOURCE_DIR}/tests/src/projection.cpp")
  package_add_test(intersection
    "${PROJECT_SOURCE_DIR}/tests/src/intersection.cu")
  target_link_libraries(intersection PRIVATE intersect)

  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
    DEPENDS "${TEST_LIST}")
endif()

option(BUILD_BENCHMARKS "build benchmarks" OFF)
if(BUILD_BENCHMARKS OR CMAKE_EXPORT_COMPILE_COMMANDS)
  set(BENCHMARK_ENABLE_TESTING OFF)
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/benchmark")

  macro(package_add_bench BENCH_NAME)
    add_executable(${BENCH_NAME} ${ARGN})
    target_link_libraries(${BENCH_NAME} PRIVATE benchmark::benchmark
      benchmark::benchmark_main)
    set_target_properties(${BENCH_NAME} PROPERTIES FOLDER benchmarks)
    target_add_standard_include(${BENCH_NAME})
    list(APPEND BENCH_LIST ${BENCH_NAME})
  endmacro()

  package_add_bench(scan "${PROJECT_SOURCE_DIR}/benchmarks/src/scan.cu")
  # sometimes there are some issues with cub and warp intrinsics with sorting
  # (on clang)
  # package_add_bench(sort "${PROJECT_SOURCE_DIR}/benchmarks/src/sort.cu")

  add_custom_target(bench COMMAND ${CMAKE_CTEST_COMMAND}
    DEPENDS "${BENCH_LIST}")
endif()

add_executable(
  path
  "${PROJECT_SOURCE_DIR}/src/main.cpp"
  )
target_link_libraries(path PRIVATE renderer docopt_s scenefile_compat
  Qt5::Widgets)
target_add_standard_include(path)
