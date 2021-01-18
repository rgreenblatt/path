cmake_policy(SET CMP0048 NEW)
project(
  path
  VERSION 0.1
  LANGUAGES CUDA CXX)

if(NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID
                                                     STREQUAL "GNU")))
  message(
    FATAL_ERROR
      "only clang and gcc are supported for c++ - using ${CMAKE_CXX_COMPILER_ID}"
  )
endif()

if(NOT (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang"))
  message(
    FATAL_ERROR
      "only clang is supported for cuda - using ${CMAKE_CUDA_COMPILER_ID}")
endif()

# required for clang cuda support
cmake_minimum_required(VERSION 3.18)

add_compile_options(-Wall -Wextra)

# * use libc++ for this project and submodules
# * won't work until libc++ has concepts

# add_compile_options(-stdlib=libc++)

# * unfortunately required with c++20 when using clang 11 and libstdc++
# * not required (right now) on trunk

# include_directories( SYSTEM
# $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_SOURCE_DIR}/system/include/>)

# I think -ffast-math is enabled by -Ofast, but the documentation isn't
# perfectly clear (certainly most options of -ffast-math are enabled by -Ofast,
# but I'm not sure about the cuda specific ones)
set(RELEASE_ARGS -Ofast;-ffast-math;-march=native)

# must be quoted for some reason I don't understand?
add_compile_options("$<$<CONFIG:RELEASE>:${RELEASE_ARGS}>")
add_compile_options("$<$<CONFIG:RELWITHDEBINFO>:${RELEASE_ARGS};-UNDEBUG>")
add_compile_options($<$<CONFIG:DEBUG>:-ftrapv>)

option(FORCE_COLORED_OUTPUT
       "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if(${FORCE_COLORED_OUTPUT})
  # TODO gnu/clang only
  add_compile_options(-fdiagnostics-color=always)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(
  Qt5
  COMPONENTS Widgets Xml
  REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(BUILD_SHARED_LIBS OFF)

# * remove _CMAKE_COMPILE_AS_CUDA_FLAG to support ccache with clang
# * this is clang specific
# * won't be required on latest release...
# * see https://github.com/ccache/ccache/issues/772
set(CMAKE_CUDA_COMPILE_WHOLE_COMPILATION
    "<CMAKE_CUDA_COMPILER> ${_CMAKE_CUDA_EXTRA_FLAGS} <DEFINES> <INCLUDES> <FLAGS> -c <SOURCE> -o <OBJECT>"
)

option(SYNTAX_ONLY "only check syntax (-fsyntax-only)" OFF)
# * very hacky, but saves a decent amount of compile time
# * not much difference with SYNTAX_ONLY
option(CPU_ONLY "don't build for the gpu (very hacky...)" OFF)

if(SYNTAX_ONLY)
  add_compile_options(-fsyntax-only)

  # gross.... (https://gitlab.kitware.com/cmake/cmake/-/issues/21717)
  set(CMAKE_CXX_COMPILE_OBJECT "${CMAKE_CXX_COMPILE_OBJECT} && touch <OBJECT>")
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY "touch <TARGET>")
  set(CMAKE_CXX_CREATE_SHARED_MODULE "touch <TARGET>")
  set(CMAKE_CXX_LINK_EXECUTABLE "touch <TARGET>")
  set(CMAKE_CXX_ARCHIVE_CREATE "touch <TARGET>")
  set(CMAKE_CXX_ARCHIVE_APPEND "touch <TARGET>")
  set(CMAKE_CXX_ARCHIVE_FINISH "touch <TARGET>")
  set(CMAKE_CXX_COMPILER_LAUNCHER "")

  # building only host should yield almost all diagnostics while also being much
  # faster - cuda clang specific
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--cuda-host-only>)

  # first one isn't actually used as far as I know...
  set(CMAKE_CUDA_COMPILE_OBJECT
      "${CMAKE_CUDA_COMPILE_OBJECT} && touch <OBJECT>")
  set(CMAKE_CUDA_COMPILE_PTX_COMPILATION
      "${CMAKE_CUDA_COMPILE_PTX_COMPILATION} && touch <OBJECT>")
  set(CMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION
      "${CMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION} && touch <OBJECT>")
  set(CMAKE_CUDA_COMPILE_WHOLE_COMPILATION
      "${CMAKE_CUDA_COMPILE_WHOLE_COMPILATION} && touch <OBJECT>")
  set(CMAKE_CUDA_CREATE_SHARED_LIBRARY "touch <TARGET>")
  set(CMAKE_CUDA_CREATE_SHARED_MODULE "touch <TARGET>")
  set(CMAKE_CUDA_LINK_EXECUTABLE "touch <TARGET>")
  set(CMAKE_CUDA_ARCHIVE_CREATE "touch <TARGET>")
  set(CMAKE_CUDA_ARCHIVE_APPEND "touch <TARGET>")
  set(CMAKE_CUDA_ARCHIVE_FINISH "touch <TARGET>")
  set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE "touch <TARGET>")
  set(CMAKE_CUDA_DEVICE_LINK_LIBRARY "touch <TARGET>")
  set(CMAKE_CUDA_DEVICE_LINK_COMPILE "touch <TARGET>")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "")
endif()

add_subdirectory("${PROJECT_SOURCE_DIR}/extern/docopt.cpp")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/yaml-cpp")

if(CPU_ONLY)
  set(CMAKE_CUDA_COMPILE_WHOLE_COMPILATION "${CMAKE_CXX_COMPILE_OBJECT}")

  if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    message(
      FATAL_ERROR
        "only clang works with CPU_ONLY? (TODO). Using ${CMAKE_CXX_COMPILER_ID}"
    )
  endif()

  add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-xc++;-Wno-unused-command-line-argument>)
  include_directories(
    SYSTEM
    $<$<COMPILE_LANGUAGE:CUDA>:"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

macro(target_add_standard TARGET_NAME)
  target_include_directories(${TARGET_NAME}
                             PUBLIC "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(
    ${TARGET_NAME} SYSTEM
    PUBLIC "${PROJECT_SOURCE_DIR}/extern/eigen/"
           "${PROJECT_SOURCE_DIR}/extern/include/"
           "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
  # required for Eigen: https://gitlab.com/libeigen/eigen/-/issues/1826
  target_compile_options(
    ${TARGET_NAME} PUBLIC -Wno-deprecated-copy
                          -Wno-deprecated-anon-enum-enum-conversion)
  # avoid any potential issues due to using cuda - maybe also slightly faster?
  target_compile_definitions(
    ${TARGET_NAME} PUBLIC EIGEN_DEFAULT_DENSE_INDEX_TYPE=int
                          EIGEN_MAX_ALIGN_BYTES=0)

  # BIG compile time savings...
  target_compile_definitions(${TARGET_NAME} PRIVATE MAGIC_ENUM_RANGE_MIN=0
                                                    MAGIC_ENUM_RANGE_MAX=8)

  # required for some thrust data structures (not currently used)
  target_compile_options(${TARGET_NAME} PRIVATE -fsized-deallocation)

  if(CPU_ONLY)
    target_compile_definitions(${TARGET_NAME} PRIVATE CPU_ONLY)
  endif()

  if(SYNTAX_ONLY)
    # builds are fast enough to check all combinations with syntax only
    target_compile_definitions(${TARGET_NAME} PRIVATE BUILD_ALL)
  endif()
endmacro()

add_library(
  intersect STATIC
  "${PROJECT_SOURCE_DIR}/src/intersect/accel/kdtree/kdtree.cu"
  "${PROJECT_SOURCE_DIR}/src/intersect/accel/sphere_partition.cpp"
  "${PROJECT_SOURCE_DIR}/src/intersect/accel/kdtree/detail/generator.cu")
target_add_standard(intersect)
target_link_libraries(intersect PRIVATE OpenMP::OpenMP_CXX)

add_library(kernel STATIC "${PROJECT_SOURCE_DIR}/src/kernel/work_division.cpp")
target_add_standard(kernel)

add_library(
  renderer STATIC
  "${PROJECT_SOURCE_DIR}/src/render/detail/renderer_impl_general_render_individual_gpu.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/renderer_impl_general_render_individual_cpu.cpp"
  "${PROJECT_SOURCE_DIR}/src/render/detail/renderer_impl_general_render_bulk_gpu.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/renderer_impl_general_render_bulk_cpu.cpp"
  "${PROJECT_SOURCE_DIR}/src/render/renderer.cu"
  "${PROJECT_SOURCE_DIR}/src/integrate/light_sampler/random_triangle/random_triangle.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/renderer_impl.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/reduce_float_rgb_gpu.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/reduce_float_rgb_cpu.cpp"
  "${PROJECT_SOURCE_DIR}/src/rng/sobel/sobel.cu")
target_add_standard(renderer)
target_link_libraries(renderer PRIVATE OpenMP::OpenMP_CXX CUDA::curand
                                       intersect kernel)

add_library(
  scenefile_compat STATIC
  "${PROJECT_SOURCE_DIR}/src/scene/camera.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/scenefile_compat/CS123XmlSceneParser.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/scenefile_compat/scenefile_loader.cpp")
target_add_standard(scenefile_compat)
target_link_libraries(scenefile_compat PRIVATE Qt5::Xml)

add_library(renderer_from_files STATIC
            "${PROJECT_SOURCE_DIR}/src/render/renderer_from_files.cpp")
target_add_standard(renderer_from_files)
target_link_libraries(renderer_from_files PRIVATE yaml-cpp renderer
                                                  scenefile_compat)

option(BUILD_TESTING "" OFF)

if(BUILD_TESTING OR CMAKE_EXPORT_COMPILE_COMMANDS)
  enable_testing()

  include(GoogleTest)

  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest")

  macro(package_add_test TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME} PRIVATE gtest gmock gtest_main)
    if(SYNTAX_ONLY)

    else()
      gtest_discover_tests(
        ${TEST_NAME}
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}")
    endif()

    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    target_add_standard(${TEST_NAME})
    list(APPEND TEST_LIST ${TEST_NAME})
  endmacro()

  package_add_test(bitset_test "${PROJECT_SOURCE_DIR}/tests/src/bitset.cpp")
  package_add_test(reduce_test "${PROJECT_SOURCE_DIR}/tests/src/reduce.cu")
  target_link_libraries(reduce_test PRIVATE kernel)

  package_add_test(edges_test "${PROJECT_SOURCE_DIR}/tests/src/edges.cpp")
  package_add_test(device_vector_test
                   "${PROJECT_SOURCE_DIR}/tests/src/device_vector.cu")
  package_add_test(lambda_test "${PROJECT_SOURCE_DIR}/tests/src/lambda.cu")
  # package_add_test(sphere_partition
  # "${PROJECT_SOURCE_DIR}/tests/src/sphere_partition.cpp")
  # target_link_libraries(sphere_partition PRIVATE renderer)
  package_add_test(binary_search_test
                   "${PROJECT_SOURCE_DIR}/tests/src/binary_search.cpp")
  package_add_test(projection "${PROJECT_SOURCE_DIR}/tests/src/projection.cpp")
  package_add_test(intersection_test
                   "${PROJECT_SOURCE_DIR}/tests/src/intersection.cu")
  target_link_libraries(intersection_test PRIVATE intersect)
  package_add_test(optional_test "${PROJECT_SOURCE_DIR}/tests/src/optional.cpp")
  package_add_test(work_division_test
                   "${PROJECT_SOURCE_DIR}/tests/src/work_division.cpp")
  target_link_libraries(work_division_test PRIVATE renderer)
  package_add_test(span_test "${PROJECT_SOURCE_DIR}/tests/src/span.cpp")
  package_add_test(tag_test "${PROJECT_SOURCE_DIR}/tests/src/tag.cpp")
  package_add_test(bit_utils_test
                   "${PROJECT_SOURCE_DIR}/tests/src/bit_utils.cpp")
  package_add_test(bit_utils_gpu_test
                   "${PROJECT_SOURCE_DIR}/tests/src/bit_utils_gpu.cu")
  package_add_test(all_values_test
                   "${PROJECT_SOURCE_DIR}/tests/src/all_values.cpp")
  package_add_test(macro_map_test
                   "${PROJECT_SOURCE_DIR}/tests/src/macro_map.cpp")

  add_custom_target(
    run_build_tests
    COMMAND ${CMAKE_CTEST_COMMAND}
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
    target_add_standard(${BENCH_NAME})
    list(APPEND BENCH_LIST ${BENCH_NAME})
  endmacro()

  package_add_bench(scan_bench "${PROJECT_SOURCE_DIR}/benchmarks/src/scan.cu")
  # sometimes there are some issues with cub and warp intrinsics with sorting
  # not sure if this has been fixed - some inconsistancy in the past? (on clang)
  package_add_bench(sort_bench "${PROJECT_SOURCE_DIR}/benchmarks/src/sort.cu")
  package_add_bench(render_bench
                    "${PROJECT_SOURCE_DIR}/benchmarks/src/render.cpp")
  package_add_bench(reduce_bench
                    "${PROJECT_SOURCE_DIR}/benchmarks/src/reduce.cu")
  target_link_libraries(render_bench PRIVATE renderer_from_files)
  target_compile_options(reduce_bench PRIVATE -fconstexpr-steps=10000000)

  # doesn't actually run...
  add_custom_target(
    bench
    COMMAND
    DEPENDS "${BENCH_LIST}")
endif()

add_executable(path "${PROJECT_SOURCE_DIR}/src/main.cpp")
target_link_libraries(path PRIVATE renderer_from_files docopt_s Qt5::Widgets)
target_add_standard(path)

add_executable(check_downsample
               "${PROJECT_SOURCE_DIR}/src/check_downsample.cpp")
target_link_libraries(check_downsample PRIVATE renderer_from_files Qt5::Widgets)
target_add_standard(check_downsample)

add_executable(loaded_float_rgb
               "${PROJECT_SOURCE_DIR}/src/loaded_float_rgb.cpp")
target_link_libraries(loaded_float_rgb PRIVATE Qt5::Widgets)
target_add_standard(loaded_float_rgb)
