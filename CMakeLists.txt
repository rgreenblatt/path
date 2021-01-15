set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_QT_COMPILER clang++)
set(CMAKE_CUDA_COMPILER clang++)

cmake_policy(SET CMP0048 NEW)
project(path VERSION 0.1 LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.18)

add_compile_options(-Wall -Wextra -fsized-deallocation)
add_link_options(-fuse-ld=lld)

# set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)

# use libc++ for this project and submodules 
# won't work until libc++ has concepts
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -stdlib=libc++")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

# unfortunately required with c++20
add_compile_options(-isystem ${CMAKE_SOURCE_DIR}/system/include/)
add_compile_options(-DMAGIC_ENUM_RANGE_MIN=0 -DMAGIC_ENUM_RANGE_MAX=8)

# required for Eigen
add_compile_options(-Wno-deprecated-copy)
add_compile_options(-Wno-deprecated-anon-enum-enum-conversion)

add_compile_options(-fconstexpr-steps=80000000 -fbracket-depth=512)
# add_compile_options(-fmodules)
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
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcuda-ptxas '-w'")

# set(CMAKE_CUDA_FLAGS_DEBUG 
#   "${CMAKE_CUDA_FLAGS_DEBUG} -fsanitize=address,undefined")

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

set(BUILD_SHARED_LIBS OFF)

# remove _CMAKE_COMPILE_AS_CUDA_FLAG to support ccache
# set(CMAKE_CUDA_COMPILE_WHOLE_COMPILATION
#     "<CMAKE_CUDA_COMPILER> ${_CMAKE_CUDA_EXTRA_FLAGS} <DEFINES> <INCLUDES> <FLAGS> -c <SOURCE> -o <OBJECT>")

option(SYNTAX_ONLY "only check syntax (-fsyntax-only)" OFF)
option(CPU_ONLY "don't build for the gpu" OFF)

if(SYNTAX_ONLY)
  # builds are fast enough to check all combinations with syntax only
  add_definitions(-DBUILD_ALL)
  add_compile_options(-fsyntax-only)

  # building only host should yield almost all diagnostics while also
  # being much faster
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cuda-host-only")

  set(CMAKE_CXX_COMPILE_OBJECT "${CMAKE_CXX_COMPILE_OBJECT} && touch <OBJECT>")
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY "touch <TARGET>")
  set(CMAKE_CXX_CREATE_SHARED_MODULE "touch <TARGET>")
  set(CMAKE_CXX_LINK_EXECUTABLE "touch <TARGET>")
  set(CMAKE_CXX_ARCHIVE_CREATE "touch <TARGET>")
  set(CMAKE_CXX_ARCHIVE_APPEND "touch <TARGET>")
  set(CMAKE_CXX_ARCHIVE_FINISH "touch <TARGET>")
  set(CMAKE_CXX_COMPILER_LAUNCHER "")

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
  add_definitions(-DCPU_ONLY)
  # very hacky, but saves a decent amount of compile time...
  set(CMAKE_CUDA_COMPILE_WHOLE_COMPILATION "${CMAKE_CXX_COMPILE_OBJECT}")
  add_compile_options(-xc++ -isystem/opt/cuda/include/ 
    -Wno-unused-command-line-argument)
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
    "/opt/cuda/include/")
endmacro()

add_library(
  intersect
  STATIC
  "${PROJECT_SOURCE_DIR}/src/intersect/accel/kdtree/kdtree.cu"
  "${PROJECT_SOURCE_DIR}/src/intersect/accel/sphere_partition.cpp"

  "${PROJECT_SOURCE_DIR}/src/intersect/accel/kdtree/detail/generator.cu"
  )
target_add_standard_include(intersect)
target_link_libraries(intersect PRIVATE OpenMP::OpenMP_CXX)

add_library(
  kernel
  STATIC
  "${PROJECT_SOURCE_DIR}/src/kernel/work_division.cpp"
)
target_add_standard_include(kernel)

add_library(
  renderer
  STATIC
  "${PROJECT_SOURCE_DIR}/src/render/detail/renderer_impl_general_render_individual_gpu.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/renderer_impl_general_render_individual_cpu.cpp"
  "${PROJECT_SOURCE_DIR}/src/render/detail/renderer_impl_general_render_bulk_gpu.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/renderer_impl_general_render_bulk_cpu.cpp"

  "${PROJECT_SOURCE_DIR}/src/render/renderer.cu"

  "${PROJECT_SOURCE_DIR}/src/integrate/light_sampler/random_triangle/random_triangle.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/renderer_impl.cu"
  "${PROJECT_SOURCE_DIR}/src/render/detail/reduce_float_rgb_gpu.cu"
  "${PROJECT_SOURCE_DIR}/src/rng/sobel/sobel.cu"
  )
target_add_standard_include(renderer)
target_link_libraries(renderer PRIVATE OpenMP::OpenMP_CXX -lcurand intersect
  kernel)

add_library(
  scenefile_compat
  STATIC

  "${PROJECT_SOURCE_DIR}/src/scene/camera.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/scenefile_compat/CS123XmlSceneParser.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/scenefile_compat/scenefile_loader.cpp"
  )
target_add_standard_include(scenefile_compat)
target_link_libraries(scenefile_compat PRIVATE Qt5::Xml)

add_library(
  renderer_from_files
  STATIC
  "${PROJECT_SOURCE_DIR}/src/render/renderer_from_files.cpp"
)
target_add_standard_include(renderer_from_files)
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
    gtest_discover_tests(${TEST_NAME}
      WORKING_DIRECTORY ${PROJECT_DIR}
      PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
      )
    endif()

    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    target_add_standard_include(${TEST_NAME})
    list(APPEND TEST_LIST ${TEST_NAME})
  endmacro()

  package_add_test(bitset_test "${PROJECT_SOURCE_DIR}/tests/src/bitset.cpp")
  package_add_test(reduce_test "${PROJECT_SOURCE_DIR}/tests/src/reduce.cu")
  target_link_libraries(reduce_test PRIVATE kernel)

  package_add_test(edges_test "${PROJECT_SOURCE_DIR}/tests/src/edges.cpp")
  package_add_test(device_vector_test
    "${PROJECT_SOURCE_DIR}/tests/src/device_vector.cu")
  package_add_test(lambda_test
    "${PROJECT_SOURCE_DIR}/tests/src/lambda.cu")
  # package_add_test(sphere_partition
  #   "${PROJECT_SOURCE_DIR}/tests/src/sphere_partition.cpp")
  # target_link_libraries(sphere_partition PRIVATE renderer)
  package_add_test(binary_search_test
    "${PROJECT_SOURCE_DIR}/tests/src/binary_search.cpp")
  package_add_test(projection "${PROJECT_SOURCE_DIR}/tests/src/projection.cpp")
  package_add_test(intersection_test
    "${PROJECT_SOURCE_DIR}/tests/src/intersection.cu")
  target_link_libraries(intersection_test PRIVATE intersect)
  package_add_test(optional_test
    "${PROJECT_SOURCE_DIR}/tests/src/optional.cpp")
  package_add_test(work_division_test
    "${PROJECT_SOURCE_DIR}/tests/src/work_division.cpp")
  target_link_libraries(work_division_test PRIVATE renderer)
  package_add_test(span_test "${PROJECT_SOURCE_DIR}/tests/src/span.cpp")
  package_add_test(tag_test "${PROJECT_SOURCE_DIR}/tests/src/tag.cpp")

  add_custom_target(run_build_tests COMMAND ${CMAKE_CTEST_COMMAND}
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

  package_add_bench(scan_bench "${PROJECT_SOURCE_DIR}/benchmarks/src/scan.cu")
  # sometimes there are some issues with cub and warp intrinsics with sorting
  # (on clang)
  package_add_bench(sort_bench "${PROJECT_SOURCE_DIR}/benchmarks/src/sort.cu")
  package_add_bench(render_bench "${PROJECT_SOURCE_DIR}/benchmarks/src/render.cpp")
  target_link_libraries(render_bench PRIVATE renderer_from_files)

  add_custom_target(bench COMMAND ${CMAKE_CTEST_COMMAND}
    DEPENDS "${BENCH_LIST}")
endif()

add_executable(
  path
  "${PROJECT_SOURCE_DIR}/src/main.cpp"
  )
target_link_libraries(path PRIVATE renderer_from_files docopt_s Qt5::Widgets)
target_add_standard_include(path)

add_executable(
  check_downsample
  "${PROJECT_SOURCE_DIR}/src/check_downsample.cpp"
  )
target_link_libraries(check_downsample PRIVATE renderer_from_files Qt5::Widgets)
target_add_standard_include(check_downsample)

add_executable(
  loaded_float_rgb
  "${PROJECT_SOURCE_DIR}/src/loaded_float_rgb.cpp"
  )
target_link_libraries(loaded_float_rgb PRIVATE Qt5::Widgets)
target_add_standard_include(loaded_float_rgb)
