set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_COMPILER clang++-9)
set(CMAKE_QT_COMPILER clang++-9)
set(CMAKE_CUDA_COMPILER clang++-9)

cmake_policy(SET CMP0048 NEW)
project(final VERSION 0.1 LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# set(CMAKE_VERBOSE_MAKEFILE TRUE)

find_package(Qt5 COMPONENTS Widgets Xml REQUIRED)
find_package(OpenMP REQUIRED)
# find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(GENERAL_FLAGS "-std=c++17 -Wall -Wextra -fsized-deallocation")
set(CMAKE_QT_FLAGS "${CMAKE_CXX_FLAGS} ${GENERAL_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GENERAL_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${GENERAL_FLAGS}")
set(RELEASE_FlAGS "-Ofast -march=native -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FlAGS}")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} ${RELEASE_FlAGS}")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")

qt5_wrap_cpp(UI_MAINWINDOW "${PROJECT_SOURCE_DIR}/include/ui/mainwindow.h")
qt5_wrap_cpp(UI_VIEW "${PROJECT_SOURCE_DIR}/include/ui/canvas.h")

add_definitions(-DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int
  -DEIGEN_MAX_ALIGN_BYTES=0)

set(STANDARD_INCLUDE 
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/extern/eigen/"
  "/usr/local/cuda/include/"
  "${QT5_INCLUDE_DIR}")

add_library(
  renderer

  "${PROJECT_SOURCE_DIR}/src/ray/render.cu"

  "${PROJECT_SOURCE_DIR}/src/ray/detail/render_impl.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/impl/fill.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/impl/float_to_bgra.cpp"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/impl/float_to_bgra.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/impl/raytrace/all.cpp"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/impl/raytrace/kdtree_false.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/impl/raytrace/kdtree_true.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/impl/raytrace/loop_all_false.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/impl/raytrace/loop_all_true.cu"

  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/kdtree/kdtree.cpp"

  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/compute_aabbs.cpp"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/compute_aabbs.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/construct.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/copy_to_sortable.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/dir_tree_generator.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/dir_tree_generator.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/fill_indexes.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/permute.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/setup.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/sort.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/sphere_partition.cpp"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/fill_keys.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/detail/accel/dir_tree/fill_keys.cpp"

  "${PROJECT_SOURCE_DIR}/src/scene/camera.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/pool_scene.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/reflec_ball.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/scene.cpp"
  )
target_include_directories(renderer PUBLIC ${STANDARD_INCLUDE})
target_link_libraries(renderer OpenMP::OpenMP_CXX Qt5::Widgets)

if(BUILD_TESTING OR CMAKE_EXPORT_COMPILE_COMMANDS)
  enable_testing()

  include(GoogleTest)

  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest")

  macro(package_add_test TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME} gtest gmock gtest_main)
    gtest_discover_tests(${TEST_NAME}
      WORKING_DIRECTORY ${PROJECT_DIR}
      PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
      )
    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    target_include_directories(${TEST_NAME} PUBLIC ${STANDARD_INCLUDE})
    list(APPEND TEST_LIST ${TEST_NAME})
  endmacro()

  package_add_test(bitset "${PROJECT_SOURCE_DIR}/tests/src/bitset.cu")

  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
    DEPENDS "${TEST_LIST}")
endif()

option(BUILD_BENCHMARKS "build benchmarks" OFF)
if(BUILD_BENCHMARKS OR CMAKE_EXPORT_COMPILE_COMMANDS)
  set(BENCHMARK_ENABLE_TESTING OFF)
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/benchmark")

  macro(package_add_bench BENCH_NAME)
    add_executable(${BENCH_NAME} ${ARGN})
    target_link_libraries(${BENCH_NAME} benchmark::benchmark
      benchmark::benchmark_main)
    set_target_properties(${BENCH_NAME} PROPERTIES FOLDER benchmarks)
    target_include_directories(${BENCH_NAME} PUBLIC ${STANDARD_INCLUDE})
    list(APPEND BENCH_LIST ${BENCH_NAME})
  endmacro()

  package_add_bench(scan "${PROJECT_SOURCE_DIR}/benchmarks/src/scan.cu")

  add_custom_target(bench COMMAND ${CMAKE_CTEST_COMMAND}
    DEPENDS "${BENCH_LIST}")
endif()

add_executable(
  final
  "${UI_MAINWINDOW}"
  "${UI_VIEW}"
  "${PROJECT_SOURCE_DIR}/src/ui/mainwindow.ui"
  "${PROJECT_SOURCE_DIR}/src/main.cpp"
  "${PROJECT_SOURCE_DIR}/src/ui/mainwindow.cpp"
  "${PROJECT_SOURCE_DIR}/src/ui/canvas.cpp"
  )
qt5_use_modules(final Widgets)
target_link_libraries(final Qt5::Widgets renderer)
target_include_directories(final PUBLIC ${STANDARD_INCLUDE})

add_executable(
  image_test
  "${PROJECT_SOURCE_DIR}/src/bin/image_test.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/cs123_scene.cpp"
  "${PROJECT_SOURCE_DIR}/src/cs123_compat/Scene.cpp"
  "${PROJECT_SOURCE_DIR}/src/cs123_compat/CS123XmlSceneParser.cpp"
  )
target_link_libraries(image_test Qt5::Widgets Qt5::Xml renderer)
target_include_directories(image_test PUBLIC ${STANDARD_INCLUDE})

# add_executable(
#   video_test
#   "${PROJECT_SOURCE_DIR}/src/bin/video_test.cpp"
#   )
# target_link_libraries(video_test Qt5::Widgets Qt5::Xml renderer ${OpenCV_LIBS})
# target_include_directories(video_test PUBLIC ${STANDARD_INCLUDE})

add_executable(
  sort_bench
  "${PROJECT_SOURCE_DIR}/src/bin/sort_bench.cu"
  "${PROJECT_SOURCE_DIR}/src/lib/parallel_for_loop.cpp"
  )
target_include_directories(sort_bench PUBLIC ${STANDARD_INCLUDE})
target_link_libraries(sort_bench Threads::Threads)

add_executable(
  scan_bench
  "${PROJECT_SOURCE_DIR}/src/bin/scan_bench.cu"
  )
target_include_directories(scan_bench PUBLIC ${STANDARD_INCLUDE})
target_link_libraries(scan_bench Threads::Threads)

