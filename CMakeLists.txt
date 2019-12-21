SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

SET(CMAKE_CXX_COMPILER clang++-9)
SET(CMAKE_QT_COMPILER clang++-9)
SET(CMAKE_CUDA_COMPILER clang++-9)

PROJECT(final LANGUAGES CUDA CXX)

CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

SET(CMAKE_AUTOMOC ON)
SET(CMAKE_AUTORCC ON)
SET(CMAKE_AUTOUIC ON)

# SET(CMAKE_VERBOSE_MAKEFILE TRUE)

FIND_PACKAGE(Qt5 COMPONENTS Widgets Xml REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

SET(GENERAL_FLAGS "-std=c++17 -Wall -Wextra -fsized-deallocation")
SET(CMAKE_QT_FLAGS "${CMAKE_CXX_FLAGS} ${GENERAL_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GENERAL_FLAGS}")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${GENERAL_FLAGS}")
SET(RELEASE_FlAGS "-Ofast -march=native -g -DNDEBUG")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FlAGS}")
SET(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} ${RELEASE_FlAGS}")
SET(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")

QT5_WRAP_CPP(UI_MAINWINDOW "${PROJECT_SOURCE_DIR}/include/ui/mainwindow.h")
QT5_WRAP_CPP(UI_VIEW "${PROJECT_SOURCE_DIR}/include/ui/canvas.h")

ADD_DEFINITIONS(-DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int)

ADD_LIBRARY(
  renderer
  "${PROJECT_SOURCE_DIR}/src/ray/render_impl.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/render_impl_traversal_grid.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/render_impl_fill.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/render_impl_raytrace.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/render.cu"
  "${PROJECT_SOURCE_DIR}/src/ray/kdtree.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/scene.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/pool_scene.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/reflec_ball.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/camera.cpp"
  "${PROJECT_SOURCE_DIR}/src/ray/traversal_grid.cpp"
  )

TARGET_INCLUDE_DIRECTORIES(renderer PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/eigen/"
  "/usr/local/cuda/include/"
  "${QT5_INCLUDE_DIR}"
  )
TARGET_LINK_LIBRARIES(renderer OpenMP::OpenMP_CXX Qt5::Widgets)

ADD_EXECUTABLE(
  final
  "${UI_MAINWINDOW}"
  "${UI_VIEW}"
  "${PROJECT_SOURCE_DIR}/src/ui/mainwindow.ui"
  "${PROJECT_SOURCE_DIR}/src/main.cpp"
  "${PROJECT_SOURCE_DIR}/src/ui/mainwindow.cpp"
  "${PROJECT_SOURCE_DIR}/src/ui/canvas.cpp"
  )

ADD_EXECUTABLE(
  image_test
  "${PROJECT_SOURCE_DIR}/src/bin/image_test.cpp"
  "${PROJECT_SOURCE_DIR}/src/scene/cs123_scene.cpp"
  "${PROJECT_SOURCE_DIR}/src/cs123_compat/Scene.cpp"
  "${PROJECT_SOURCE_DIR}/src/cs123_compat/CS123XmlSceneParser.cpp"
  )

ADD_EXECUTABLE(
  video_test
  "${PROJECT_SOURCE_DIR}/src/bin/video_test.cpp"
  )

QT5_USE_MODULES(final Widgets)
TARGET_LINK_LIBRARIES(final Qt5::Widgets renderer)
TARGET_INCLUDE_DIRECTORIES(final PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/eigen/"
  "/usr/local/cuda/include/"
  "${QT5_INCLUDE_DIR}"
  )

TARGET_LINK_LIBRARIES(image_test Qt5::Widgets Qt5::Xml renderer)
TARGET_INCLUDE_DIRECTORIES(image_test PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/eigen/"
  "/usr/local/cuda/include/"
  "${QT5_INCLUDE_DIR}"
  )

TARGET_LINK_LIBRARIES(video_test Qt5::Widgets Qt5::Xml renderer ${OpenCV_LIBS})
TARGET_INCLUDE_DIRECTORIES(video_test PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/eigen/"
  "/usr/local/cuda/include/"
  "${QT5_INCLUDE_DIR}"
  )
