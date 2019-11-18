SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

SET(CMAKE_CXX_COMPILER clang++-9)
SET(CMAKE_QT_COMPILER clang++-9)
SET(CMAKE_CUDA_COMPILER clang++-9)

PROJECT(final LANGUAGES CUDA CXX)

CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

SET(CMAKE_AUTOMOC ON)
SET(CMAKE_AUTORCC ON)
SET(CMAKE_AUTOUIC ON)

FIND_PACKAGE(Qt5 COMPONENTS Widgets REQUIRED)
# FIND_PACKAGE(Eigen3 REQUIRED NO_MODULE)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Wconversion")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -Wall -Wextra -Wconversion")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

QT5_WRAP_CPP(UI_MAINWINDOW "${PROJECT_SOURCE_DIR}/include/ui/mainwindow.h")
QT5_WRAP_CPP(UI_VIEW "${PROJECT_SOURCE_DIR}/include/ui/canvas.h")

ADD_EXECUTABLE(
  final
  "${UI_MAINWINDOW}"
  "${UI_VIEW}"
  "${PROJECT_SOURCE_DIR}/src/ui/mainwindow.ui"
  "${PROJECT_SOURCE_DIR}/src/main.cpp"
  "${PROJECT_SOURCE_DIR}/src/ui/mainwindow.cpp"
  "${PROJECT_SOURCE_DIR}/src/ui/canvas.cpp"
  "${PROJECT_SOURCE_DIR}/src/ray/render.cu"
  "${PROJECT_SOURCE_DIR}/src/scene/pool_scene.cu"
  "${PROJECT_SOURCE_DIR}/src/scene/scene.cpp"
  )

ADD_EXECUTABLE(
  image_test
  "${PROJECT_SOURCE_DIR}/src/bin/image_test.cpp"
  "${PROJECT_SOURCE_DIR}/src/ray/render.cu"
  "${PROJECT_SOURCE_DIR}/src/scene/pool_scene.cu"
  "${PROJECT_SOURCE_DIR}/src/scene/scene.cpp"
  )

QT5_USE_MODULES(final Widgets)

# ADD_DEFINITIONS(-DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int)

TARGET_LINK_LIBRARIES(final Qt5::Widgets)
TARGET_INCLUDE_DIRECTORIES(final PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/eigen/"
  "${QT5_INCLUDE_DIR}"
  )

TARGET_LINK_LIBRARIES(image_test Qt5::Widgets)
  
TARGET_INCLUDE_DIRECTORIES(image_test PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/eigen/"
  "${QT5_INCLUDE_DIR}"
  )
