cmake_minimum_required(VERSION 3.18)
project(PROJEKT2_0 LANGUAGES CXX CUDA)
enable_language(CUDA)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if (WIN32)
    set(CMAKE_PREFIX_PATH "C:/Qt/6.9.1/msvc2022_64/lib/cmake")
elseif (UNIX AND NOT APPLE)
    set(CMAKE_PREFIX_PATH "/home/damiand/Qt/6.9.1/gcc_64/lib/cmake")
endif()

find_package(Qt6 COMPONENTS
        Core
        Gui
        Widgets
        Multimedia
        MultimediaWidgets
        REQUIRED
)


if (WIN32)
    set(OpenCV_DIR "C:/Users/dell/Documents/opencv/build/x64/vc15/lib")
endif()
find_package(OpenCV REQUIRED)


find_package(CUDAToolkit REQUIRED)
if (WIN32)
    set(CMAKE_CUDA_ARCHITECTURES 75)
else()
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()


add_executable(PROJEKT2_0
        src/main.cpp
        resources.qrc

        "src/source files/MainWindow.cpp"
        "src/header files/MainWindow.h"

        "src/source files/VideoWindow.cpp"
        "src/header files/VideoWindow.h"

        "src/source files/ToolBar.cpp"
        "src/header files/ToolBar.h"

        "src/source files/ControlBar.cpp"
        "src/header files/ControlBar.h"

        "src/opencv/source files/SenderOpenCV.cpp"
        "src/opencv/header files/SenderOpenCV.h"

        "src/cuda/greyscale.cu"
        "src/cuda/gaussianblur.cu"
        "src/cuda/sepia.cu"
        "src/cuda/negative.cu"
        "src/cuda/sobel.cu"
)

target_include_directories(PROJEKT2_0 PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(PROJEKT2_0 PRIVATE
        Qt::Core
        Qt::Gui
        Qt::Widgets
        Qt::Multimedia
        Qt::MultimediaWidgets
        ${OpenCV_LIBS}
        CUDA::cudart
)

if (WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(DEBUG_SUFFIX)
    if (CMAKE_BUILD_TYPE MATCHES "Debug")
        set(DEBUG_SUFFIX "d")
    endif()


    set(QT_BASE_PATH "C:/Qt/6.9.1/msvc2022_64")


    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/platforms/"
    )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${QT_BASE_PATH}/plugins/platforms/qwindows${DEBUG_SUFFIX}.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/platforms/"
    )


    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/multimedia/"
    )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${QT_BASE_PATH}/plugins/multimedia/ffmpegmediaplugin${DEBUG_SUFFIX}.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/multimedia/"
    )


    foreach(QT_LIB Core Gui Widgets Multimedia MultimediaWidgets Network)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${QT_BASE_PATH}/bin/Qt6${QT_LIB}${DEBUG_SUFFIX}.dll"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        )
    endforeach()


    set(CUDA_BIN_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin")
    foreach(CUDA_DLL cudart64_130.dll)
        if (EXISTS "${CUDA_BIN_DIR}/${CUDA_DLL}")
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CUDA_BIN_DIR}/${CUDA_DLL}"
                    "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
        endif()
    endforeach()


    set_target_properties(${PROJECT_NAME} PROPERTIES
            VS_DEBUGGER_ENVIRONMENT
            "PATH=C:/Qt/6.9.1/msvc2022_64/bin;%PATH%;
        QT_PLUGIN_PATH=$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins;
        QT_QPA_PLATFORM_PLUGIN_PATH=$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/platforms;
        QT_DEBUG_PLUGINS=1"
    )
endif()