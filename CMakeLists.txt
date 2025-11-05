cmake_minimum_required(VERSION 3.16)
project(PROJEKT2_0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if (WIN32)
    set(CMAKE_PREFIX_PATH "C:/Qt/6.9.1/mingw_64/lib/cmake")
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
    set(OpenCV_DIR "C:/Users/Katarzyna/Documents/OpenCV/opencv/build/x64/vc15/lib")
endif()

find_package(OpenCV REQUIRED)

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
)


target_link_libraries(PROJEKT2_0
        Qt::Core
        Qt::Gui
        Qt::Widgets
        Qt::Multimedia
        Qt::MultimediaWidgets
        ${OpenCV_LIBS}
)

target_include_directories(PROJEKT2_0 PRIVATE ${OpenCV_INCLUDE_DIRS})

if (WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(DEBUG_SUFFIX)
    if (MSVC AND CMAKE_BUILD_TYPE MATCHES "Debug")
        set(DEBUG_SUFFIX "d")
    endif ()

    set(QT_INSTALL_PATH "${CMAKE_PREFIX_PATH}")
    if (NOT EXISTS "${QT_INSTALL_PATH}/bin")
        set(QT_INSTALL_PATH "${QT_INSTALL_PATH}/..")
        if (NOT EXISTS "${QT_INSTALL_PATH}/bin")
            set(QT_INSTALL_PATH "${QT_INSTALL_PATH}/..")
        endif ()
    endif ()

    if (EXISTS "${QT_INSTALL_PATH}/plugins/platforms/qwindows${DEBUG_SUFFIX}.dll")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/platforms/"
        )
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                "${QT_INSTALL_PATH}/plugins/platforms/qwindows${DEBUG_SUFFIX}.dll"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/platforms/"
        )
    endif ()

    if (EXISTS "${QT_INSTALL_PATH}/plugins/multimedia/ffmpegmediaplugin${DEBUG_SUFFIX}.dll")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/multimedia/"
        )
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                "${QT_INSTALL_PATH}/plugins/multimedia/ffmpegmediaplugin${DEBUG_SUFFIX}.dll"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/multimedia/"
        )
    endif ()

    foreach (QT_LIB Core Gui Widgets Multimedia MultimediaWidgets Network)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                "${QT_INSTALL_PATH}/bin/Qt6${QT_LIB}${DEBUG_SUFFIX}.dll"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        )
    endforeach ()


    if (WIN32)
        set(MINGW_BIN_DIR "C:/Qt/6.9.1/mingw_64/bin")

        foreach(DLL_FILE
                libgcc_s_seh-1.dll
                libstdc++-6.dll
                libwinpthread-1.dll
        )
            if (EXISTS "${MINGW_BIN_DIR}/${DLL_FILE}")
                add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${MINGW_BIN_DIR}/${DLL_FILE}"
                        "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
            endif()
        endforeach()
    endif()


    if (EXISTS "${QT_INSTALL_PATH}/plugins/multimedia/ffmpegmediaplugin${DEBUG_SUFFIX}.dll")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/multimedia/"
        )
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                "${QT_INSTALL_PATH}/plugins/multimedia/ffmpegmediaplugin${DEBUG_SUFFIX}.dll"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/multimedia/"
        )
    endif ()

endif()
