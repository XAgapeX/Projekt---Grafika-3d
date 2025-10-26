#include <QApplication>
#include "header files/MainWindow.h"
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);

    std::cout << "OpenCV version: " << CV_VERSION << std::endl;


    MainWindow mainWindow;
    mainWindow.show();

    return a.exec();
}
