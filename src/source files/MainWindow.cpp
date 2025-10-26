#include "../header files/MainWindow.h"
#include "../header files/VideoWindow.h"
#include <QPushButton>
#include <QLabel>
#include <QIcon>
#include <QVBoxLayout>
#include <QStackedWidget>
#include <QPixmap>

MainWindow::MainWindow(QWidget *parent)
    : QWidget(parent) {
    setWindowTitle("Video Player");
    setWindowIcon(QIcon(":/resources/icons/play.png"));
    resize(1200, 800);
    setStyleSheet("background-color: #f5f5f5;");

    QLabel *iconLabel = new QLabel(this);
    QPixmap pixmap(":/resources/icons/play.png");
    iconLabel->setPixmap(pixmap.scaled(64, 64, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    iconLabel->setAlignment(Qt::AlignCenter);

    QLabel *label = new QLabel("Video Player", this);
    label->setAlignment(Qt::AlignCenter);
    label->setStyleSheet("font-size: 24px; font-weight: bold; color: #21618c;");

    openButton = new QPushButton("Open Video", this);
    openButton->setFixedSize(120, 40);
    openButton->setStyleSheet(
        "background-color: #21618c; color: white; border-radius: 5px; font-size: 16px; font-weight: bold;");

    QVBoxLayout *headerLayout = new QVBoxLayout();
    headerLayout->addWidget(iconLabel);
    headerLayout->addWidget(label);
    headerLayout->addWidget(openButton, 0, Qt::AlignCenter);

    QWidget *headerWidget = new QWidget(this);
    headerWidget->setLayout(headerLayout);

    stackedWidget = new QStackedWidget(this);

    QWidget *homeWidget = new QWidget(this);
    QVBoxLayout *homeLayout = new QVBoxLayout(homeWidget);
    homeLayout->addWidget(headerWidget, 0, Qt::AlignCenter);
    homeWidget->setLayout(homeLayout);

    videoWindow = new VideoWindow(this);

    stackedWidget->addWidget(homeWidget);
    stackedWidget->addWidget(videoWindow);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(stackedWidget);
    setLayout(mainLayout);

    connect(openButton, &QPushButton::clicked, this, [this]() {
        stackedWidget->setCurrentWidget(videoWindow);
        videoWindow->loadVideo();
    });
}
