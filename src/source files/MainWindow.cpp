#include "../header files/MainWindow.h"
#include "../header files/VideoWindow.h"
#include <QPushButton>
#include <QLabel>
#include <QHBoxLayout>

MainWindow::MainWindow(QWidget *parent)
    : QWidget(parent), videoWindow(nullptr) {
    setWindowTitle("Video Player");
    setWindowIcon(QIcon(":/resources/icons/play.png"));
    resize(1200, 800);
    setStyleSheet(R"(
        QWidget {
            background: "#fff";
        }
    )");

    QLabel *iconLabel = new QLabel(this);
    QPixmap pixmap(":/resources/icons/play.png");
    iconLabel->setPixmap(pixmap.scaled(64, 64, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    iconLabel->setAlignment(Qt::AlignCenter);

    QLabel *label = new QLabel("Video Player", this);
    label->setAlignment(Qt::AlignCenter);
    label->setStyleSheet(R"(
        QLabel {
            font-size: 24px;
            font-weight: bold;
            color: #21618c;
            background-color: transparent;
        }
    )");

    openButton = new QPushButton("Open", this);
    openButton->setFixedSize(120, 40);
    openButton->setStyleSheet(R"(
        QPushButton {
            background: qlineargradient(
                x1: 0, y1: 0,
                x2: 1, y2: 1,
                stop: 0 #5dade2,
                stop: 1 #21618c
            );
            color: white;
            border: none;
            border-radius: 5px;
            padding: 5px 10px;
            font-size: 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: qlineargradient(
                x1: 0, y1: 0,
                x2: 1, y2: 1,
                stop: 0 #85c1e9,
                stop: 1 #2874a6
            );
        }
    )");


    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(25);
    layout->addWidget(iconLabel);
    layout->addWidget(label);

    QWidget *buttonWrapper = new QWidget(this);
    QHBoxLayout *buttonLayout = new QHBoxLayout(buttonWrapper);
    buttonLayout->setContentsMargins(0, 0, 0, 0);
    buttonLayout->addWidget(openButton, 0, Qt::AlignCenter);
    layout->addWidget(buttonWrapper);

    layout->setAlignment(Qt::AlignCenter);

    connect(openButton, &QPushButton::clicked, this, &MainWindow::openVideoWindow);
}

void MainWindow::openVideoWindow() {
    if (!videoWindow) {
        videoWindow = new VideoWindow();
    }
    videoWindow->show();
    videoWindow->raise();
    videoWindow->activateWindow();
}
