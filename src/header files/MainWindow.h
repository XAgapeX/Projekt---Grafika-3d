#pragma once
#include <QWidget>
#include <QStackedWidget>
class QPushButton;
class VideoWindow;


class MainWindow : public QWidget {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private:
    QPushButton *openButton;
    VideoWindow *videoWindow;
    QStackedWidget *stackedWidget;
};
