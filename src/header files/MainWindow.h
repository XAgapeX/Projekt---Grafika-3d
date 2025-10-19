#pragma once
#include <QWidget>

class QPushButton;
class VideoWindow;

class MainWindow : public QWidget {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void openVideoWindow();

private:
    QPushButton *openButton;
    VideoWindow *videoWindow;
};
