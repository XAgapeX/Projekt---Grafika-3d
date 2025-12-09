#pragma once
#include <QWidget>
#include <QStackedWidget>
class QPushButton;
class VideoWindow;

/**
 * @class MainWindow
 * @brief Główne okno aplikacji odtwarzacza wideo.
 *
 * Odpowiada za wyświetlanie ekranu startowego
 * oraz za przełączanie do okna odtwarzania wideo (VideoWindow)
 * za pomocą QStackedWidget.
 */

class MainWindow : public QWidget {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private:
    QPushButton *openButton;
    VideoWindow *videoWindow;
    QStackedWidget *stackedWidget;
};
