#pragma once
#include <QWidget>

class QVideoWidget;
class QMediaPlayer;
class QPushButton;

class VideoWindow : public QWidget {
    Q_OBJECT

public:
    explicit VideoWindow(QWidget *parent = nullptr);

private slots:
    void loadVideo();

private:
    QMediaPlayer *player;
    QVideoWidget *videoWidget;
    QPushButton *loadButton;
};
