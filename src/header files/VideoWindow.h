#pragma once
#include <../../../src/header files/ToolBar.h>
#include <../../../src/header files/ControlBar.h>

#include <QWidget>
#include <QMediaPlayer>
#include <QVideoWidget>
#include <QAudioOutput>

class VideoWindow : public QWidget {
    Q_OBJECT
public:
    explicit VideoWindow(QWidget *parent = nullptr);

    void loadVideo();

private:
    QMediaPlayer *player;
    QVideoWidget *videoWidget;
    QAudioOutput *audioOutput;
    ToolBar *toolBar;
    ControlBar *controlBar;
};
