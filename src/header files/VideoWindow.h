#pragma once
#include "../header files/ToolBar.h"
#include "../header files/ControlBar.h"
#include "../cuda/grayscale.cuh"

#include <QWidget>
#include <QMediaPlayer>
#include <QVideoSink>
#include <QAudioOutput>
#include <QLabel>
#include <QVideoFrame>
#include <QImage>


class VideoWindow : public QWidget {
    Q_OBJECT
public:
    explicit VideoWindow(QWidget *parent = nullptr);
    void loadVideo();

protected:
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void applyGrayscaleFilter();
    void applyGaussianFilter();
    void applySepiaFilter();
    void applyNegativeFilter();
    void applySobelFilter();
    void onFrameAvailable(const QVideoFrame &frame);

private:
    QMediaPlayer *player;
    QAudioOutput *audioOutput;
    QVideoSink *videoSink;
    QLabel *videoLabel;
    ToolBar *toolBar;
    ControlBar *controlBar;

    bool grayscaleActive = false;
    bool gaussianActive = false;
    bool sepiaActive = false;
    bool negativeActive = false;
    bool sobelActive = false;
    QImage lastFrame;
};
