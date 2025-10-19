#include "../header files/VideoWindow.h"
#include <QVBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QMediaPlayer>
#include <QVideoWidget>
#include <QAudioOutput>

VideoWindow::VideoWindow(QWidget *parent)
    : QWidget(parent)
{
    setWindowTitle("Odtwarzacz wideo");
    resize(640, 480);

    player = new QMediaPlayer(this);
    videoWidget = new QVideoWidget(this);

    audioOutput = new QAudioOutput(this);
    player->setVideoOutput(videoWidget);
    player->setAudioOutput(audioOutput);
    audioOutput->setVolume(50);

    loadButton = new QPushButton("Wybierz film", this);

    auto *layout = new QVBoxLayout(this);
    layout->addWidget(videoWidget);
    layout->addWidget(loadButton);

    connect(loadButton, &QPushButton::clicked, this, &VideoWindow::loadVideo);
}

void VideoWindow::loadVideo() {
    QString file = QFileDialog::getOpenFileName(
        this, "Wybierz plik wideo", QString(),
        "Wideo (*.mp4 *.avi *.mkv *.mov)");

    if (!file.isEmpty()) {
        player->setSource(QUrl::fromLocalFile(file));
        player->play();
    }
}
