#include "../header files/VideoWindow.h"
#include <QFileDialog>
#include <QVBoxLayout>

VideoWindow::VideoWindow(QWidget *parent)
    : QWidget(parent)
{
    player = new QMediaPlayer(this);
    videoWidget = new QVideoWidget(this);
    audioOutput = new QAudioOutput(this);

    player->setVideoOutput(videoWidget);
    player->setAudioOutput(audioOutput);
    audioOutput->setVolume(50);

    auto *layout = new QVBoxLayout(this);
    layout->addWidget(videoWidget);
    setLayout(layout);
}

void VideoWindow::loadVideo() {
    QFileDialog dialog(this);
    dialog.setWindowTitle("Wybierz plik wideo");
    dialog.setNameFilter("Wideo (*.mp4 *.avi *.mkv *.mov)");
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);
    dialog.setStyleSheet("background-color: white; color: black;");

    if (dialog.exec() == QDialog::Accepted) {
        QString file = dialog.selectedFiles().first();
        if (!file.isEmpty()) {
            player->setSource(QUrl::fromLocalFile(file));
            player->play();
        }
    }
}
