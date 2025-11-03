#include "../header files/VideoWindow.h"
#include "../header files/ToolBar.h"
#include "../header files/ControlBar.h"
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

    toolBar = new ToolBar(this);
    toolBar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    toolBar->setMaximumHeight(50);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(toolBar);
    layout->addWidget(videoWidget);
    setLayout(layout);

    controlBar = new ControlBar(player, this);
    layout->addWidget(controlBar);

    setContentsMargins(0,0,0,0);
    setStyleSheet("margin:0; padding:0;");

    connect(toolBar, &ToolBar::openAnotherVideoClicked, this, &VideoWindow::loadVideo);

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
