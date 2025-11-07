#include "../header files/VideoWindow.h"
#include "../header files/ToolBar.h"
#include "../header files/ControlBar.h"

#include <QFileDialog>
#include <QVBoxLayout>
#include <QPixmap>
#include <QScreen>
#include <QGuiApplication>
#include <QResizeEvent>

VideoWindow::VideoWindow(QWidget *parent)
    : QWidget(parent)
{
    player = new QMediaPlayer(this);
    audioOutput = new QAudioOutput(this);
    player->setAudioOutput(audioOutput);
    audioOutput->setVolume(50);

    videoLabel = new QLabel(this);
    videoLabel->setAlignment(Qt::AlignCenter);
    videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    videoLabel->setScaledContents(false);

    videoSink = new QVideoSink(this);
    player->setVideoSink(videoSink);
    connect(videoSink, &QVideoSink::videoFrameChanged, this, &VideoWindow::onFrameAvailable);

    toolBar = new ToolBar(this);
    controlBar = new ControlBar(player, this);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(toolBar);
    layout->addWidget(videoLabel);
    layout->addWidget(controlBar);
    setLayout(layout);

    QSize screenSize = QGuiApplication::primaryScreen()->availableSize();
    resize(screenSize.width() * 0.6, screenSize.height() * 0.6);

    connect(toolBar, &ToolBar::openAnotherVideoClicked, this, &VideoWindow::loadVideo);
    connect(toolBar, &ToolBar::grayscaleFilterClicked, this, &VideoWindow::applyGrayscaleFilter);
}

void VideoWindow::loadVideo() {
    QFileDialog dialog(this);
    dialog.setWindowTitle("Wybierz plik wideo");
    dialog.setNameFilter("Wideo (*.mp4 *.avi *.mkv *.mov)");
    if (dialog.exec() == QDialog::Accepted) {
        QString file = dialog.selectedFiles().first();
        if (!file.isEmpty()) {
            player->setSource(QUrl::fromLocalFile(file));
            player->play();
        }
    }
}

void VideoWindow::applyGrayscaleFilter() {
    grayscaleActive = !grayscaleActive;
}

void VideoWindow::onFrameAvailable(const QVideoFrame &frame) {
    if (!frame.isValid()) return;

    QVideoFrame copy(frame);
    if (!copy.map(QVideoFrame::ReadOnly))
        return;

    QImage img = copy.toImage().convertToFormat(QImage::Format_RGB888);
    copy.unmap();

    if (grayscaleActive) {
        QImage gray(img.width(), img.height(), QImage::Format_Grayscale8);
        applyGrayscaleCUDA(img.bits(), gray.bits(), img.width(), img.height());
        lastFrame = gray;
    } else {
        lastFrame = img;
    }


    QPixmap scaledPixmap = QPixmap::fromImage(lastFrame).scaled(
        videoLabel->size(),
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation
    );
    videoLabel->setPixmap(scaledPixmap);
}


void VideoWindow::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
    if (!lastFrame.isNull()) {
        QPixmap scaledPixmap = QPixmap::fromImage(lastFrame).scaled(
            videoLabel->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        );
        videoLabel->setPixmap(scaledPixmap);
    }
}
