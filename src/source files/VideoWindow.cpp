#include "../header files/VideoWindow.h"
#include "../header files/ToolBar.h"
#include "../header files/ControlBar.h"

#include "../cuda/grayscale.cuh"
#include "../cuda/gaussianblur.cuh"
#include "../cuda/sepia.cuh"
#include "../cuda/negative.cuh"
#include "../cuda/sobel.cuh"
#include "../cuda/brightness.cuh"
#include "../cuda/contrast.cuh"
#include "../cuda/cartoon.cuh"

#include <QFileDialog>
#include <QVBoxLayout>
#include <QPixmap>
#include <QScreen>
#include <QGuiApplication>
#include <QResizeEvent>
#include <QDebug>

/**
 * @brief Konstruktor VideoWindow.
 *
 * Inicjalizuje odtwarzacz wideo, wyjście audio, odbiornik klatek,
 * widget wyświetlający obraz oraz paski ToolBar i ControlBar.
 *
 * Ustawia połączenia sygnałów filtrów oraz suwaków jasności i kontrastu.
 * Przetwarzanie klatek odbywa się w metodzie onFrameAvailable().
 *
 * @param parent rodzic widgetu
 */
VideoWindow::VideoWindow(QWidget *parent)
    : QWidget(parent),
      grayscaleActive(false),
      gaussianActive(false),
      sepiaActive(false),
      negativeActive(false),
      sobelActive(false),
      cartoonActive(false)
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
    connect(toolBar, &ToolBar::gaussianFilterClicked, this, &VideoWindow::applyGaussianFilter);
    connect(toolBar, &ToolBar::sepiaFilterClicked, this, &VideoWindow::applySepiaFilter);
    connect(toolBar, &ToolBar::negativeFilterClicked, this, &VideoWindow::applyNegativeFilter);
    connect(toolBar, &ToolBar::sobelFilterClicked, this, &VideoWindow::applySobelFilter);
    connect(toolBar, &ToolBar::brightnessChanged, this, [this](int v){brightnessValue = v - 50; });
    connect(toolBar, &ToolBar::contrastChanged, this, [this](int v){contrastValue = 0.5f + (v / 100.0f);});
    connect(toolBar, &ToolBar::cartoonFilterClicked, this, &VideoWindow::applyCartoonFilter);
}

/**
 * @brief Otwiera okno wyboru pliku i rozpoczyna odtwarzanie.
 */
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

/**
 * @brief Przełącza filtr grayscale.
 */
void VideoWindow::applyGrayscaleFilter() {
    grayscaleActive = !grayscaleActive;
}

/**
 * @brief Przełącza filtr Gaussian Blur.
 */
void VideoWindow::applyGaussianFilter() {
    gaussianActive = !gaussianActive;
}

/**
 * @brief Przełącza filtr Sepia.
 */
void VideoWindow::applySepiaFilter() {
    sepiaActive = !sepiaActive;
}

/**
 * @brief Przełącza filtr Negative.
 */
void VideoWindow::applyNegativeFilter() {
    negativeActive = !negativeActive;
}

/**
 * @brief Przełącza filtr Sobel.
 */
void VideoWindow::applySobelFilter() {
    sobelActive = !sobelActive;
}

/**
 * @brief Przełącza filtr Cartoon.
 */
void VideoWindow::applyCartoonFilter() {
    cartoonActive = !cartoonActive;
}

/**
 * @brief Przetwarzanie klatki RGB poprzez filtry CUDA i wyświetlenie wyniku.
 *
 * Kolejność obróbki:
 * 1. Jasność (CUDA)
 * 2. Kontrast (CUDA)
 * 3. Grayscale (CUDA)
 * 4. Gaussian Blur (CUDA)
 * 5. Sepia (CUDA)
 * 6. Negative (CUDA)
 * 7. Sobel (CUDA)
 * 8. Cartoon (CUDA)
 *
 * Wynik jest zapisywany do lastFrame i skalowany do wymiarów videoLabel.
 *
 * @param frame klatka odebrana z QVideoSink
 */
void VideoWindow::onFrameAvailable(const QVideoFrame &frame) {
    if (!frame.isValid()) return;

    QVideoFrame copy(frame);
    if (!copy.map(QVideoFrame::ReadOnly))
        return;

    QImage img = copy.toImage().convertToFormat(QImage::Format_RGB888);
    copy.unmap();

    QImage processed = img;

    if (brightnessValue != 0) {
        QImage out(img.width(), img.height(), QImage::Format_RGB888);
        applyBrightnessCUDA(processed.bits(), out.bits(),
                            img.width(), img.height(),
                            img.bytesPerLine(),
                            brightnessValue * 3);
        processed = out;
    }

    if (contrastValue != 1.0f) {
        QImage out(img.width(), img.height(), QImage::Format_RGB888);
        applyContrastCUDA(processed.bits(), out.bits(),
                           img.width(), img.height(),
                           img.bytesPerLine(),
                           contrastValue);
        processed = out;
    }

    if (grayscaleActive) {
        QImage grayY(img.width(), img.height(), QImage::Format_Grayscale8);
        QImage grayCb(img.width(), img.height(), QImage::Format_Grayscale8);
        QImage grayCr(img.width(), img.height(), QImage::Format_Grayscale8);

        applyGrayscaleCUDA(processed.bits(),
                           grayY.bits(),
                           grayCb.bits(),
                           grayCr.bits(),
                           img.width(),
                           img.height(),
                           img.bytesPerLine());

        processed = grayY.convertToFormat(QImage::Format_RGB888);
    }

    if (gaussianActive) {
        QImage blurred(img.width(), img.height(), QImage::Format_RGB888);
        applyGaussianBlurCUDA(processed.bits(), blurred.bits(),
                              img.width(), img.height(),
                              img.bytesPerLine());
        processed = blurred;
    }

    if (sepiaActive) {
        QImage sepia(img.width(), img.height(), QImage::Format_RGB888);
        applySepiaCUDA(processed.bits(), sepia.bits(),
                       img.width(), img.height(),
                       img.bytesPerLine());
        processed = sepia;
    }

    if (negativeActive) {
        QImage negative(img.width(), img.height(), QImage::Format_RGB888);
        applyNegativeCUDA(processed.bits(), negative.bits(),
                          img.width(), img.height(),
                          img.bytesPerLine());
        processed = negative;
    }

    if (sobelActive) {
        QImage sobel(img.width(), img.height(), QImage::Format_RGB888);
        applySobelCUDA(processed.bits(), sobel.bits(),
            img.width(), img.height(),img.bytesPerLine());
        processed = sobel;
    }

    if (cartoonActive) {
        QImage cartoon(img.width(), img.height(), QImage::Format_RGB888);
        applyCartoonCUDA(processed.bits(), cartoon.bits(),
                         img.width(), img.height(), img.bytesPerLine());
        processed = cartoon;
    }

    lastFrame = processed;

    QPixmap scaledPixmap = QPixmap::fromImage(lastFrame).scaled(
        videoLabel->size(),
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation
    );
    videoLabel->setPixmap(scaledPixmap);
}

/**
 * @brief Dopasowuje obraz po zmianie rozmiaru okna.
 *
 * Skaluje lastFrame z zachowaniem proporcji.
 *
 * @param event zdarzenie zmiany rozmiaru
 */
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
