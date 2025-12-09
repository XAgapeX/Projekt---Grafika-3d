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

/**
 * @class VideoWindow
 * @brief Główne okno wyświetlania i przetwarzania wideo.
 *
 * Odpowiada za:
 * - odtwarzanie wideo (QMediaPlayer + QVideoSink),
 * - prezentację aktualnej klatki w QLabel,
 * - stosowanie filtrów CUDA w czasie rzeczywistym (grayscale, gaussian blur,
 *   sepia, negative, sobel, cartoon),
 * - regulację jasności i kontrastu,
 * - integrację z ToolBar i ControlBar.
 *
 * Klasa przetwarza każdą klatkę obrazu w metodzie onFrameAvailable()
 * i renderuje wynik na ekranie.
 */
class VideoWindow : public QWidget {
    Q_OBJECT
public:

    /**
     * @brief Konstruktor okna odtwarzacza wideo.
     *
     * Inicjalizuje QMediaPlayer, QAudioOutput, QVideoSink oraz QLabel,
     * tworzy pasek narzędzi (ToolBar) i pasek sterowania (ControlBar),
     * ustawia układ pionowy oraz łączy wszystkie sygnały filtrów i suwaków.
     *
     * @param parent rodzic widgetu (opcjonalny)
     */
    explicit VideoWindow(QWidget *parent = nullptr);

    /**
     * @brief Otwiera okno dialogowe wyboru pliku i rozpoczyna odtwarzanie wideo.
     */
    void loadVideo();

protected:
    /**
     * @brief Reimplementacja zdarzenia zmiany rozmiaru.
     *
     * Po zmianie rozmiaru okna dopasowuje wyświetlaną klatkę,
     * aby zachować proporcje obrazu.
     *
     * @param event obiekt zdarzenia zmiany rozmiaru
     */
    void resizeEvent(QResizeEvent *event) override;

private slots:

    /** @brief Włącza/wyłącza filtr Grayscale (CUDA). */
    void applyGrayscaleFilter();

    /** @brief Włącza/wyłącza filtr Gaussian Blur (CUDA). */
    void applyGaussianFilter();

    /** @brief Włącza/wyłącza filtr Sepia (CUDA). */
    void applySepiaFilter();

    /** @brief Włącza/wyłącza filtr Negative (CUDA). */
    void applyNegativeFilter();

    /** @brief Włącza/wyłącza filtr Sobel (CUDA). */
    void applySobelFilter();

    /** @brief Włącza/wyłącza filtr Cartoon (CUDA). */
    void applyCartoonFilter();

    /**
     * @brief Przetwarzanie klatki wideo.
     *
     * Odbiera klatkę z QVideoSink, konwertuje ją na QImage, stosuje aktywne filtry
     * (jasność, kontrast oraz dowolne filtry CUDA), zapisuje wynik do lastFrame
     * oraz aktualizuje wyświetlany obraz.
     *
     * @param frame odebrana klatka wideo
     */
    void onFrameAvailable(const QVideoFrame &frame);

private:
    QMediaPlayer *player;        ///< Silnik odtwarzania multimediów.
    QAudioOutput *audioOutput;   ///< Wyjście audio.
    QVideoSink *videoSink;       ///< Odbiornik klatek z QMediaPlayer.
    QLabel *videoLabel;          ///< Widok wyświetlający aktualną klatkę.
    ToolBar *toolBar;            ///< Pasek filtrów i regulacji.
    ControlBar *controlBar;      ///< Pasek sterowania wideo.

    int brightnessValue = 0;     ///< Wartość jasności (-50 do +50).
    float contrastValue = 1.0f;  ///< Wartość kontrastu (0.5–1.5).

    bool grayscaleActive = false; ///< Czy filtr grayscale jest aktywny.
    bool gaussianActive = false;  ///< Czy filtr gaussian blur jest aktywny.
    bool sepiaActive = false;     ///< Czy filtr sepia jest aktywny.
    bool negativeActive = false;  ///< Czy filtr negative jest aktywny.
    bool sobelActive = false;     ///< Czy filtr sobel jest aktywny.
    bool cartoonActive = false;   ///< Czy filtr cartoon jest aktywny.

    QImage lastFrame;             ///< Ostatnia przetworzona klatka.
};

