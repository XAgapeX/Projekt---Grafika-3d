#pragma once
#define TOOLBAR_H

#include <QPushButton>
#include <QHBoxLayout>
#include <QLabel>

/**
 * @class ToolBar
 * @brief Pasek narzędzi wykorzystywany w oknie odtwarzacza wideo.
 *
 * Odpowiada za wyświetlanie przycisków filtrów obrazu, suwaków
 * regulacji jasności i kontrastu oraz przycisku otwierającego nowe wideo.
 *
 * Emituje sygnały powiązane z kliknięciami filtrów oraz zmianą parametrów obrazu,
 * które następnie są obsługiwane w VideoWindow.
 */
class ToolBar : public QWidget {
    Q_OBJECT

public:
    /**
     * @brief Konstruktor paska narzędzi.
     *
     * Tworzy wszystkie przyciski filtrów, suwaki jasności i kontrastu,
     * przycisk „Open Another Video”, oraz buduje układ poziomy
     * zawierający wszystkie elementy interfejsu.
     *
     * @param parent rodzic widgetu (domyślnie nullptr)
     */
    explicit ToolBar(QWidget *parent = nullptr);

    signals:
        /** @brief Sygnał emitowany po kliknięciu przycisku otwierania nowego wideo. */
        void openAnotherVideoClicked();

    /** @brief Sygnał aktywujący filtr grayscale. */
    void grayscaleFilterClicked();

    /** @brief Sygnał aktywujący filtr Gaussian Blur. */
    void gaussianFilterClicked();

    /** @brief Sygnał aktywujący filtr Sepia. */
    void sepiaFilterClicked();

    /** @brief Sygnał aktywujący filtr Negative. */
    void negativeFilterClicked();

    /** @brief Sygnał aktywujący filtr Sobel. */
    void sobelFilterClicked();

    /** @brief Emitowany, gdy zmienia się jasność. */
    void brightnessChanged(int value);

    /** @brief Emitowany, gdy zmienia się kontrast. */
    void contrastChanged(int value);

    /** @brief Sygnał aktywujący filtr Cartoon. */
    void cartoonFilterClicked();

private:
    QPushButton *openTest;   ///< Przycisk „Open Another Video”.
    QLabel *filtersLabel;    ///< Etykieta „Filters :”.
};
