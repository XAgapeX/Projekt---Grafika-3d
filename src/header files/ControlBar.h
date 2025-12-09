#ifndef CONTROLBAR_H
#define CONTROLBAR_H

#include <QWidget>
#include <QPushButton>
#include <QSlider>
#include <QHBoxLayout>
#include <QLabel>
#include <QMediaPlayer>

/**
 * @class ControlBar
 * @brief Pasek sterowania odtwarzaczem (play/pause, czas, głośność).
 *
 * Odpowiada za interakcję z QMediaPlayer:
 * - odtwarzanie/pauza
 * - restart
 * - zmiana głośności
 * - suwak postępu
 * - wyświetlanie czasu
 */
class ControlBar : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief Konstruktor paska sterowania.
     * @param player wskaźnik na QMediaPlayer, który ma być kontrolowany
     * @param parent widget nadrzędny
     */
    explicit ControlBar(QMediaPlayer *player, QWidget *parent = nullptr);

private:
    QPushButton *playPauseButton;   ///< Przycisk play/pause
    QPushButton *refreshButton;     ///< Przycisk restartu nagrania
    QLabel *volumeIcon;             ///< Ikona głośności
    QSlider *volumeSlider;          ///< Suwak głośności
    QSlider *positionSlider;        ///< Suwak pozycji nagrania
    QLabel *timeLabel;              ///< Tekst czasu "MM:SS / MM:SS"
    QMediaPlayer *player;           ///< Odtwarzacz kontrolowany przez pasek

    QIcon playIcon;                 ///< Ikona "play"
    QIcon pauseIcon;                ///< Ikona "pause"
    QIcon refreshIcon;              ///< Ikona odświeżenia
    QPixmap volumePixmap;           ///< Ikona głośności w formacie QPixmap

private slots:
    /**
     * @brief Przełącza odtwarzanie/pauzę.
     */
    void togglePlayPause();

    /**
     * @brief Aktualizuje pozycję suwaka oraz tekst czasu.
     * @param position aktualna pozycja w milisekundach
     */
    void updatePosition(qint64 position);

    /**
     * @brief Ustawia zakres suwaka pozycji po załadowaniu nagrania.
     * @param duration długość utworu w milisekundach
     */
    void updateDuration(qint64 duration);

    /**
     * @brief Przewija nagranie.
     * @param value nowa pozycja
     */
    void seek(int value);

    /**
     * @brief Zmiana głośności.
     * @param value wartość 0–100
     */
    void changeVolume(int value);
};

#endif // CONTROLBAR_H
