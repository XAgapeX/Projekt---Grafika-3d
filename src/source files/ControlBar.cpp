#include "../header files/ControlBar.h"
#include <QAudioOutput>
#include <QTimer>
#include <QStyle>
#include <QLabel>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSlider>

/**
 * @brief Tworzy pasek sterowania i podłącza go do QMediaPlayer.
 *
 * Inicjalizowane elementy:
 * - play/pause
 * - restart
 * - czas + suwak pozycji
 * - ikona + suwak głośności
 *
 * Ustawiane są połączenia:
 * - zmiana pozycji
 * - zmiana czasu trwania
 * - kliknięcie play/pause
 * - zmiana głośności
 *
 * @param player wskaźnik na QMediaPlayer
 * @param parent widget rodzica
 */
ControlBar::ControlBar(QMediaPlayer *player, QWidget *parent)
    : QWidget(parent), player(player)
{
    setStyleSheet("background-color: #21618c; color: white;");
    setFixedHeight(40);

    playPauseButton = new QPushButton(this);
    refreshButton = new QPushButton(this);
    volumeIcon = new QLabel(this);
    volumeSlider = new QSlider(Qt::Horizontal, this);
    volumeSlider->setStyleSheet("background-color: transparent; border: none;");
    positionSlider = new QSlider(Qt::Horizontal, this);
    positionSlider->setStyleSheet("background-color: transparent; border: none;");
    timeLabel = new QLabel("00:00 / 00:00", this);
    timeLabel->setStyleSheet("color: #21618c; background-color: transparent;");

    playIcon = QIcon(":/resources/icons/play-button-arrowhead.png");
    pauseIcon = QIcon(":/resources/icons/pause.png");
    refreshIcon = QIcon(":/resources/icons/refersh.png");
    volumePixmap = QPixmap(":/resources/icons/volume.png");

    playPauseButton->setIcon(playIcon);
    playPauseButton->setIconSize(QSize(24, 24));
    playPauseButton->setStyleSheet("background-color: transparent; border: none;");

    refreshButton->setIcon(refreshIcon);
    refreshButton->setIconSize(QSize(24, 24));
    refreshButton->setStyleSheet("background-color: transparent; border: none;");

    volumeIcon->setPixmap(volumePixmap.scaled(20, 20, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    volumeIcon->setStyleSheet("background-color: transparent; border: none;");

    volumeSlider->setRange(0, 40);
    volumeSlider->setValue(player->audioOutput()->volume() * 100);

    positionSlider->setRange(0, 0);

    QHBoxLayout *layout = new QHBoxLayout(this);
    layout->setContentsMargins(5, 5, 5, 5);
    layout->addWidget(playPauseButton);
    layout->addWidget(refreshButton);
    layout->addWidget(timeLabel);
    layout->addWidget(positionSlider);
    layout->addWidget(volumeIcon);
    layout->addWidget(volumeSlider);
    setLayout(layout);

    connect(playPauseButton, &QPushButton::clicked, this, &ControlBar::togglePlayPause);
    connect(refreshButton, &QPushButton::clicked, player, &QMediaPlayer::stop);
    connect(volumeSlider, &QSlider::valueChanged, this, &ControlBar::changeVolume);
    connect(positionSlider, &QSlider::sliderMoved, this, &ControlBar::seek);
    connect(player, &QMediaPlayer::positionChanged, this, &ControlBar::updatePosition);
    connect(player, &QMediaPlayer::durationChanged, this, &ControlBar::updateDuration);
}

/**
 * @brief Przełącza odtwarzanie lub pauzę oraz zmienia ikonę.
 */
void ControlBar::togglePlayPause()
{
    if (player->playbackState() == QMediaPlayer::PlayingState) {
        player->pause();
        QTimer::singleShot(50, this, [this]() {
            playPauseButton->setIcon(playIcon);
        });
    } else {
        player->play();
        QTimer::singleShot(50, this, [this]() {
            playPauseButton->setIcon(pauseIcon);
        });
    }
}

/**
 * @brief Aktualizuje pozycję i tekst czasu wyświetlany na pasku.
 * @param position aktualny czas wideo w milisekundach
 */
void ControlBar::updatePosition(qint64 position)
{
    positionSlider->blockSignals(true);
    positionSlider->setValue(static_cast<int>(position));
    positionSlider->blockSignals(false);

    int sec = position / 1000;
    int min = sec / 60;
    sec = sec % 60;

    int totalSec = player->duration() / 1000;
    int totalMin = totalSec / 60;
    totalSec = totalSec % 60;

    timeLabel->setText(QString("%1:%2 / %3:%4")
                       .arg(min, 2, 10, QChar('0'))
                       .arg(sec, 2, 10, QChar('0'))
                       .arg(totalMin, 2, 10, QChar('0'))
                       .arg(totalSec, 2, 10, QChar('0')));
}

/**
 * @brief Ustawia zakres suwaka postępu.
 * @param duration długość nagrania (ms)
 */
void ControlBar::updateDuration(qint64 duration)
{
    positionSlider->setRange(0, static_cast<int>(duration));
}

/**
 * @brief Przewija nagranie.
 * @param value nowa pozycja w milisekundach
 */
void ControlBar::seek(int value)
{
    player->setPosition(value);
}

/**
 * @brief Zmiana głośności odtwarzacza.
 * @param value wartość 0–100
 */
void ControlBar::changeVolume(int value)
{
    player->audioOutput()->setVolume(value / 100.0);
}
