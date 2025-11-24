#ifndef CONTROLBAR_H
#define CONTROLBAR_H

#include <QWidget>
#include <QPushButton>
#include <QSlider>
#include <QHBoxLayout>
#include <QLabel>
#include <QMediaPlayer>


class ControlBar : public QWidget
{
    Q_OBJECT

public:
    explicit ControlBar(QMediaPlayer *player, QWidget *parent = nullptr);

private:
    QPushButton *playPauseButton;
    QPushButton *refreshButton;
    QLabel *volumeIcon;
    QSlider *volumeSlider;
    QSlider *positionSlider;
    QLabel *timeLabel;
    QMediaPlayer *player;

    QIcon playIcon;
    QIcon pauseIcon;
    QIcon refreshIcon;
    QPixmap volumePixmap;

private slots:
    void togglePlayPause();
    void updatePosition(qint64 position);
    void updateDuration(qint64 duration);
    void seek(int value);
    void changeVolume(int value);
};

#endif // CONTROLBAR_H
