#ifndef TOOLBAR_H
#define TOOLBAR_H

#include <QPushButton>
#include <QHBoxLayout>
#include <QLabel>


class VideoWindow;

class ToolBar : public QWidget {
    Q_OBJECT

public:
    explicit ToolBar(QWidget *parent = nullptr);

signals:
    void openAnotherVideoClicked();
    void grayscaleFilterClicked();
    void gaussianFilterClicked();
    void sepiaFilterClicked();
    void negativeFilterClicked();
    void sobelFilterClicked();
    void brightnessChanged(int value);
    void contrastChanged(int value);
    void cartoonFilterClicked();



private:
    QPushButton *openTest;
    QLabel *filtersLabel;
};

#endif // TOOLBAR_H
