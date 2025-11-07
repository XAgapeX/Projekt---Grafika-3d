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


private:
    QPushButton *openTest;
    QLabel *filtersLabel;
};

#endif // TOOLBAR_H
