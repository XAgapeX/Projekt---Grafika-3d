#include "../header files/MainWindow.h"
#include "../header files/VideoWindow.h"
#include <QPushButton>
#include <QLabel>
#include <QIcon>
#include <QVBoxLayout>
#include <QStackedWidget>
#include <QPixmap>

/**
 * @brief Konstruktor klasy MainWindow.
 *
 * Tworzy główne okno aplikacji odtwarzacza wideo, inicjalizuje interfejs
 * użytkownika, ustawia ikonę, tytuł oraz buduje układ widgetów.
 *
 * Funkcja:
 * - ustawia parametry okna (tytuł, ikona, rozmiar, tło),
 * - tworzy sekcję nagłówkową (ikona + etykieta + przycisk "Open Video"),
 * - tworzy QStackedWidget zawierający ekran startowy i okno odtwarzacza,
 * - łączy przycisk otwierania filmu z przełączeniem widoku i wczytaniem wideo.
 *
 * @param parent Wskaźnik na rodzica widgetu (domyślnie nullptr).
 */

MainWindow::MainWindow(QWidget *parent)
    : QWidget(parent)
{
    setWindowTitle("Video Player");
    setWindowIcon(QIcon(":/resources/icons/play.png"));
    resize(1200, 800);

    setStyleSheet("background-color: #f5f5f5;");

    QLabel *iconLabel = new QLabel(this);
    QPixmap pixmap(":/resources/icons/play.png");
    iconLabel->setPixmap(pixmap.scaled(64, 64, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    iconLabel->setAlignment(Qt::AlignCenter);

    QLabel *label = new QLabel("Video Player", this);
    label->setAlignment(Qt::AlignCenter);
    label->setStyleSheet("font-size: 24px; font-weight: bold; color: #21618c;");

    openButton = new QPushButton("Open Video", this);
    openButton->setFixedSize(120, 40);
    openButton->setStyleSheet(
        "background-color: #21618c; color: white; border-radius: 5px; font-size: 16px; font-weight: bold;");

    QVBoxLayout *headerLayout = new QVBoxLayout();
    headerLayout->setContentsMargins(0,0,0,0);
    headerLayout->setSpacing(10);
    headerLayout->addWidget(iconLabel, 0, Qt::AlignCenter);
    headerLayout->addWidget(label, 0, Qt::AlignCenter);
    headerLayout->addWidget(openButton, 0, Qt::AlignCenter);

    QWidget *headerWidget = new QWidget(this);
    headerWidget->setLayout(headerLayout);
    headerWidget->setStyleSheet("background-color: #f5f5f5;");


    stackedWidget = new QStackedWidget(this);


    QWidget *homeWidget = new QWidget(this);
    homeWidget->setStyleSheet("background-color: #f5f5f5;");
    QVBoxLayout *homeLayout = new QVBoxLayout(homeWidget);
    homeLayout->setContentsMargins(0,0,0,0);
    homeLayout->setSpacing(0);
    homeLayout->addWidget(headerWidget, 0, Qt::AlignCenter);
    homeWidget->setLayout(homeLayout);


    videoWindow = new VideoWindow(this);
    videoWindow->setStyleSheet("background-color: black;");


    stackedWidget->addWidget(homeWidget);
    stackedWidget->addWidget(videoWindow);


    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0,0,0,0);
    mainLayout->setSpacing(0);
    mainLayout->addWidget(stackedWidget);
    setLayout(mainLayout);


    connect(openButton, &QPushButton::clicked, this, [this]() {
        stackedWidget->setCurrentWidget(videoWindow);
        videoWindow->loadVideo();
    });
}
