#include "../header files/ToolBar.h"
#include <QLabel>
#include <QSlider>
/**
 * @brief Konstruktor klasy ToolBar.
 *
 * Inicjalizuje cały pasek narzędziowy: tworzy przycisk otwierania nowego wideo,
 * przyciski filtrów obrazu (grayscale, gaussian blur, sepia, negative, sobel, cartoon),
 * suwaki regulacji jasności i kontrastu, a następnie ustawia układ QHBoxLayout
 * umieszczający wszystkie elementy w jednym poziomym pasku.
 *
 * Każdy przycisk filtra jest połączony sygnałem z odpowiednią funkcją emitującą
 * sygnał dostępny w sekcji signals.
 *
 * Suwaki jasności i kontrastu wysyłają sygnały z aktualną wartością (0–100),
 * które mogą zostać odczytane w VideoWindow lub innym komponencie.
 *
 * @param parent rodzic widgetu (opcjonalny)
 */
ToolBar::ToolBar(QWidget *parent) : QWidget(parent) {
    setAttribute(Qt::WA_StyledBackground, true);
    setStyleSheet("ToolBar { background-color: #21618c !important; }");

    openTest = new QPushButton("Open Another Video", this);
    openTest->setStyleSheet(
        "background-color: white; color: #21618c; border-radius: 5px; "
        "font-size: 16px; font-weight: bold;");
    openTest->setMaximumHeight(40);
    openTest->setFixedWidth(180);

    filtersLabel = new QLabel("Filters :", this);
    filtersLabel->setStyleSheet(
        "background-color:transparent; color: white; font-size: 16px; font-weight: bold;");

    QPushButton *filter1 = new QPushButton("Grayscale", this);
    connect(filter1, &QPushButton::clicked, this, &ToolBar::grayscaleFilterClicked);

    QPushButton *filter2 = new QPushButton("Gaussian Blur", this);
    connect(filter2, &QPushButton::clicked, this, &ToolBar::gaussianFilterClicked);

    QPushButton *filter3 = new QPushButton("Sepia", this);
    connect(filter3, &QPushButton::clicked, this, &ToolBar::sepiaFilterClicked);

    QPushButton *filter4 = new QPushButton("Negative", this);
    connect(filter4, &QPushButton::clicked, this, &ToolBar::negativeFilterClicked);

    QPushButton *filter5 = new QPushButton("Sobel", this);
    connect(filter5, &QPushButton::clicked, this, &ToolBar::sobelFilterClicked);

    QLabel *filter6_label = new QLabel("Brightness and Contrast:", this);
    filter6_label->setStyleSheet(
        "background-color:transparent; color: white; font-size: 16px; font-weight: bold;");

    QPushButton *filter6 = new QPushButton("Cartoon", this);
    connect(filter6, &QPushButton::clicked, this, &ToolBar::cartoonFilterClicked);

    QList<QPushButton *> filterButtons = {filter1, filter2, filter3, filter4, filter5, filter6};
    for (auto *btn : filterButtons) {
        btn->setMinimumHeight(30);
        btn->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        btn->setStyleSheet(
            "background-color: white; "
            "color: #21618c; "
            "border-radius: 5px; "
            "font-weight: bold;"
        );
    }

    QHBoxLayout *layout = new QHBoxLayout(this);
    layout->setContentsMargins(10, 5, 10, 5);
    layout->setSpacing(10);

    layout->addWidget(openTest);
    layout->addWidget(filtersLabel);

    for (auto *btn : filterButtons)
        layout->addWidget(btn);

    layout->addWidget(filter6_label);

    QLabel *brightnessIcon = new QLabel(this);
    brightnessIcon->setAttribute(Qt::WA_TranslucentBackground);
    brightnessIcon->setPixmap(QPixmap(":/resources/icons/brightness.png")
                                  .scaled(24, 24, Qt::KeepAspectRatio, Qt::SmoothTransformation));

    QSlider *brightnessSlider = new QSlider(Qt::Horizontal, this);
    brightnessSlider->setRange(0, 100);
    brightnessSlider->setValue(50);
    brightnessSlider->setFixedWidth(100);
    connect(brightnessSlider, &QSlider::valueChanged,this, &ToolBar::brightnessChanged);


    QLabel *contrastIcon = new QLabel(this);
    contrastIcon->setAttribute(Qt::WA_TranslucentBackground);
    contrastIcon->setPixmap(QPixmap(":/resources/icons/contrast.png")
                                 .scaled(24, 24, Qt::KeepAspectRatio, Qt::SmoothTransformation));

    QSlider *contrastSlider = new QSlider(Qt::Horizontal, this);
    contrastSlider->setRange(0, 100);
    contrastSlider->setValue(50);
    contrastSlider->setFixedWidth(100);
    connect(contrastSlider, &QSlider::valueChanged,this, &ToolBar::contrastChanged);

    QHBoxLayout *slidersLayout = new QHBoxLayout();

    QString sliderStyle = R"(
    QSlider { background: transparent; }
    QSlider::groove:horizontal {
        height: 8px;
        background: white;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: white;
        border: 1px solid #21618c;
        width: 16px;
        margin: -4px 0;
        border-radius: 8px;
    })";

    brightnessSlider->setStyleSheet(sliderStyle);
    contrastSlider->setStyleSheet(sliderStyle);

    slidersLayout->addWidget(brightnessIcon);
    slidersLayout->addWidget(brightnessSlider);

    slidersLayout->addSpacing(10);

    slidersLayout->addWidget(contrastIcon);
    slidersLayout->addWidget(contrastSlider);

    layout->addLayout(slidersLayout);

    layout->addStretch();

    connect(openTest, &QPushButton::clicked, this, &ToolBar::openAnotherVideoClicked);
}
