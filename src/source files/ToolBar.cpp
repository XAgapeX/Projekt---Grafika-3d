#include "../header files/ToolBar.h"
#include <QLabel>
#include <QPalette>


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

    QList<QPushButton*> filterButtons = {filter1, filter2, filter3, filter4, filter5};
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

    layout->addStretch();

    setBackgroundRole(QPalette::Window);
    setAutoFillBackground(true);
    setLayout(layout);

    connect(openTest, &QPushButton::clicked, this, &ToolBar::openAnotherVideoClicked);
}
