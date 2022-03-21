#include "breast_cancer_classifier.h"
#include "./ui_breast_cancer_classifier.h"

#include <QScreen>
#include <QStandardPaths>
#include <QImageReader>
#include <QImageWriter>
#include <QMessageBox>
#include <QScrollBar>
#include <QColorSpace>

#include <iostream>

BreastCancerClassifier::BreastCancerClassifier(QWidget *parent)
    : QMainWindow(parent), ui_(new Ui::BreastCancerClassifier)
{
    ui_->setupUi(this);

    imageLabel_ = ui_->imageLabel;
    scrollArea_ = ui_->scrollArea;

    imageLabel_->setBackgroundRole(QPalette::Base);
    imageLabel_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    imageLabel_->setScaledContents(true);

    scrollArea_->setBackgroundRole(QPalette::Dark);
    scrollArea_->setWidget(imageLabel_);
    imageLabel_->setVisible(false);
    scrollArea_->setVisible(true);
    createActions();
}

BreastCancerClassifier::~BreastCancerClassifier()
{
    delete ui_;
}

void BreastCancerClassifier::open()
{
    QFileDialog dialog(this, tr("Open File"));
    initializeImageFileDialog(dialog, QFileDialog::AcceptOpen);

    while (dialog.exec() == QDialog::Accepted && !loadFile(dialog.selectedFiles().constFirst())) {  }
}

void BreastCancerClassifier::zoomIn()
{
    scaleImage(1.25);
}

void BreastCancerClassifier::zoomOut()
{
    scaleImage(0.8);
}

void BreastCancerClassifier::normalSize()
{
    imageLabel_->adjustSize();
    scaleFactor_ = 1.0;
    ui_->zoomInButton->setEnabled(true);
    ui_->zoomOutButton->setEnabled(true);
}

void BreastCancerClassifier::about()
{
    QMessageBox::about(this, tr("About BreastCancerClassifier"),
                       tr("<p><b>BreastCancerClassifier</b> is a program designed "
                          "to help doctors with classification of possible presence "
                          " of breast cancer changes.</p>"));
}

void BreastCancerClassifier::initializeImageFileDialog(QFileDialog& dialog, QFileDialog::AcceptMode acceptMode)
{
    static bool firstDialog = true;
    if (firstDialog)
    {
        firstDialog = false;
        const QStringList picturesLocations = QStandardPaths::standardLocations(QStandardPaths::PicturesLocation);
        dialog.setDirectory(picturesLocations.isEmpty() ? QDir::currentPath() : picturesLocations.last());
    }

    QStringList mimeTypeFilters;
    const QByteArrayList supportedMimeTypes = acceptMode == QFileDialog::AcceptOpen
        ? QImageReader::supportedMimeTypes() : QImageWriter::supportedMimeTypes();

    for (const QByteArray& mimeTypeName : supportedMimeTypes)
    {
        mimeTypeFilters.append(mimeTypeName);
    }

    mimeTypeFilters.sort();
    dialog.setMimeTypeFilters(mimeTypeFilters);
    dialog.selectMimeTypeFilter("image/png");
    dialog.setAcceptMode(acceptMode);
    if (acceptMode == QFileDialog::AcceptSave)
    {
        dialog.setDefaultSuffix("png");
    }
}

bool BreastCancerClassifier::loadFile(const QString& filename)
{
    QImageReader reader(filename);
    reader.setAutoTransform(true);
    const QImage newImage = reader.read();
    if (newImage.isNull())
    {
        QMessageBox::information(this, QGuiApplication::applicationDisplayName(),
                                 tr("Cannot load %1: %2")
                                 .arg(QDir::toNativeSeparators(filename), reader.errorString()));
        return false;
    }

    setImage(newImage);
    normalSize();
    setWindowFilePath(filename);

    const QString message = tr("Opened \"%1\", %2x%3, Depth: %4")
        .arg(QDir::toNativeSeparators(filename)).arg(image_.width()).arg(image_.height()).arg(image_.depth());
    statusBar()->showMessage(message);

    return true;
}

void BreastCancerClassifier::createActions()
{
    QMenu* fileMenu = menuBar()->addMenu(tr("&File"));

    QAction* openAct = fileMenu->addAction(tr("&Open..."), this, &BreastCancerClassifier::open);
    openAct->setShortcut(QKeySequence::Open);

    fileMenu->addSeparator();

    QAction* exitAct = fileMenu->addAction(tr("E&xit"), this, &QWidget::close);
    exitAct->setShortcut(tr("Ctrl+Q"));

    QMenu* viewMenu = menuBar()->addMenu(tr("&View"));

    zoomInAct_ = viewMenu->addAction(tr("Zoom &In (25%)"), this, &BreastCancerClassifier::zoomIn);
    zoomInAct_->setShortcut(QKeySequence::ZoomIn);
    zoomInAct_->setEnabled(false);

    zoomOutAct_ = viewMenu->addAction(tr("Zoom &Out (25%)"), this, &BreastCancerClassifier::zoomOut);
    zoomOutAct_->setShortcut(QKeySequence::ZoomOut);
    zoomOutAct_->setEnabled(false);

    normalSizeAct_ = viewMenu->addAction(tr("&Normal Size"), this, &BreastCancerClassifier::normalSize);
    normalSizeAct_->setShortcut(tr("Ctrl+S"));
    normalSizeAct_->setEnabled(false);

    viewMenu->addSeparator();

    QMenu* helpMenu = menuBar()->addMenu(tr("&Help"));

    helpMenu->addAction(tr("&About"), this, &BreastCancerClassifier::about);
}

void BreastCancerClassifier::updateActions()
{
    zoomInAct_->setEnabled(true);
    zoomOutAct_->setEnabled(true);
    normalSizeAct_->setEnabled(true);
}

void BreastCancerClassifier::setImage(const QImage& newImage)
{
    image_ = newImage;
    if (image_.colorSpace().isValid())
        image_.convertToColorSpace(QColorSpace::SRgb);

    imageLabel_->setPixmap(QPixmap::fromImage(image_));

    scaleFactor_ = 1.0;
    imageLabel_->setVisible(true);
    updateActions();
}

void BreastCancerClassifier::scaleImage(double factor)
{
    scaleFactor_ *= factor;
    imageLabel_->resize(scaleFactor_ * imageLabel_->pixmap(Qt::ReturnByValue).size());

    adjustScrollBar(scrollArea_->horizontalScrollBar(), factor);
    adjustScrollBar(scrollArea_->verticalScrollBar(), factor);

    zoomInAct_->setEnabled(scaleFactor_ < 3.0);
    zoomOutAct_->setEnabled(scaleFactor_ > 0.333);

    ui_->zoomInButton->setEnabled(scaleFactor_ < 3.0);
    ui_->zoomOutButton->setEnabled(scaleFactor_ > 0.333);
}

void BreastCancerClassifier::adjustScrollBar(QScrollBar* scrollBar, double factor)
{
    scrollBar->setValue(int(factor * scrollBar->value()
                            + ((factor - 1) * scrollBar->pageStep() / 2)));
}






void BreastCancerClassifier::on_selectImageButton_released()
{
    open();
}

void BreastCancerClassifier::on_zoomInButton_released()
{
    zoomIn();
}

void BreastCancerClassifier::on_zoomOutButton_released()
{
    zoomOut();
}

void BreastCancerClassifier::on_resetButton_released()
{
    normalSize();
}

void BreastCancerClassifier::on_testButton_released()
{
    QRect rect(175, 175, 50, 50);
    imageLabel_->size().width();
//    std::cout << scrollArea_->horizontalScrollBar()->sliderPosition() << "\n" << scrollArea_->verticalScrollBar()->sliderPosition() << "\n";
    QImage cropped = imageLabel_->pixmap(Qt::ReturnByValue).toImage().copy(rect);
    cropped.save("cropped_image.png");

    /*
        Think about how to crop the middle 50x50 square of the viewed image - not the middle 50x50 square of the whole image.
        You must make the cropping work with the scrolling of the image, because right now you are always cropping the middle
        square, regardless of the displayed part of an image.

        Make a yellow/orange/red square denoting which part of the displayed image will be cropped. Make it resizable when
        zooming in or out.

        Right now you are cropping only the 50x50 square which is located 175 pixels to the bottom right of the point (0, 0), which
        is located in the upper left corner.
    */
}
