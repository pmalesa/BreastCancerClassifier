#include "breast_cancer_classifier.h"
#include "./ui_breast_cancer_classifier.h"

#include <string>
#include <sstream>
#include <iostream>
#include <cmath>

#include <QScreen>
#include <QStandardPaths>
#include <QImageWriter>
#include <QMessageBox>
#include <QScrollBar>
#include <QColorSpace>
#include <QPainter>


BreastCancerClassifier::BreastCancerClassifier(QWidget *parent)
    : QMainWindow(parent), ui_(new Ui::BreastCancerClassifier),
      selectionFramePath_("../images/selection_frame.png")
{
    /* Initialization of Python embedding */
    pName = PyUnicode_FromString("classify_image");
    pModule = PyImport_Import(pName);
    if (pModule)
    {
        pFunc = PyObject_GetAttrString(pModule, "classify");
    }
    else
    {
        QMessageBox::warning(this, QString("BreastCancerClassifier"),
                                 tr("ERROR: Module could not be imported."));
    }

    ui_->setupUi(this);

    imageLabel_ = ui_->imageLabel;
    scrollArea_ = ui_->scrollArea;

    reader_.setFileName(selectionFramePath_);
    reader_.setAutoTransform(true);
    selectionFrame_ = reader_.read();
    if (selectionFrame_.colorSpace().isValid())
        selectionFrame_.convertToColorSpace(QColorSpace::SRgb);
    selectionFrameX_ = 0;
    selectionFrameY_ = 0;

    imageLabel_->setBackgroundRole(QPalette::Base);
    imageLabel_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    imageLabel_->setScaledContents(true);

    scrollArea_->setBackgroundRole(QPalette::Dark);
    scrollArea_->setWidget(imageLabel_);
    imageLabel_->setVisible(false);
    scrollArea_->setVisible(true);
    createActions();

    ui_->imageNavigationGroupBox->setEnabled(false);
    ui_->classifyButton->setEnabled(false);
    ui_->saveSelectionButton->setEnabled(false);
    ui_->resetButton->setEnabled(false);
    ui_->zoomInButton->setEnabled(false);
    ui_->zoomOutButton->setEnabled(false);
    ui_->unselectImageButton->setEnabled(false);
}

BreastCancerClassifier::~BreastCancerClassifier()
{
    delete ui_;
    Py_Finalize();
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
    reader_.setFileName(filename);
    reader_.setAutoTransform(true);
    const QImage newImage = reader_.read();
    if (newImage.isNull())
    {
        QMessageBox::warning(this, QString("BreastCancerClassifier"),
                                 tr("Cannot load %1: %2")
                                 .arg(QDir::toNativeSeparators(filename), reader_.errorString()));
        return false;
    }

    if (newImage.size().width() > imageMaxSize_ || newImage.size().height() > imageMaxSize_)
    {
        QMessageBox::warning(this, QString("BreastCancerClassifier"),
                                 tr("Loaded image is too wide or tall. Maximum width and height allowed is 1000 pixels.")
                                 .arg(QDir::toNativeSeparators(filename), reader_.errorString()));
        return false;
    }

    if (newImage.size().width() < imageMinSize_ || newImage.size().height() < imageMinSize_)
    {
        QMessageBox::warning(this, QString("BreastCancerClassifier"),
                                 tr("Loaded image is too small. Minimal width and height allowed is 50 pixels.")
                                 .arg(QDir::toNativeSeparators(filename), reader_.errorString()));
        return false;
    }

    setImage(newImage);
    normalSize();
    setWindowFilePath(filename);
    ui_->selectedImageLineEdit->setText(filename);

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

    pixmap_ = QPixmap::fromImage(image_);

    /* Determining initial position of the selection frame */
    if (image_.size() != QSize(imageMinSize_, imageMinSize_))
    {
        backgroundPixmap_ = QPixmap(image_.width() + 2 * selectionFrameWidth_, image_.height() + 2 * selectionFrameWidth_);

        if (image_.width() - selectionAreaSize_ <= selectionFrameWidth_ && image_.height() - selectionAreaSize_ <= selectionFrameWidth_)
        {
            selectionFrameX_ = 0;
            selectionFrameY_ = 0;
        }
        else if (image_.width() - selectionAreaSize_ < selectionFrameWidth_ && image_.height() - selectionAreaSize_ >= selectionFrameWidth_)
        {
            selectionFrameX_ = 0;
            selectionFrameY_ = backgroundPixmap_.height() / 2 - selectionAreaSize_ / 2;
        }
        else if (image_.width() - selectionAreaSize_ >= selectionFrameWidth_ && image_.height() - selectionAreaSize_ < selectionFrameWidth_)
        {
            selectionFrameX_ = backgroundPixmap_.width() / 2 - selectionAreaSize_ / 2;
            selectionFrameY_ = 0;
        }
        else
        {
            selectionFrameX_ = backgroundPixmap_.width() / 2 - selectionAreaSize_ / 2;
            selectionFrameY_ = backgroundPixmap_.height() / 2 - selectionAreaSize_ / 2;
        }
        refreshSelectionFrame();
        ui_->imageNavigationGroupBox->setEnabled(true);
    }
    else
    {
        selectionFrameX_ = -selectionFrameWidth_;
        selectionFrameY_ = -selectionFrameWidth_;
        imageLabel_->setPixmap(pixmap_);
    }

    scaleFactor_ = 1.0;
    imageLabel_->setVisible(true);
    updateActions();
    ui_->classifyButton->setEnabled(true);
    ui_->saveSelectionButton->setEnabled(true);
    ui_->resetButton->setEnabled(true);
    ui_->zoomInButton->setEnabled(true);
    ui_->zoomOutButton->setEnabled(true);
    ui_->unselectImageButton->setEnabled(true);
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

void BreastCancerClassifier::refreshSelectionFrame()
{
    painter_.begin(&backgroundPixmap_);
    painter_.fillRect(backgroundPixmap_.rect(), Qt::gray);
    painter_.drawImage(selectionFrameWidth_, selectionFrameWidth_, image_);
    painter_.drawImage(selectionFrameX_, selectionFrameY_, selectionFrame_);
    painter_.end();
    imageLabel_->setPixmap(backgroundPixmap_);
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
    if (image_.size() != QSize(imageMinSize_, imageMinSize_))
    {
        selectionFrameX_ = backgroundPixmap_.width() / 2 - selectionAreaSize_ / 2;
        selectionFrameY_ = backgroundPixmap_.height() / 2 - selectionAreaSize_ / 2;
        refreshSelectionFrame();
        ui_->imageNavigationGroupBox->setEnabled(true);
    }
}

void BreastCancerClassifier::on_moveUpButton_released()
{
    selectionFrameY_ -= 1;
    if (selectionFrameY_ < 0)
        selectionFrameY_ = 0;
    refreshSelectionFrame();
}

void BreastCancerClassifier::on_moveRightButton_released()
{
    selectionFrameX_ += 1;
    if (selectionFrameX_ + selectionAreaSize_ > image_.size().width() )
        selectionFrameX_ = image_.size().width() - selectionAreaSize_;
    refreshSelectionFrame();
}

void BreastCancerClassifier::on_moveDownButton_released()
{
    selectionFrameY_ += 1;
    if (selectionFrameY_ + selectionAreaSize_ > image_.size().height() )
        selectionFrameY_ = image_.size().height() - selectionAreaSize_;
    refreshSelectionFrame();
}

void BreastCancerClassifier::on_moveLeftButton_released()
{
    selectionFrameX_ -= 1;
    if (selectionFrameX_ < 0)
        selectionFrameX_ = 0;
    refreshSelectionFrame();
}

void BreastCancerClassifier::on_saveSelectionButton_released()
{
    QRect rect(selectionFrameX_ + 5, selectionFrameY_ + 5, 50, 50);
    imageLabel_->size().width();
    QImage cropped = imageLabel_->pixmap(Qt::ReturnByValue).toImage().copy(rect);
    cropped.save("selected_image.png");
}

void BreastCancerClassifier::on_moveRightButton_2_released()
{
    selectionFrameX_ += 20;
    if (selectionFrameX_ + selectionAreaSize_ > image_.size().width() )
        selectionFrameX_ = image_.size().width() - selectionAreaSize_;
    refreshSelectionFrame();
}

void BreastCancerClassifier::on_moveDownButton_2_released()
{
    selectionFrameY_ += 20;
    if (selectionFrameY_ + selectionAreaSize_ > image_.size().height() )
        selectionFrameY_ = image_.size().height() - selectionAreaSize_;
    refreshSelectionFrame();
}

void BreastCancerClassifier::on_moveLeftButton_2_released()
{
    selectionFrameX_ -= 20;
    if (selectionFrameX_ < 0)
        selectionFrameX_ = 0;
    refreshSelectionFrame();
}

void BreastCancerClassifier::on_moveUpButton_2_released()
{
    selectionFrameY_ -= 20;
    if (selectionFrameY_ < 0)
        selectionFrameY_ = 0;
    refreshSelectionFrame();
}

void BreastCancerClassifier::on_unselectImageButton_released()
{
    ui_->selectedImageLineEdit->setText("");
    image_ = QImage();
    pixmap_ = QPixmap();
    imageLabel_->setVisible(false);
    ui_->imageNavigationGroupBox->setEnabled(false);
    ui_->classifyButton->setEnabled(false);
    ui_->saveSelectionButton->setEnabled(false);
    ui_->resetButton->setEnabled(false);
    ui_->zoomInButton->setEnabled(false);
    ui_->zoomOutButton->setEnabled(false);
    ui_->unselectImageButton->setEnabled(false);
}



/*
    REMARKS:
    - Maybe find a way to continuously move the image, not in 20 pixel steps.
*/



void BreastCancerClassifier::on_classifyButton_released()
{
    on_saveSelectionButton_released();
    if (pFunc && PyCallable_Check(pFunc))
    {
        pValue = PyObject_CallObject(pFunc, NULL);
        double predictionValue = PyFloat_AsDouble(pValue);
        predictionValue = std::ceil(predictionValue * 10000.0) / 100.0;
        if (predictionValue <= 33.3)
            ui_->resultsTextBrowser->setTextColor(Qt::green);
        else if (predictionValue > 33.3 && predictionValue <= 66.6)
            ui_->resultsTextBrowser->setTextColor(Qt::yellow);
        else
            ui_->resultsTextBrowser->setTextColor(Qt::red);
        std::ostringstream strs;
        strs << predictionValue;
        std::string valueStr = strs.str();
        valueStr += "%";
        ui_->resultsTextBrowser->setText(QString(valueStr.c_str()));
    }
    else
    {
        std::cout << "Error occurred while calling Python's classify() function.\n";
    }
}
