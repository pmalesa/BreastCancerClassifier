#include "breast_cancer_classifier.h"
#include "./ui_breast_cancer_classifier.h"

#include <QScreen>
#include <QStandardPaths>
#include <QImageReader>
#include <QImageWriter>
#include <QMessageBox>
#include <QScrollBar>
#include <QColorSpace>

BreastCancerClassifier::BreastCancerClassifier(QWidget *parent)
    : QMainWindow(parent), //ui_(new Ui::BreastCancerClassifier),
      imageLabel_(new QLabel), scrollArea_(new QScrollArea)
{
    //ui_->setupUi(this);

    imageLabel_->setBackgroundRole(QPalette::Base);
    imageLabel_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    imageLabel_->setScaledContents(true);

    scrollArea_->setBackgroundRole(QPalette::Dark);
    scrollArea_->setWidget(imageLabel_);
    scrollArea_->setVisible(false);
    setCentralWidget(scrollArea_);

    createActions();

    resize(QGuiApplication::primaryScreen()->availableSize() * 3 / 5);
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
    scrollArea_->setVisible(true);
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
}

void BreastCancerClassifier::adjustScrollBar(QScrollBar* scrollBar, double factor)
{
    scrollBar->setValue(int(factor * scrollBar->value()
                            + ((factor - 1) * scrollBar->pageStep() / 2)));
}


























