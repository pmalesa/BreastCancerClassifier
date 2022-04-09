#ifndef BREASTCANCERCLASSIFIER_H
#define BREASTCANCERCLASSIFIER_H

#include <QMainWindow>
#include <QImage>
#include <QPainter>
#include <QLabel>
#include <QScrollArea>
#include <QFileDialog>
#include <QImageReader>

#include "pyhelper.h"

QT_BEGIN_NAMESPACE
namespace Ui { class BreastCancerClassifier; }
QT_END_NAMESPACE

class BreastCancerClassifier : public QMainWindow
{
    Q_OBJECT

public:
    BreastCancerClassifier(QWidget* parent = nullptr);
    ~BreastCancerClassifier();

private slots:
    void open();
    void zoomIn();
    void zoomOut();
    void normalSize();
    void about();

    void on_selectImageButton_released();

    void on_zoomInButton_released();

    void on_zoomOutButton_released();

    void on_resetButton_released();

    void on_moveUpButton_released();

    void on_moveRightButton_released();

    void on_moveDownButton_released();

    void on_moveLeftButton_released();

    void on_saveSelectionButton_released();

    void on_moveRightButton_2_released();

    void on_moveDownButton_2_released();

    void on_moveLeftButton_2_released();

    void on_moveUpButton_2_released();

    void on_unselectImageButton_released();

    void on_classifyButton_released();

private:
    void initializeImageFileDialog(QFileDialog& dialog, QFileDialog::AcceptMode acceptMode);
    bool loadFile(const QString& filename);
    void createActions();
    void updateActions();
    void setImage(const QImage& newImage);
    void scaleImage(double factor);
    void adjustScrollBar(QScrollBar* scrollBar, double factor);
    void refreshSelectionFrame();

    CPyObject pArgs;
    CPyObject pValue;
    CPyObject pFunc;
    CPyObject pModule;
    CPyObject pName;
    CPyInstance pInstance;

    Ui::BreastCancerClassifier* ui_;
    QImageReader reader_;
    QImage image_;
    const int imageMaxSize_ = 1000;
    const int imageMinSize_ = 50;
    QImage selectionFrame_;
    QString selectionFramePath_;
    const int selectionAreaSize_ = 50;
    const int selectionFrameWidth_ = 5;
    QPainter painter_;
    QPixmap pixmap_;
    QPixmap backgroundPixmap_;
    QLabel* imageLabel_;
    int selectionFrameX_;
    int selectionFrameY_;
    QScrollArea* scrollArea_;
    double scaleFactor_ = 1;

    QAction* zoomInAct_;
    QAction* zoomOutAct_;
    QAction* normalSizeAct_;
    QAction* fitToWindowAct_;
};
#endif // BREASTCANCERCLASSIFIER_H
