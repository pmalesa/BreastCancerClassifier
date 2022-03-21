#ifndef BREASTCANCERCLASSIFIER_H
#define BREASTCANCERCLASSIFIER_H

#include <QMainWindow>
#include <QImage>
#include <QLabel>
#include <QScrollArea>
#include <QFileDialog>

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

    void on_testButton_released();

private:
    void initializeImageFileDialog(QFileDialog& dialog, QFileDialog::AcceptMode acceptMode);
    bool loadFile(const QString& filename);
    void createActions();
    void updateActions();
    void setImage(const QImage& newImage);
    void scaleImage(double factor);
    void adjustScrollBar(QScrollBar* scrollBar, double factor);

    Ui::BreastCancerClassifier* ui_;
    QImage image_;
    QLabel* imageLabel_;
    QScrollArea* scrollArea_;
    double scaleFactor_ = 1;

    QAction* zoomInAct_;
    QAction* zoomOutAct_;
    QAction* normalSizeAct_;
    QAction* fitToWindowAct_;
};
#endif // BREASTCANCERCLASSIFIER_H
