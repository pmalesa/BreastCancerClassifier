#include "breast_cancer_classifier.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    a.setWindowIcon(QIcon("./images/medical.ico"));
    BreastCancerClassifier w;
    w.show();
    return a.exec();
}
