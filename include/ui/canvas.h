#pragma once

#include <QWidget>
#include <QTime>
#include <QTimer>

class Canvas : public QWidget {
    Q_OBJECT

public:
    Canvas(QWidget *parent);
    ~Canvas() override;

private:
    QTime time_;
    QTimer timer_;
    bool capture_mouse_;
    QImage image_;

    /* void initializeGL(); */
    /* void paintGL(); */
    /* void resizeGL(int w, int h); */

    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;


    void resize(int width, int height);

  private slots:
    void tick();
};
