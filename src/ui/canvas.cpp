#include "ui/canvas.h"
#include "BGRA.h"

#include <QApplication>
#include <QKeyEvent>
#include <QPainter>

#include <iostream>
#include <dbg.h>

Canvas::Canvas(QWidget *parent)
    : QWidget(parent), time_(), timer_(), capture_mouse_(false) {
  // Canvas needs all mouse move events, not just mouse drag events
  setMouseTracking(true);

  // Hide the cursor
  if (capture_mouse_) {
    QApplication::setOverrideCursor(Qt::BlankCursor);
  }

  // Canvas needs keyboard focus
  setFocusPolicy(Qt::StrongFocus);

  // The update loop is implemented using a timer
  connect(&timer_, SIGNAL(timeout()), this, SLOT(tick()));

  dbg("TIMER SET");
  
  image_ = QImage(1, 1, QImage::Format_RGB32);

}

Canvas::~Canvas() {}

void Canvas::mousePressEvent(QMouseEvent *event) {}

void Canvas::mouseMoveEvent(QMouseEvent *event) {
  // This starter code implements mouse capture, which gives the change in
  // mouse position since the last mouse movement. The mouse needs to be
  // recentered after every movement because it might otherwise run into
  // the edge of the screen, which would stop the user from moving further
  // in that direction. Note that it is important to check that deltaX and
  // deltaY are not zero before recentering the mouse, otherwise there will
  // be an infinite loop of mouse move events.
  if (capture_mouse_) {
    int deltaX = event->x() - width() / 2;
    int deltaY = event->y() - height() / 2;
    if (!deltaX && !deltaY)
      return;
    QCursor::setPos(mapToGlobal(QPoint(width() / 2, height() / 2)));

    // TODO: Handle mouse movements here
  }
}

void Canvas::mouseReleaseEvent(QMouseEvent *event) {}

void Canvas::keyPressEvent(QKeyEvent *event) {
  if (event->key() == Qt::Key_Escape)
    QApplication::quit();

  // TODO: Handle keyboard presses here
}

void Canvas::keyReleaseEvent(QKeyEvent *event) {}

void Canvas::resize(int width, int height) {
  // clamp negative sizes so we always have at least one pixel
  if (width < 1) {
    width = 1;
  }
  if (height < 1) {
    height = 1;
  }

  image_ = QImage(width, height, QImage::Format_RGB32);

  // set the new image to black
  memset(image_.bits(), 0, static_cast<size_t>(width * height) * sizeof(BGRA));

  // resize and repaint the window (resizing the window doesn't always repaint
  // it, like when you set the same size twice)
  setFixedSize(width, height);
  update();
}

void Canvas::tick() {
  // Get the number of seconds since the last tick (variable update rate)
  float seconds = time_.restart() * 0.001f;

  // TODO: Implement the demo update here
  
  dbg("TICK");
  
  QPainter painter(this);
  painter.drawImage(QPoint(0, 0), image_);

  // Flag this view for repainting (Qt will call paintGL() soon after)
  update();
}
