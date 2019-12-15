#pragma once

#include "ray/execution_model.h"
#include "scene/pool_scene.h"

#include <QTime>
#include <QTimer>
#include <QWidget>

#include <mutex>
#include <thread>
#include <vector>

namespace ray {
template <ray::ExecutionModel execution_model> class Renderer;
}

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
  float fps_;
  unsigned super_sampling_rate_ = 1;
  unsigned recursive_iterations_ = 3;
  int width_;
  int height_;
  Eigen::Vector3f look_;
  Eigen::Vector3f up_;
  Eigen::Vector3f pos_;


  enum class Event {
    Shutdown,
    SetSampling,
    SetIterations,
    SetBallPosition,
    SetBallVelocity,
    SetBallRotation,
    SetBallAngularVelocity,
    SetLook,
    SetUp,
    SetPos,
    SetBreak,
    BallReset,
    TopView,
    SideView,
  };

  struct InputEvent {
    Event event;
    unsigned unsigned_value;
    Eigen::Quaternionf quat_value;
    Eigen::Vector3f vec3f_value;
    Eigen::Vector2f vec2f_value;
    InputEvent(Event event) : event(event) {}
  };

  std::vector<InputEvent> event_queue_;
  std::mutex event_queue_mutex_;
  std::thread handle_;

  Eigen::Affine3f film_to_world_;
  Eigen::Projective3f world_to_film_;
  ray::Renderer<ray::ExecutionModel::GPU> *renderer_;

  static constexpr float min_physics_step_size = 0.0001;
  static constexpr float fps_alpha = 0.05;

  /* void initializeGL(); */
  /* void paintGL(); */
  /* void resizeGL(int w, int h); */

  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;

  void keyPressEvent(QKeyEvent *event) override;
  void keyReleaseEvent(QKeyEvent *event) override;

  void paintEvent(QPaintEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;

  void resize(int width, int height);

  void sideView();
  void topView();

  void resetTransform();
  void resetRenderer();

private slots:
  void tick();
};
