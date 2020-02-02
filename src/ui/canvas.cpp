#include "ui/canvas.h"
#include "cli/linenoise.hpp"
#include "lib/bgra.h"
#include "ray/render.h"
#include "scene/camera.h"
#include "scene/pool_scene.h"
#include "scene/reflec_balls.h"

#include <QApplication>
#include <QKeyEvent>
#include <QPainter>
#include <boost/lexical_cast.hpp>

#include <iostream>

Canvas::Canvas(QWidget *parent)
    : QWidget(parent), time_(), timer_(), capture_mouse_(false), fps_(0) {
  insideView();
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

  timer_.setInterval(0);
  timer_.start(1);

  resize(width(), height());

  handle_ = std::thread([&] {
    const auto history_path = ".pool_command_history.txt";

    // Enable the multi-line mode
    linenoise::SetMultiLine(true);

    // Set max length of the history
    linenoise::SetHistoryMaxLen(300);

    // Load history
    linenoise::LoadHistory(history_path);

    while (true) {
      // Read line
      std::string line;
      auto quit = linenoise::Readline("> ", line);

      {
        std::lock_guard<std::mutex> guard(event_queue_mutex_);

        if (quit) {
          event_queue_.push_back(InputEvent(Event::Shutdown));
          break;
        }

        std::istringstream buffer(line);

        std::vector<std::string> words{
            std::istream_iterator<std::string>(buffer),
            std::istream_iterator<std::string>()};

        if (words.size() >= 1) {
          if (words[0] == "fps") {
            std::cout << "fps: " << fps_ << std::endl;
          } else if (words[0] == "break") {
            event_queue_.push_back(InputEvent(Event::SetBreak));
          } else if (words[0] == "ball_p") {
            if (words.size() == 4) {
              try {
                InputEvent event(Event::SetBallPosition);

                event.unsigned_value = boost::lexical_cast<unsigned>(words[1]);

                float x = boost::lexical_cast<float>(words[2]);
                float z = boost::lexical_cast<float>(words[3]);

                event.vec2f_value = Eigen::Vector2f(x, z);
                event_queue_.push_back(event);
              } catch (const boost::bad_lexical_cast &e) {
                std::cout << "failed to parse ball_p" << std::endl;
              }
            } else {
              std::cout << "ball_p [number] [x] [z]" << std::endl;
            }
          } else if (words[0] == "ball_v") {
            if (words.size() == 4) {
              try {
                InputEvent event(Event::SetBallVelocity);

                event.unsigned_value = boost::lexical_cast<unsigned>(words[1]);

                float x = boost::lexical_cast<float>(words[2]);
                float z = boost::lexical_cast<float>(words[3]);

                event.vec2f_value = Eigen::Vector2f(x, z);
                event_queue_.push_back(event);
              } catch (const boost::bad_lexical_cast &e) {
                std::cout << "failed to parse ball_v" << std::endl;
              }
            } else {
              std::cout << "ball_v [number] [x] [z]" << std::endl;
            }
          } else if (words[0] == "ball_r") {
            if (words.size() == 6) {
              try {
                InputEvent event(Event::SetBallRotation);

                event.unsigned_value = boost::lexical_cast<unsigned>(words[1]);

                float x = boost::lexical_cast<float>(words[2]);
                float y = boost::lexical_cast<float>(words[3]);
                float z = boost::lexical_cast<float>(words[4]);
                float w = boost::lexical_cast<float>(words[5]);

                event.quat_value = Eigen::Quaternionf(x, y, z, w);
                event_queue_.push_back(event);
              } catch (const boost::bad_lexical_cast &e) {
                std::cout << "failed to parse ball_r" << std::endl;
              }
            } else {
              std::cout << "ball_r [number] [x] [y] [z]" << std::endl;
            }
          } else if (words[0] == "ball_r_v") {
            if (words.size() == 5) {
              try {
                InputEvent event(Event::SetBallAngularVelocity);

                event.unsigned_value = boost::lexical_cast<unsigned>(words[1]);

                float x = boost::lexical_cast<float>(words[2]);
                float y = boost::lexical_cast<float>(words[3]);
                float z = boost::lexical_cast<float>(words[4]);

                event.vec3f_value = Eigen::Vector3f(x, y, z);
                event_queue_.push_back(event);
              } catch (const boost::bad_lexical_cast &e) {
                std::cout << "failed to parse ball_r_v" << std::endl;
              }
            } else {
              std::cout << "ball_r_v [number] [x] [y] [z]" << std::endl;
            }
          } else if (words[0] == "ball_reset") {
            event_queue_.push_back(InputEvent(Event::BallReset));
          } else if (words[0] == "top_view") {
            event_queue_.push_back(InputEvent(Event::TopView));
          } else if (words[0] == "side_view") {
            event_queue_.push_back(InputEvent(Event::SideView));
          } else if (words[0] == "inside_view") {
            event_queue_.push_back(InputEvent(Event::InsideView));
          } else if (words[0] == "look") {
            if (words.size() == 4) {
              try {
                InputEvent event(Event::SetLook);

                float x = boost::lexical_cast<float>(words[1]);
                float y = boost::lexical_cast<float>(words[2]);
                float z = boost::lexical_cast<float>(words[3]);

                event.vec3f_value = Eigen::Vector3f(x, y, z);
                event_queue_.push_back(event);
              } catch (const boost::bad_lexical_cast &e) {
                std::cout << "failed to parse look" << std::endl;
              }
            } else {
              std::cout << "look [x] [y] [z]" << std::endl;
            }
          } else if (words[0] == "up") {
            if (words.size() == 4) {
              try {
                InputEvent event(Event::SetUp);

                float x = boost::lexical_cast<float>(words[1]);
                float y = boost::lexical_cast<float>(words[2]);
                float z = boost::lexical_cast<float>(words[3]);

                event.vec3f_value = Eigen::Vector3f(x, y, z);
                event_queue_.push_back(event);
              } catch (const boost::bad_lexical_cast &e) {
                std::cout << "failed to parse up" << std::endl;
              }
            } else {
              std::cout << "up [x] [y] [z]" << std::endl;
            }
          } else if (words[0] == "pos") {
            if (words.size() == 4) {
              try {
                InputEvent event(Event::SetPos);

                float x = boost::lexical_cast<float>(words[1]);
                float y = boost::lexical_cast<float>(words[2]);
                float z = boost::lexical_cast<float>(words[3]);

                event.vec3f_value = Eigen::Vector3f(x, y, z);
                event_queue_.push_back(event);
              } catch (const boost::bad_lexical_cast &e) {
                std::cout << "failed to parse pos" << std::endl;
              }
            } else {
              std::cout << "pos [x] [y] [z]" << std::endl;
            }
          } else if (words[0] == "set") {
            if (words.size() == 3) {
              if (words[1] == "sampling") {
                try {
                  unsigned value = boost::lexical_cast<unsigned>(words[2]);
                  InputEvent event(Event::SetSampling);
                  event.unsigned_value = value;
                  event_queue_.push_back(event);
                } catch (const boost::bad_lexical_cast &e) {
                  std::cout << "failed to parse sampling" << std::endl;
                }
              } else if (words[1] == "iterations") {
                try {
                  unsigned value = boost::lexical_cast<unsigned>(words[2]);
                  InputEvent event(Event::SetIterations);
                  event.unsigned_value = value;
                  event_queue_.push_back(event);
                } catch (const boost::bad_lexical_cast &e) {
                  std::cout << "failed to parse iterations" << std::endl;
                }
              } else {
                std::cout << "set name not recognized" << std::endl;
              }
            } else {
              std::cout << "set [name] [value]" << std::endl;
            }
          } else {
            std::cout << "invalid command" << std::endl;
          }
        }
      }

      // Add text to history
      linenoise::AddHistory(line.c_str());
    }

    // Save history
    linenoise::SaveHistory(history_path);
  });
}

void Canvas::resizeEvent(QResizeEvent *) { resize(width(), height()); }

void Canvas::mousePressEvent(QMouseEvent *event) {
  resetRenderer();
  time_.restart();
  update();
}

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

void Canvas::resize(int width_in, int height_in) {
  width_ = width_in;
  height_ = height_in;
  // clamp negative sizes so we always have at least one pixel
  if (width_ < 1) {
    width_ = 1;
  }
  if (height_ < 1) {
    height_ = 1;
  }

  image_ = QImage(width_, height_, QImage::Format_RGB32);

  // set the new image to black
  memset(image_.bits(), 0,
         static_cast<size_t>(width_ * height_) * sizeof(BGRA));

  resetRenderer();
  resetTransform();

  update();
}

void Canvas::resetRenderer() {
  std::unique_ptr<scene::Scene> s = std::make_unique<scene::ReflecBalls>();
  renderer_ = std::make_unique<ray::Renderer<ExecutionModel::GPU>>(
      width_, height_, super_sampling_rate_, recursive_iterations_, s);
}

void Canvas::sideView() {
  look_ = Eigen::Vector3f(-2, -1, 0);
  up_ = Eigen::Vector3f(0, 1, 0);
  pos_ = Eigen::Vector3f(50, 30, 0);
}

void Canvas::insideView() {
  look_ = Eigen::Vector3f(-2, -1, 0);
  up_ = Eigen::Vector3f(0, 1, 0);
  pos_ = Eigen::Vector3f(20, 15, 0);
}

void Canvas::topView() {
  look_ = Eigen::Vector3f(0, -1, 0);
  up_ = Eigen::Vector3f(1, 0, 0);
  pos_ = Eigen::Vector3f(0, 50, 0);
}

void Canvas::resetTransform() {
  auto [film_to_world, world_to_film] = scene::get_camera_transform(
      look_, up_, pos_, 1.0f, width_, height_, 30.0f);
  film_to_world_ = film_to_world;
  world_to_film_ = world_to_film;
}

void Canvas::tick() {
  {
    std::lock_guard<std::mutex> guard(event_queue_mutex_);
    auto &pool_scene = renderer_->get_scene();
    for (const auto &event : event_queue_) {
      auto set_ball_prop = [&](const auto &f) {
#if 0
        if (pool_scene.getNumBalls() > event.unsigned_value) {
          f(pool_scene.getBallState(event.unsigned_value));
        } else {
          std::cout << "invalid ball index" << std::endl;
        }
#else
        std::cout << "not currently supported" << std::endl;
#endif
      };

      switch (event.event) {
      case Event::Shutdown:
        handle_.join();
        QApplication::quit();
        break;
      case Event::SetIterations:
        recursive_iterations_ = event.unsigned_value;
        resetRenderer();
        break;
      case Event::SetSampling:
        super_sampling_rate_ = event.unsigned_value;
        resetRenderer();
        break;
      case Event::SetBallPosition:
        set_ball_prop([&](scene::PoolScene::BallState &ball) {
          ball.pos = event.vec2f_value;
        });
        break;
      case Event::SetBallVelocity:
        set_ball_prop([&](scene::PoolScene::BallState &ball) {
          ball.vel = event.vec2f_value;
        });
        break;
      case Event::SetBallRotation:
        set_ball_prop([&](scene::PoolScene::BallState &ball) {
          ball.rot = event.quat_value;
        });
        break;
      case Event::SetBallAngularVelocity:
        set_ball_prop([&](scene::PoolScene::BallState &ball) {
          ball.angular_vel = event.vec3f_value;
        });
        break;
      case Event::SetLook:
        look_ = event.vec3f_value;
        resetTransform();
        break;
      case Event::SetUp:
        up_ = event.vec3f_value;
        resetTransform();
        break;
      case Event::SetPos:
        pos_ = event.vec3f_value;
        resetTransform();
        break;
      case Event::SetBreak:
#if 0
        pool_scene_.setBreak();
#else
        std::cout << "not currently supported" << std::endl;
#endif
        break;
      case Event::BallReset:
        set_ball_prop([&](scene::PoolScene::BallState &ball) {
          ball.angular_vel = Eigen::Vector3f::Zero();
          ball.vel = Eigen::Vector2f::Zero();
          ball.rot = Eigen::Quaternionf::Identity();
        });
        break;
      case Event::TopView:
        topView();
        resetTransform();
        break;
      case Event::SideView:
        sideView();
        resetTransform();
      case Event::InsideView:
        insideView();
        resetTransform();
        break;
      }
    }
    event_queue_.clear();
  }

  update();
}

void Canvas::paintEvent(QPaintEvent * /* event */) {
  // Get the number of seconds since the last tick (variable update rate)
  float secs;
  {
    std::lock_guard<std::mutex> guard(event_queue_mutex_);

    secs = time_.restart() * 0.001f;

    if (secs > 1e-5) {
      fps_ = fps_ * (1 - fps_alpha) + fps_alpha / secs;
    }
  }

  unsigned steps = std::max(secs / min_physics_step_size, 1.0f);
  float time_per_step = secs / steps;
  for (unsigned i = 0; i < steps; i++) {
    renderer_->get_scene().step(time_per_step);
  }

  if (renderer_) {
    renderer_->render(reinterpret_cast<BGRA *>(image_.bits()), film_to_world_,
                      world_to_film_, true, true, false);
  }

  QPainter painter(this);
  painter.drawImage(QPoint(0, 0), image_);
}
