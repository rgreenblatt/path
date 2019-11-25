/**
 * @file   CamtransCamera.cpp
 *
 * This is the perspective camera class you will need to fill in for the
 * Camtrans lab.  See the lab handout for more details.
 */

#include "CamtransCamera.h"
#include <Settings.h>

CamtransCamera::CamtransCamera()
    : Camera(), near_(1.0f), far_(30.0f), translation_matrix_(1),
      rotation_matrix_(1), scale_matrix_(1), perspective_transformation_(1),
      theta_h_(glm::radians(60.0f)), theta_w_(theta_h_), eye_(2, 2, 2, 1),
      look_(-2, -2, -2, 0), up_(0, 1, 0, 0) {
  // explict to avoid potential issues with virtual overrides if any are virtual
  CamtransCamera::orientLook(eye_, look_, up_);

  CamtransCamera::updateTranslationMatrix();
  CamtransCamera::updateRotationMatrix();
  CamtransCamera::updateScaleMatrix();
  CamtransCamera::updatePerspectiveMatrix();
  perspective_transformation_[3][3] = 0.0f;
  perspective_transformation_[2][3] = -1.0f;
}

void CamtransCamera::setAspectRatio(float a) {
  theta_w_ = std::atan(a * std::tan(theta_h_ / 2)) * 2;
  updateProjectionMatrix();
}

glm::mat4x4 CamtransCamera::getProjectionMatrix() const {
  return perspective_transformation_ * scale_matrix_;
}

glm::mat4x4 CamtransCamera::getViewMatrix() const {
  return rotation_matrix_ * translation_matrix_;
}

glm::mat4x4 CamtransCamera::getScaleMatrix() const {
  return scale_matrix_;
}

glm::mat4x4 CamtransCamera::getPerspectiveMatrix() const {
  return perspective_transformation_;
}

glm::vec4 CamtransCamera::getPosition() const {
  return eye_;
}

glm::vec4 CamtransCamera::getLook() const {
  return -w_;
}

glm::vec4 CamtransCamera::getUp() const {
  return up_;
}

glm::vec4 CamtransCamera::getU() const {
  return u_;
}

glm::vec4 CamtransCamera::getV() const {
  return v_;
}

glm::vec4 CamtransCamera::getW() const {
  return w_;
}

float CamtransCamera::getAspectRatio() const {
  return std::tan(theta_w_ / 2) / std::tan(theta_h_ / 2);
}

float CamtransCamera::getHeightAngle() const {
  return theta_h_;
}

void CamtransCamera::updateTranslationMatrix() {
  auto &col = translation_matrix_[3];
  col = -eye_;
  col[3] = 1;
}

void CamtransCamera::updateRotationMatrix() {
  auto set_row = [&](const auto& vec, const int i) {
    rotation_matrix_[0][i] = vec[0];
    rotation_matrix_[1][i] = vec[1];
    rotation_matrix_[2][i] = vec[2];
  };

  set_row(u_, 0);
  set_row(v_, 1);
  set_row(w_, 2);
}

void CamtransCamera::updateScaleMatrix() {
  scale_matrix_[0][0] = 1.0f / (std::tan(theta_w_ / 2) * far_);
  scale_matrix_[1][1] = 1.0f / (std::tan(theta_h_ / 2) * far_);
  scale_matrix_[2][2] = 1.0f / far_;
}

void CamtransCamera::updatePerspectiveMatrix() {
  float c = near_ / far_;

  perspective_transformation_[2][2] = -1 / (1 + c);
  perspective_transformation_[3][2] = -c / (1 + c);
}

void CamtransCamera::updateProjectionMatrix() {
  updatePerspectiveMatrix();
  updateScaleMatrix();
}

void CamtransCamera::updateViewMatrix() {
  updateTranslationMatrix();
  updateRotationMatrix();
}

void CamtransCamera::orientLook(const glm::vec4 &eye, const glm::vec4 &look,
                                const glm::vec4 &up) {
  eye_ = eye;
  look_ = look;
  up_ = up;
  
  w_ = glm::normalize(-look_);
  v_ = glm::normalize(up_ - glm::dot(up_, w_) * w_);
  u_ = glm::vec4(glm::cross(glm::vec3(v_), glm::vec3(w_)), 0);

  updateViewMatrix();
  updateProjectionMatrix();
}

void CamtransCamera::setHeightAngle(float h) {
  theta_h_ = glm::radians(h);

  updateProjectionMatrix();
}

void CamtransCamera::translate(const glm::vec4 &v) {
  eye_ += v;

  updateViewMatrix();
}

// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
glm::vec4 rotate_vec_by_angle_around_axis(const glm::vec4 &v,
                                          const glm::vec4 &k,
                                          const float theta) {
  return v * std::cos(theta) +
         glm::vec4(glm::cross(glm::vec3(k), glm::vec3(v)) * std::sin(theta),
                   0) +
         k * glm::dot(k, v) * (1 - std::cos(theta));
}

void CamtransCamera::rotateU(float degrees) {
  const float theta = glm::radians(degrees);

  w_ = rotate_vec_by_angle_around_axis(w_, u_, theta);
  v_ = rotate_vec_by_angle_around_axis(v_, u_, theta);

  updateViewMatrix();
}

void CamtransCamera::rotateV(float degrees) {
  const float theta = glm::radians(degrees);
  
  w_ = rotate_vec_by_angle_around_axis(w_, v_, theta);
  u_ = rotate_vec_by_angle_around_axis(u_, v_, theta);

  updateViewMatrix();
}

void CamtransCamera::rotateW(float degrees) {
  const float theta = glm::radians(degrees);

  v_ = rotate_vec_by_angle_around_axis(v_, w_, theta);
  u_ = rotate_vec_by_angle_around_axis(u_, w_, theta);

  updateViewMatrix();
}

void CamtransCamera::setClip(float nearPlane, float farPlane) {
  near_ = nearPlane;
  far_ = farPlane;

  updateProjectionMatrix();
}
