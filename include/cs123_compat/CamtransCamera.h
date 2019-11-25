#pragma once

#include "Camera.h"

/**
 * @class CamtransCamera
 *.2
 * The perspective camera to be implemented in the Camtrans lab.
 */
class CamtransCamera : public Camera {
public:
  // Initialize your camera.
  CamtransCamera();

  // Sets the aspect ratio of this camera. Automatically called by the GUI when
  // the window is resized.
  virtual void setAspectRatio(float aspectRatio);

  // Returns the projection matrix given the current camera settings.
  virtual glm::mat4x4 getProjectionMatrix() const;

  // Returns the view matrix given the current camera settings.
  virtual glm::mat4x4 getViewMatrix() const;

  // Returns the matrix that scales down the perspective view volume into the
  // canonical perspective view volume, given the current camera settings.
  virtual glm::mat4x4 getScaleMatrix() const;

  // Returns the matrix the unhinges the perspective view volume, given the
  // current camera settings.
  virtual glm::mat4x4 getPerspectiveMatrix() const;

  // Returns the current position of the camera.
  glm::vec4 getPosition() const;

  // Returns the current 'look' vector for this camera.
  glm::vec4 getLook() const;

  // Returns the current 'up' vector for this camera (the 'V' vector).
  glm::vec4 getUp() const;

  glm::vec4 getU() const;
  glm::vec4 getV() const;
  glm::vec4 getW() const;

  // Returns the currently set aspect ratio.
  float getAspectRatio() const;

  // Returns the currently set height angle.
  float getHeightAngle() const;

  // Move this camera to a new eye position, and orient the camera's axes given
  // look and up vectors.
  void orientLook(const glm::vec4 &eye, const glm::vec4 &look,
                  const glm::vec4 &up);

  // Sets the height angle of this camera.
  void setHeightAngle(float h);

  // Translates the camera along a given vector.
  void translate(const glm::vec4 &v);

  // Rotates the camera about the U axis by a specified number of degrees.
  void rotateU(float degrees);

  // Rotates the camera about the V axis by a specified number of degrees.
  void rotateV(float degrees);

  // Rotates the camera about the W axis by a specified number of degrees.
  void rotateW(float degrees);

  // Sets the near and far clip planes for this camera.
  void setClip(float nearPlane, float farPlane);

  void updateTranslationMatrix();
  void updateRotationMatrix();
  void updateScaleMatrix();
  void updatePerspectiveMatrix();
  void updateProjectionMatrix();
  void updateViewMatrix();

private:
  float aspect_ratio_;
  float near_, far_;
  glm::mat4 translation_matrix_, rotation_matrix_, scale_matrix_;
  glm::mat4 perspective_transformation_;
  float theta_h_, theta_w_;
  glm::vec4 eye_, look_, up_;
  glm::vec4 u_, v_, w_;
};
