#include "core/Camera.h"
#include <SDL3/SDL.h>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

Camera::Camera() {
  updateViewMatrix();
  updateProjectionMatrix();
}

void Camera::update(float deltaTime) {
  // Apply accumulated velocity to position.
  // Velocity is in camera space, so rotate each axis by orientation first.
  if (glm::length(velocity) > 0.01f) {
    glm::vec3 forward = glm::rotate(orientation, glm::vec3(0, 0, -1));
    glm::vec3 right = glm::rotate(orientation, glm::vec3(1, 0, 0));
    glm::vec3 up = glm::vec3(0, 1, 0);

    glm::vec3 moveDir = glm::vec3(0);
    moveDir += forward * velocity.z;
    moveDir += right * velocity.x;
    moveDir += up * velocity.y;

    position += moveDir * moveSpeed * deltaTime;
  }

  // Rebuild orientation from yaw/pitch Euler angles.
  glm::quat quatYaw = glm::angleAxis(glm::radians(yaw), glm::vec3(0, 1, 0));
  glm::quat quatPitch = glm::angleAxis(glm::radians(pitch), glm::vec3(1, 0, 0));
  orientation = glm::normalize(quatYaw * quatPitch);

  updateViewMatrix();
}

void Camera::updateViewMatrix() {
  glm::vec3 forward = glm::rotate(orientation, glm::vec3(0, 0, -1));
  glm::vec3 up = glm::vec3(0, 1, 0);

  viewMatrix = glm::lookAt(position, position + forward, up);
}

void Camera::updateProjectionMatrix() {
  projectionMatrix = perspectiveReverseZ(fovY, aspectRatio, nearZ, farZ);
}

glm::mat4 Camera::perspectiveReverseZ(float fovY_, float aspect, float n,
                                      float f) {
  float tanHalfFovy = std::tan(glm::radians(fovY_) / 2.0f);

  glm::mat4 result(0);
  result[0][0] = 1.0f / (aspect * tanHalfFovy);
  result[1][1] = -1.0f / tanHalfFovy;
  result[2][2] = n / (f - n);
  result[2][3] = -1.0f;
  result[3][2] = (n * f) / (f - n);

  return result;
}

void Camera::processKeyboard(const bool *keys) {
  velocity = glm::vec3(0);

  if (keys[SDL_SCANCODE_W])   velocity.z -= 1.0f; // forward  (-Z)
  if (keys[SDL_SCANCODE_S])   velocity.z += 1.0f; // backward (+Z)
  if (keys[SDL_SCANCODE_A])   velocity.x -= 1.0f; // left
  if (keys[SDL_SCANCODE_D])   velocity.x += 1.0f; // right
  if (keys[SDL_SCANCODE_SPACE]) velocity.y += 1.0f; // up
  if (keys[SDL_SCANCODE_LCTRL]) velocity.y -= 1.0f; // down

  // Normalize to prevent diagonal movement from being faster than axis-aligned.
  if (glm::length(velocity) > 0.01f) {
    velocity = glm::normalize(velocity);
  }
}

void Camera::fitToScene(float sceneRadius) {
  // Place the camera outside the scene, looking toward the origin with a slight downward tilt.
  position = glm::vec3(0.0f, sceneRadius * 0.5f, sceneRadius * 2.0f);
  yaw   = 0.0f;
  pitch = -10.0f;

  // Rebuild orientation from the new yaw/pitch immediately (same as update()).
  glm::quat quatYaw   = glm::angleAxis(glm::radians(yaw),   glm::vec3(0, 1, 0));
  glm::quat quatPitch = glm::angleAxis(glm::radians(pitch), glm::vec3(1, 0, 0));
  orientation = glm::normalize(quatYaw * quatPitch);

  updateViewMatrix();
}

void Camera::processMouseMovement(float xoffset, float yoffset) {
  yaw += xoffset * mouseSensitivity;
  pitch += yoffset * mouseSensitivity;

  // Clamp pitch to prevent the camera from flipping past vertical.
  pitch = glm::clamp(pitch, -89.0f, 89.0f);
}

void Camera::setMouseSensitivity(float sensitivity) {
  mouseSensitivity = sensitivity;
}

void Camera::setPerspective(float fovYDegrees, float aspectRatio_, float nearZ_,
                            float farZ_) {
  fovY = fovYDegrees;
  aspectRatio = aspectRatio_;
  nearZ = nearZ_;
  farZ = farZ_;
  updateProjectionMatrix();
}
