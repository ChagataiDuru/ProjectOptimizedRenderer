#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class Camera {
public:
    Camera();

    // Input processing
    void processKeyboard(const bool* keys);  // SDL3 keyboard state
    void processMouseMovement(float xoffset, float yoffset);

    // Configuration
    void setMouseSensitivity(float sensitivity);
    void setPerspective(float fovYDegrees, float aspectRatio, float nearZ, float farZ);

    // Phase 3.7: position and orient the camera to frame a normalized scene.
    // Call after loading a model — places the camera at 2× sceneRadius, looking toward origin.
    void fitToScene(float sceneRadius);

    // Update (called once per frame)
    void update(float deltaTime);

    // Getters
    const glm::mat4& getViewMatrix()       const { return viewMatrix; }
    const glm::mat4& getProjectionMatrix() const { return projectionMatrix; }
    const glm::vec3& getPosition()         const { return position; }
    const glm::quat& getOrientation()      const { return orientation; }
    float            getNearZ()            const { return nearZ; }
    float            getFarZ()             const { return farZ; }

private:
    // Position and orientation
    glm::vec3 position = glm::vec3(0, 0, 3);
    glm::quat orientation = glm::quat(1, 0, 0, 0);  // identity quaternion

    // Input state
    glm::vec3 velocity = glm::vec3(0);  // accumulated movement direction
    float moveSpeed = 5.0f;             // units per second
    float mouseSensitivity = 0.1f;

    // Euler angles for camera rotation
    float yaw = 0.0f;    // rotation around Y axis (left/right)
    float pitch = 0.0f;  // rotation around X axis (up/down)

    // Matrices (cached, updated in update())
    glm::mat4 viewMatrix = glm::mat4(1);
    glm::mat4 projectionMatrix = glm::mat4(1);

    // Projection parameters
    float fovY = 45.0f;
    float aspectRatio = 16.0f / 9.0f;
    float nearZ = 0.01f;
    float farZ = 1000.0f;

    // Helper methods
    void updateViewMatrix();
    void updateProjectionMatrix();

    // Reverse-Z perspective matrix (depth range [1, 0] instead of [0, 1])
    glm::mat4 perspectiveReverseZ(float fovY, float aspect, float n, float f);
};
