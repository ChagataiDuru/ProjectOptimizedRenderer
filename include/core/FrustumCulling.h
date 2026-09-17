#pragma once

#include <glm/glm.hpp>

#include <array>
#include <cstddef>

// CPU-side culling helpers shared by the main scene pass and the shadow pass.
namespace culling {

// Plane with an inward-pointing unit normal: points with dot(normal, p) + d >= 0 are inside.
struct Plane {
    glm::vec3 normal = glm::vec3(0.0f);
    float d = 0.0f;
};

// Gribb–Hartmann extraction for a clip matrix with an OpenGL-style [-1, 1] depth range
// (e.g. glm::perspectiveRH_NO). Order: left, right, bottom, top, near, far.
std::array<Plane, 6> extractFrustumPlanesNO(const glm::mat4& viewProj);

// True when the whole AABB lies on the outside of the plane.
bool aabbOutsidePlane(const glm::vec3& bmin, const glm::vec3& bmax, const Plane& plane);

// True when the AABB is outside at least one of the planes.
bool aabbOutsideAny(const glm::vec3& bmin, const glm::vec3& bmax,
                    const Plane* planes, size_t count);

} // namespace culling
