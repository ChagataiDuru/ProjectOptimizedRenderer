#include "core/FrustumCulling.h"

namespace culling {

std::array<Plane, 6> extractFrustumPlanesNO(const glm::mat4& viewProj)
{
    auto row = [&](int i) -> glm::vec4 {
        return { viewProj[0][i], viewProj[1][i], viewProj[2][i], viewProj[3][i] };
    };
    const glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);

    std::array<Plane, 6> planes{};
    auto set = [&](int idx, glm::vec4 v) {
        const float len = glm::length(glm::vec3(v));
        if (len > 1e-8f) v /= len;
        planes[idx] = { glm::vec3(v), v.w };
    };
    set(0, r3 + r0);
    set(1, r3 - r0);
    set(2, r3 + r1);
    set(3, r3 - r1);
    set(4, r3 + r2);
    set(5, r3 - r2);
    return planes;
}

bool aabbOutsidePlane(const glm::vec3& bmin, const glm::vec3& bmax, const Plane& plane)
{
    const glm::vec3 pVertex(
        (plane.normal.x >= 0.0f) ? bmax.x : bmin.x,
        (plane.normal.y >= 0.0f) ? bmax.y : bmin.y,
        (plane.normal.z >= 0.0f) ? bmax.z : bmin.z);
    return (glm::dot(plane.normal, pVertex) + plane.d) < 0.0f;
}

bool aabbOutsideAny(const glm::vec3& bmin, const glm::vec3& bmax,
                    const Plane* planes, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (aabbOutsidePlane(bmin, bmax, planes[i])) return true;
    }
    return false;
}

} // namespace culling
