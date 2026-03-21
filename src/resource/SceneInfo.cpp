#include "resource/SceneInfo.h"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>

SceneInfo computeSceneInfo(const glm::vec3& boundsMin,
                           const glm::vec3& boundsMax,
                           float targetSize)
{
    const glm::vec3 center = (boundsMin + boundsMax) * 0.5f;
    const glm::vec3 extent = boundsMax - boundsMin;
    const float longest = std::max({ extent.x, extent.y, extent.z });

    const float scale = (longest > 0.0001f) ? targetSize / longest : 1.0f;

    // Build the normalization matrix: first translate center to origin, then scale.
    // Column-major GLM: M = Scale * Translate, applied as M * v in the shader.
    glm::mat4 m(1.0f);
    m = glm::scale(m, glm::vec3(scale));
    m = glm::translate(m, -center);

    return {
        .center           = center,
        .scaleFactor      = scale,
        .normalizedRadius = glm::length(extent) * 0.5f * scale,
        .modelMatrix      = m,
    };
}
