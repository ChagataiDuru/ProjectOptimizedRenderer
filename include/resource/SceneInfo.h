#pragma once
#include <glm/glm.hpp>

// Normalization data computed from a model's AABB.
// The modelMatrix scales and centers the model so its longest axis = targetSize.
// All spatial systems (camera, shadows, bias) should reference normalizedRadius
// instead of hardcoded constants so they work correctly for any input model.
struct SceneInfo {
    glm::vec3 center;           // AABB center in model space
    float     scaleFactor;      // multiply model-space positions by this to reach world scale
    float     normalizedRadius; // bounding sphere radius after normalization (~targetSize / 2)
    glm::mat4 modelMatrix;      // Scale(scaleFactor) * Translate(-center)
};

// Compute a SceneInfo that normalizes the model so its longest AABB axis = targetSize.
SceneInfo computeSceneInfo(const glm::vec3& boundsMin,
                           const glm::vec3& boundsMax,
                           float targetSize = 10.0f);
