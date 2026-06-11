#pragma once

#include <cstdint>
#include <limits>

inline constexpr uint32_t kInvalidRenderHandleIndex = std::numeric_limits<uint32_t>::max();

struct MeshHandle {
    uint32_t index = kInvalidRenderHandleIndex;

    bool isValid() const { return index != kInvalidRenderHandleIndex; }
};

struct MaterialHandle {
    uint32_t index = kInvalidRenderHandleIndex;

    bool isValid() const { return index != kInvalidRenderHandleIndex; }
};

struct TextureHandle {
    uint32_t index = kInvalidRenderHandleIndex;

    bool isValid() const { return index != kInvalidRenderHandleIndex; }
};
