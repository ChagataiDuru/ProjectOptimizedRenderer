#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include <array>

// Vertex layout (64 bytes, aligned for GPU buffer upload):
//   offset  0 : position (vec3, 12 bytes)
//   offset 12 : normal   (vec3, 12 bytes)
//   offset 24 : uv       (vec2,  8 bytes)
//   offset 32 : tangent  (vec3, 12 bytes)
//   offset 44 : _pad     (float[5], 20 bytes) → total 64 bytes
struct Vertex {
    glm::vec3 position;  // location 0
    glm::vec3 normal;    // location 1
    glm::vec2 uv;        // location 2
    glm::vec3 tangent;   // location 3
    float     _pad[5]{};  // pad to 64 bytes (zero-initialized; not read by shaders)

    // Describes how the vertex buffer is laid out: one interleaved binding,
    // advancing one vertex worth of data per vertex (not per instance).
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription desc{};
        desc.binding   = 0;
        desc.stride    = sizeof(Vertex);  // 64 bytes
        desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return desc;
    }

    // Describes how each shader input attribute maps to memory inside a vertex.
    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 4> attrs{};

        // position: vec3 → R32G32B32_SFLOAT
        attrs[0].binding  = 0;
        attrs[0].location = 0;
        attrs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset   = offsetof(Vertex, position);

        // normal: vec3 → R32G32B32_SFLOAT
        attrs[1].binding  = 0;
        attrs[1].location = 1;
        attrs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset   = offsetof(Vertex, normal);

        // uv: vec2 → R32G32_SFLOAT
        attrs[2].binding  = 0;
        attrs[2].location = 2;
        attrs[2].format   = VK_FORMAT_R32G32_SFLOAT;
        attrs[2].offset   = offsetof(Vertex, uv);

        // tangent: vec3 → R32G32B32_SFLOAT
        attrs[3].binding  = 0;
        attrs[3].location = 3;
        attrs[3].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[3].offset   = offsetof(Vertex, tangent);

        return attrs;
    }
};

static_assert(sizeof(Vertex) == 64, "Vertex struct must be exactly 64 bytes");
