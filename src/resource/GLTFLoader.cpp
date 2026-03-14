// CGLTF_IMPLEMENTATION must be defined in exactly one translation unit.
// This compiles the cgltf function bodies into this .o file; all other TUs
// that include cgltf.h without the define see only declarations.
#define CGLTF_IMPLEMENTATION
#include "resource/GLTFLoader.h"

#include <cgltf.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

Model GLTFLoader::loadGLTF(const std::string& filepath)
{
    spdlog::info("Loading glTF: {}", filepath);

    cgltf_options options = {};
    cgltf_data*   data    = nullptr;

    if (cgltf_parse_file(&options, filepath.c_str(), &data) != cgltf_result_success)
        throw std::runtime_error("Failed to parse glTF: " + filepath);

    if (cgltf_load_buffers(&options, data, filepath.c_str()) != cgltf_result_success) {
        cgltf_free(data);
        throw std::runtime_error("Failed to load glTF buffers: " + filepath);
    }

    Model model(filepath);

    for (cgltf_size mi = 0; mi < data->meshes_count; ++mi) {
        const cgltf_mesh& gltfMesh = data->meshes[mi];

        Mesh mesh;
        mesh.name = gltfMesh.name ? gltfMesh.name : ("Mesh_" + std::to_string(mi));

        for (cgltf_size pi = 0; pi < gltfMesh.primitives_count; ++pi) {
            const cgltf_primitive& prim = gltfMesh.primitives[pi];

            // Tracks the global vertex offset within this mesh so per-primitive
            // 0-based indices are converted to mesh-wide absolute indices.
            const uint32_t vertexOffset = static_cast<uint32_t>(mesh.vertices.size());

            // Find attribute accessors.
            const cgltf_accessor* posAcc     = nullptr;
            const cgltf_accessor* normAcc    = nullptr;
            const cgltf_accessor* uvAcc      = nullptr;
            const cgltf_accessor* tangentAcc = nullptr;

            for (cgltf_size ai = 0; ai < prim.attributes_count; ++ai) {
                const cgltf_attribute& attr = prim.attributes[ai];
                switch (attr.type) {
                    case cgltf_attribute_type_position: posAcc     = attr.data; break;
                    case cgltf_attribute_type_normal:   normAcc    = attr.data; break;
                    // Take only the first UV set (TEXCOORD_0).
                    case cgltf_attribute_type_texcoord: if (!uvAcc) uvAcc = attr.data; break;
                    case cgltf_attribute_type_tangent:  tangentAcc = attr.data; break;
                    default: break;
                }
            }

            if (!posAcc) continue;  // primitive without positions — skip

            const cgltf_size vertexCount = posAcc->count;
            for (cgltf_size vi = 0; vi < vertexCount; ++vi) {
                Vertex v;

                float pos[3] = {};
                cgltf_accessor_read_float(posAcc, vi, pos, 3);
                v.position = glm::vec3(pos[0], pos[1], pos[2]);

                if (normAcc) {
                    float n[3] = {};
                    cgltf_accessor_read_float(normAcc, vi, n, 3);
                    v.normal = glm::normalize(glm::vec3(n[0], n[1], n[2]));
                } else {
                    v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
                }

                if (uvAcc) {
                    float uv[2] = {};
                    cgltf_accessor_read_float(uvAcc, vi, uv, 2);
                    v.uv = glm::vec2(uv[0], uv[1]);
                }

                if (tangentAcc) {
                    // glTF tangents are vec4; w = ±1 for handedness (not stored in Vertex).
                    float t[4] = {};
                    cgltf_accessor_read_float(tangentAcc, vi, t, 4);
                    v.tangent = glm::vec3(t[0], t[1], t[2]);
                } else {
                    v.tangent = glm::vec3(1.0f, 0.0f, 0.0f);
                }

                mesh.vertices.push_back(v);
            }

            if (prim.indices) {
                const cgltf_size indexCount = prim.indices->count;
                for (cgltf_size ii = 0; ii < indexCount; ++ii) {
                    mesh.indices.push_back(
                        static_cast<uint32_t>(cgltf_accessor_read_index(prim.indices, ii))
                        + vertexOffset);
                }
            }
        }

        if (!mesh.vertices.empty())
            model.meshes.push_back(std::move(mesh));
    }

    cgltf_free(data);

    spdlog::info("Loaded '{}': {} meshes, {} vertices, {} indices",
                 filepath, model.meshes.size(),
                 model.getTotalVertexCount(), model.getTotalIndexCount());

    return model;
}
