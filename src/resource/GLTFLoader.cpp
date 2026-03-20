// CGLTF_IMPLEMENTATION must be defined in exactly one translation unit.
// This compiles the cgltf function bodies into this .o file; all other TUs
// that include cgltf.h without the define see only declarations.
#define CGLTF_IMPLEMENTATION
#include "resource/GLTFLoader.h"

#include <cgltf.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <limits>
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

    // Directory of the .gltf file — used to resolve relative image URIs.
    const std::string dir = filepath.substr(0, filepath.find_last_of("/\\") + 1);

    // ── Textures (indexed by cgltf image index) ──────────────────────────────
    // We iterate images rather than textures: a cgltf_texture wraps an image +
    // sampler, but for deduplication we track images (the actual file data).
    for (cgltf_size i = 0; i < data->images_count; ++i) {
        const cgltf_image& img = data->images[i];
        TextureEntry entry;
        entry.type = TextureType::Color;  // Default; corrected per-material below
        if (img.uri) {
            entry.path = dir + img.uri;
        } else {
            spdlog::warn("GLTFLoader: embedded texture without URI at index {} — skipping", i);
            entry.path = "";
        }
        model.textures.push_back(std::move(entry));
    }

    // Helper: convert cgltf_texture* → index into model.textures (-1 if null).
    // Pointer arithmetic is valid because data->images is a contiguous array.
    auto textureIndex = [&](const cgltf_texture* tex) -> int32_t {
        if (!tex || !tex->image) return -1;
        return static_cast<int32_t>(tex->image - data->images);
    };

    // ── Materials ────────────────────────────────────────────────────────────
    for (cgltf_size mi = 0; mi < data->materials_count; ++mi) {
        const cgltf_material& gltfMat = data->materials[mi];
        Material mat;
        mat.name = gltfMat.name ? gltfMat.name : ("Material_" + std::to_string(mi));

        if (gltfMat.has_pbr_metallic_roughness) {
            const auto& pbr = gltfMat.pbr_metallic_roughness;

            mat.albedoTextureIndex            = textureIndex(pbr.base_color_texture.texture);
            mat.metallicRoughnessTextureIndex = textureIndex(pbr.metallic_roughness_texture.texture);

            mat.baseColorFactor = glm::vec4(
                pbr.base_color_factor[0], pbr.base_color_factor[1],
                pbr.base_color_factor[2], pbr.base_color_factor[3]);
            mat.metallicFactor  = pbr.metallic_factor;
            mat.roughnessFactor = pbr.roughness_factor;
        }

        mat.normalTextureIndex = textureIndex(gltfMat.normal_texture.texture);
        mat.doubleSided        = gltfMat.double_sided;

        if (gltfMat.alpha_mode == cgltf_alpha_mode_mask)
            mat.alphaMode = Material::AlphaMode::Mask;
        else if (gltfMat.alpha_mode == cgltf_alpha_mode_blend)
            mat.alphaMode = Material::AlphaMode::Blend;
        mat.alphaCutoff = gltfMat.alpha_cutoff;

        model.materials.push_back(std::move(mat));
    }

    // Fix up TextureType for non-color roles: normal maps and metallic-roughness
    // textures contain linear data and must be sampled as UNORM, not SRGB.
    for (const auto& mat : model.materials) {
        if (mat.normalTextureIndex >= 0)
            model.textures[static_cast<size_t>(mat.normalTextureIndex)].type = TextureType::Linear;
        if (mat.metallicRoughnessTextureIndex >= 0)
            model.textures[static_cast<size_t>(mat.metallicRoughnessTextureIndex)].type = TextureType::Linear;
    }

    // ── Meshes — one Mesh per glTF primitive ─────────────────────────────────
    // In glTF, a mesh is an organizational container; primitives are the actual
    // draw calls and each may have a distinct material.  Splitting into one Mesh
    // per primitive lets the draw loop bind the correct material per call.
    for (cgltf_size mi = 0; mi < data->meshes_count; ++mi) {
        const cgltf_mesh& gltfMesh = data->meshes[mi];

        for (cgltf_size pi = 0; pi < gltfMesh.primitives_count; ++pi) {
            const cgltf_primitive& prim = gltfMesh.primitives[pi];

            Mesh mesh;
            mesh.name = (gltfMesh.name ? std::string(gltfMesh.name) : std::string("Mesh_") + std::to_string(mi))
                        + "_prim" + std::to_string(pi);

            // Material index: pointer arithmetic against data->materials array
            if (prim.material) {
                mesh.materialIndex = static_cast<int32_t>(prim.material - data->materials);
            }

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

            // Indices are 0-based per-primitive — each Mesh has its own vertex array.
            // The Renderer's flatten loop adds vertexBase when building the combined buffer.
            if (prim.indices) {
                for (cgltf_size ii = 0; ii < prim.indices->count; ++ii) {
                    mesh.indices.push_back(
                        static_cast<uint32_t>(cgltf_accessor_read_index(prim.indices, ii)));
                }
            }

            if (!mesh.vertices.empty())
                model.meshes.push_back(std::move(mesh));
        }
    }

    // Compute scene AABB from all vertex positions (used by shadow camera framing).
    glm::vec3 aabbMin( std::numeric_limits<float>::max());
    glm::vec3 aabbMax(-std::numeric_limits<float>::max());
    for (const auto& mesh : model.meshes) {
        for (const auto& v : mesh.vertices) {
            aabbMin = glm::min(aabbMin, v.position);
            aabbMax = glm::max(aabbMax, v.position);
        }
    }
    model.boundsMin = aabbMin;
    model.boundsMax = aabbMax;

    cgltf_free(data);

    spdlog::info("Loaded '{}': {} meshes, {} materials, {} textures, {} vertices, {} indices",
                 filepath, model.meshes.size(), model.materials.size(), model.textures.size(),
                 model.getTotalVertexCount(), model.getTotalIndexCount());

    return model;
}
