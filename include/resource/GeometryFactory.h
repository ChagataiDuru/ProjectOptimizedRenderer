#pragma once

#include "resource/Vertex.h"
#include <vector>

// GeometryFactory generates hardcoded mesh data for primitive shapes.
// MeshData is a plain struct (no GPU handles) — callers own the lifetime
// and are responsible for uploading to vertex/index buffers.
class GeometryFactory {
public:
    struct MeshData {
        std::vector<Vertex>   vertices;  // 24 for a cube (4 per face)
        std::vector<uint32_t> indices;   // 36 for a cube (6 per face)
    };

    // Unit cube centered at origin, side length 1.
    // Each face has its own 4 vertices (no sharing across faces) so that
    // per-face normals and UVs are independent.
    // Winding order: counter-clockwise when viewed from outside (Vulkan default).
    // Tangents are (1,0,0) placeholders; refine during normal-mapping phase.
    static MeshData createCube() {
        MeshData mesh;
        mesh.vertices.reserve(24);
        mesh.indices.reserve(36);

        const float s = 0.5f;  // half-extent

        // ── Front face (Z+, normal = 0,0,1) ──────────────────────────────
        // Viewed from +Z: BL→BR→TR is CCW. UV origin at bottom-left.
        mesh.vertices.push_back({{-s, -s,  s}, {0.f, 0.f, 1.f}, {0.f, 0.f}, {1.f, 0.f, 0.f}});  //  0 BL
        mesh.vertices.push_back({{ s, -s,  s}, {0.f, 0.f, 1.f}, {1.f, 0.f}, {1.f, 0.f, 0.f}});  //  1 BR
        mesh.vertices.push_back({{ s,  s,  s}, {0.f, 0.f, 1.f}, {1.f, 1.f}, {1.f, 0.f, 0.f}});  //  2 TR
        mesh.vertices.push_back({{-s,  s,  s}, {0.f, 0.f, 1.f}, {0.f, 1.f}, {1.f, 0.f, 0.f}});  //  3 TL

        // ── Back face (Z-, normal = 0,0,-1) ──────────────────────────────
        // Viewed from -Z: +X axis appears flipped, so vertices start at +X side.
        mesh.vertices.push_back({{ s, -s, -s}, {0.f, 0.f, -1.f}, {0.f, 0.f}, {1.f, 0.f, 0.f}}); //  4 BL
        mesh.vertices.push_back({{-s, -s, -s}, {0.f, 0.f, -1.f}, {1.f, 0.f}, {1.f, 0.f, 0.f}}); //  5 BR
        mesh.vertices.push_back({{-s,  s, -s}, {0.f, 0.f, -1.f}, {1.f, 1.f}, {1.f, 0.f, 0.f}}); //  6 TR
        mesh.vertices.push_back({{ s,  s, -s}, {0.f, 0.f, -1.f}, {0.f, 1.f}, {1.f, 0.f, 0.f}}); //  7 TL

        // ── Right face (X+, normal = 1,0,0) ──────────────────────────────
        // Viewed from +X: +Z is to the left, -Z is to the right.
        mesh.vertices.push_back({{ s, -s,  s}, {1.f, 0.f, 0.f}, {0.f, 0.f}, {1.f, 0.f, 0.f}});  //  8 BL
        mesh.vertices.push_back({{ s, -s, -s}, {1.f, 0.f, 0.f}, {1.f, 0.f}, {1.f, 0.f, 0.f}});  //  9 BR
        mesh.vertices.push_back({{ s,  s, -s}, {1.f, 0.f, 0.f}, {1.f, 1.f}, {1.f, 0.f, 0.f}});  // 10 TR
        mesh.vertices.push_back({{ s,  s,  s}, {1.f, 0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f, 0.f}});  // 11 TL

        // ── Left face (X-, normal = -1,0,0) ──────────────────────────────
        // Viewed from -X: -Z is to the left, +Z is to the right.
        mesh.vertices.push_back({{-s, -s, -s}, {-1.f, 0.f, 0.f}, {0.f, 0.f}, {1.f, 0.f, 0.f}}); // 12 BL
        mesh.vertices.push_back({{-s, -s,  s}, {-1.f, 0.f, 0.f}, {1.f, 0.f}, {1.f, 0.f, 0.f}}); // 13 BR
        mesh.vertices.push_back({{-s,  s,  s}, {-1.f, 0.f, 0.f}, {1.f, 1.f}, {1.f, 0.f, 0.f}}); // 14 TR
        mesh.vertices.push_back({{-s,  s, -s}, {-1.f, 0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f, 0.f}}); // 15 TL

        // ── Top face (Y+, normal = 0,1,0) ────────────────────────────────
        // Viewed from +Y looking down: UV(0,0) at -Z side (visually "near").
        mesh.vertices.push_back({{-s,  s,  s}, {0.f, 1.f, 0.f}, {0.f, 0.f}, {1.f, 0.f, 0.f}});  // 16 FL
        mesh.vertices.push_back({{ s,  s,  s}, {0.f, 1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f, 0.f}});  // 17 FR
        mesh.vertices.push_back({{ s,  s, -s}, {0.f, 1.f, 0.f}, {1.f, 1.f}, {1.f, 0.f, 0.f}});  // 18 BR
        mesh.vertices.push_back({{-s,  s, -s}, {0.f, 1.f, 0.f}, {0.f, 1.f}, {1.f, 0.f, 0.f}});  // 19 BL

        // ── Bottom face (Y-, normal = 0,-1,0) ────────────────────────────
        // Viewed from -Y looking up: UV(0,0) at -Z side.
        mesh.vertices.push_back({{-s, -s, -s}, {0.f, -1.f, 0.f}, {0.f, 0.f}, {1.f, 0.f, 0.f}}); // 20 BL
        mesh.vertices.push_back({{ s, -s, -s}, {0.f, -1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f, 0.f}}); // 21 BR
        mesh.vertices.push_back({{ s, -s,  s}, {0.f, -1.f, 0.f}, {1.f, 1.f}, {1.f, 0.f, 0.f}}); // 22 FR
        mesh.vertices.push_back({{-s, -s,  s}, {0.f, -1.f, 0.f}, {0.f, 1.f}, {1.f, 0.f, 0.f}}); // 23 FL

        // Each face: two CCW triangles from a quad (0,1,2) and (0,2,3).
        mesh.indices = {
             0,  1,  2,   0,  2,  3,  // Front
             4,  5,  6,   4,  6,  7,  // Back
             8,  9, 10,   8, 10, 11,  // Right
            12, 13, 14,  12, 14, 15,  // Left
            16, 17, 18,  16, 18, 19,  // Top
            20, 21, 22,  20, 22, 23,  // Bottom
        };

        return mesh;
    }
};
