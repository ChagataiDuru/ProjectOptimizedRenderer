#pragma once
#include "resource/Vertex.h"
#include <vector>
#include <string>

struct Mesh {
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // Index into Model::materials (-1 = no material)
    int32_t materialIndex = -1;

    Mesh() = default;
};
