#pragma once
#include "resource/Model.h"
#include <string>

class GLTFLoader {
public:
    // Load a glTF/glb file and return a populated Model.
    static Model loadGLTF(const std::string& filepath);
};
