// Single translation unit that compiles the stb_image implementation.
// Mirrors the VMA pattern in VmaImpl.cpp.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_GIF        // We don't need GIF support
#define STBI_NO_PSD        // We don't need PSD support
#define STBI_NO_HDR        // HDR will use a different loader if needed later
#include <stb_image.h>
