// Single translation unit that compiles the stb_image implementation.
// Mirrors the VMA pattern in VmaImpl.cpp.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_GIF   // We don't need GIF support
#define STBI_NO_PSD   // We don't need PSD support
// STBI_NO_HDR removed in Phase 6.5 — loadHdrPanorama() uses stbi_loadf()
#include <stb_image.h>
