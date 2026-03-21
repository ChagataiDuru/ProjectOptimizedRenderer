#version 460 core

// VSM shadow fragment shader.
// Outputs first and second depth moments to a RG32_SFLOAT color attachment.
// The blur compute pass then blurs these moments, and the PBR shader uses
// Chebyshev's inequality to estimate the probability that a fragment is lit.

layout(location = 0) out vec2 outMoments;

void main() {
    float depth = gl_FragCoord.z;  // [0, 1] standard Z from rasterizer
    outMoments = vec2(depth, depth * depth);
}
