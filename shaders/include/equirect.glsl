// Equirectangular (latitude-longitude) direction mapping shared by the sky, the IBL
// bake and the PBR shader:
//   u in [0,1] -> azimuth [-pi, pi]      (u = 0.5 looks along +X)
//   v in [0,1] -> elevation [+pi/2, -pi/2] (v = 0 is the zenith)
#ifndef POR_EQUIRECT_GLSL
#define POR_EQUIRECT_GLSL

#define POR_PI 3.14159265358979323846

vec2 dirToEquirect(vec3 dir)
{
    float azimuth   = atan(dir.z, dir.x);
    float elevation = asin(clamp(dir.y, -1.0, 1.0));
    return vec2(azimuth / (2.0 * POR_PI) + 0.5, 0.5 - elevation / POR_PI);
}

vec3 equirectToDir(vec2 uv)
{
    float azimuth   = (uv.x - 0.5) * 2.0 * POR_PI;
    float elevation = (0.5 - uv.y) * POR_PI;
    float c = cos(elevation);
    return vec3(c * cos(azimuth), sin(elevation), c * sin(azimuth));
}

#endif // POR_EQUIRECT_GLSL
