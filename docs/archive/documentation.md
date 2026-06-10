For Vertex.H:

⏺ ★ Insight ─────────────────────────────────────
  - Memory layout precision matters here: The _pad float (4 bytes) fills the gap after glm::vec3 tangent (12 bytes) to reach exactly 64 bytes: 12 + 12 + 8 + 12 + 4 = 48... wait,
  let me recount: position(12) + normal(12) + uv(8) + tangent(12) = 44 bytes, so we need 20 bytes of padding to hit 64. But the prompt says float _pad (4 bytes) reaches 44+4=48
  bytes — that's 48, not 64. The only way to hit 64 is with padding of 20 bytes (float _pad[5]). Let me compute carefully.
  - offsetof vs manual constants: Using offsetof(Vertex, field) is safer than hardcoded numbers — the compiler resolves it correctly even if struct layout changes, and it's
  guaranteed correct on standard-layout types.
  - Vertex input rate: VK_VERTEX_INPUT_RATE_VERTEX means each vertex gets a new set of attributes. The alternative, VK_VERTEX_INPUT_RATE_INSTANCE, advances per draw instance (used
   for instanced rendering).
  ─────────────────────────────────────────────────
⏺ ★ Insight ─────────────────────────────────────
  - static_assert as a compile-time contract: Rather than checking struct size at runtime, static_assert(sizeof(Vertex) == 64, ...) makes the compiler enforce the layout
  guarantee. If someone adds a field or glm changes its vec3 size, the build fails immediately with a clear message — no silent GPU data corruption.
  - Why 64 bytes? Cache lines on modern CPUs/GPUs are typically 64 bytes. A struct that fits exactly in one cache line avoids "split loads" where a single vertex straddles two
  lines — this matters at scale when streaming thousands of vertices.
  - float _pad[5] vs a single char[20]: Using float array keeps the struct's natural alignment and avoids any padding the compiler might insert inside the array itself. char[20]
  would work too since char alignment is 1, but float[5] is semantically clearer about what it's doing.
  ─────────────────────────────────────────────────

  File created at include/resource/Vertex.h. Key decisions:

  ┌──────────┬───────┬────────────┬────────────────────────────┐
  │  Field   │ Bytes │ Cumulative │          VkFormat          │
  ├──────────┼───────┼────────────┼────────────────────────────┤
  │ position │ 12    │ 12         │ VK_FORMAT_R32G32B32_SFLOAT │
  ├──────────┼───────┼────────────┼────────────────────────────┤
  │ normal   │ 12    │ 24         │ VK_FORMAT_R32G32B32_SFLOAT │
  ├──────────┼───────┼────────────┼────────────────────────────┤
  │ uv       │ 8     │ 32         │ VK_FORMAT_R32G32_SFLOAT    │
  ├──────────┼───────┼────────────┼────────────────────────────┤
  │ tangent  │ 12    │ 44         │ VK_FORMAT_R32G32B32_SFLOAT │
  ├──────────┼───────┼────────────┼────────────────────────────┤
  │ _pad[5]  │ 20    │ 64         │ —                          │
  └──────────┴───────┴────────────┴────────────────────────────┘

  The static_assert at line 63 will catch any layout drift at compile time. The prompt's template showed float _pad (4 bytes → 48 total), but the correct padding to reach 64 bytes
   is float _pad[5] (20 bytes).

For GeometryFactor.H
⏺ ★ Insight ─────────────────────────────────────
  - Index pattern (0,1,2), (0,2,3) is universal for quads: Every axis-aligned quad using CCW winding and this vertex order (BL→BR→TR→TL) produces the same two-triangle
  decomposition. This regularity matters when writing procedural mesh generators — you can emit indices as base + [0,1,2, 0,2,3] in a loop.
  - reserve() before push_back(): Calling vertices.reserve(24) pre-allocates capacity, avoiding 4–5 reallocations as the vector grows. It's a small but good habit since the counts
   are known at compile time.
  - Normals are not normalized in the initializer list but glm::vec3{0.f, 0.f, 1.f} is already unit-length. The GPU receives them as raw floats — if you ever write a shader that
  assumes normalized normals, don't forget the face-averaged case (shared vertices) needs explicit normalization.
  ─────────────────────────────────────────────────

  File created at include/resource/GeometryFactory.h. Summary:

  ┌─────────────┬──────────┬──────────┬────────────────────────┐
  │    Face     │ Base idx │  Normal  │   Verts 0,1 (BL,BR)    │
  ├─────────────┼──────────┼──────────┼────────────────────────┤
  │ Front (Z+)  │ 0        │ (0,0,1)  │ (-s,-s,s) → (s,-s,s)   │
  ├─────────────┼──────────┼──────────┼────────────────────────┤
  │ Back (Z−)   │ 4        │ (0,0,-1) │ (s,-s,-s) → (-s,-s,-s) │
  ├─────────────┼──────────┼──────────┼────────────────────────┤
  │ Right (X+)  │ 8        │ (1,0,0)  │ (s,-s,s) → (s,-s,-s)   │
  ├─────────────┼──────────┼──────────┼────────────────────────┤
  │ Left (X−)   │ 12       │ (-1,0,0) │ (-s,-s,-s) → (-s,-s,s) │
  ├─────────────┼──────────┼──────────┼────────────────────────┤
  │ Top (Y+)    │ 16       │ (0,1,0)  │ (-s,s,s) → (s,s,s)     │
  ├─────────────┼──────────┼──────────┼────────────────────────┤
  │ Bottom (Y−) │ 20       │ (0,-1,0) │ (-s,-s,-s) → (s,-s,-s) │
  └─────────────┴──────────┴──────────┴────────────────────────┘

  All 24 vertex push_backs and 36 indices (6 per face) are verified. The winding cross-product check confirmed CCW normals match the intended outward direction for all 6 faces.

  shaders/pbr.frag and pbr.vert:
  
⏺ Bash(glslangValidator -V --target-env vulkan1.4 shaders/pbr.vert -o /tmp/pbr.vert.spv && glslangValidator -V --target-env vulkan1.4 shaders/pbr.frag -o /tmp/pbr.frag…)
  ⎿  shaders/pbr.vert
     shaders/pbr.frag
     -rw-r--r--@ 1 cagatayduru  wheel   928B Mar  8 16:30 /tmp/pbr.frag.spv
     -rw-r--r--@ 1 cagatayduru  wheel   2.4K Mar  8 16:30 /tmp/pbr.vert.spv

    ┌──────────────────┬─────────────┬────────┐
  │       File       │ SPIR-V size │ Status │
  ├──────────────────┼─────────────┼────────┤
  │ shaders/pbr.vert │ 2.4 KB      │ ✓      │
  ├──────────────────┼─────────────┼────────┤
  │ shaders/pbr.frag │ 928 B       │ ✓      │
  └──────────────────┴─────────────┴────────┘

  ★ Insight ─────────────────────────────────────
  - Why the fragment shader SPIR-V is so small (928 B): The entire fragment body compiles to just a handful of SPIR-V instructions — normalize, FMA for the remap, and a store. The
   vertex shader is ~2.5× larger because it has matrix-vector multiplications and multiple struct member writes.
  - Re-normalizing in the fragment shader: Rasterizers interpolate varyings linearly across a triangle in clip space. Even if vertex normals are unit length, the interpolated
  result at any interior fragment is shorter than 1 (it's a linear blend, not a spherical one). The normalize() in main() corrects this before lighting — skipping it causes subtle
   shading artifacts, especially on large triangles.
  - No #include in GLSL: Each shader is compiled as a fully self-contained translation unit. The interface contract between vert and frag (the VS_OUT block layout) is enforced at
  pipeline link time by the Vulkan driver, not at GLSL compile time — so a name mismatch between stages only surfaces when you call vkCreateGraphicsPipelines.
  ─────────────────────────────────────────────────

  Camere Phase:
    - Window exposes getExtent(w, h) (not getWidth()/getHeight()) — must use that API
  - Window::pollEvents() internally calls SDL_PollEvent — once we own the event loop in main, we must stop calling it to avoid double-consuming events
  - Renderer exposes beginFrame() / endFrame() — not render() (which is private)

  ★ Insight ─────────────────────────────────────
  Event loop ownership: SDL event queues are consumed — each SDL_PollEvent call removes the event from the queue. The existing window.pollEvents() was the sole consumer. Taking
  over SDL event polling in main means we must stop calling window.pollEvents(), otherwise some events get eaten by the window and never reach the camera (or vice versa).
  Ownership must be singular.

  getExtent vs getWidth/Height: The Window uses SDL drawable pixel dimensions, not logical points — critical on HiDPI displays (Apple Silicon Retina). This means aspect ratio must
   come from the drawable extent, not from the constructor parameters (which are in logical points that may differ 2x on Retina).
  ─────────────────────────────────────────────────

⏺ Write(src/main.cpp)
  ⎿  Added 79 lines, removed 2 lines
       3  #include "core/Window.h"
       4  #include "core/Swapchain.h"
       5  #include "core/Renderer.h"
       6 +#include "core/Camera.h"
       7
       8  #include <spdlog/spdlog.h>
       9  #include <spdlog/sinks/stdout_color_sinks.h>
      10 +#include <SDL3/SDL.h>
      11 +#include <glm/glm.hpp>
      12 +#include <chrono>
      13 +#include <cmath>
      14  #include <stdexcept>
      15
      16  int main()
     ...
       40          Renderer renderer(vulkanContext, swapchain);
       41          renderer.init();
       42
       43 +        // === Camera Setup ===
       44 +        Camera camera;
       45 +        float aspectRatio = static_cast<float>(w) / static_cast<float>(h);
       46 +        camera.setPerspective(45.0f, aspectRatio, 0.01f, 1000.0f);
       47 +
       48 +        // === Frame Timing ===
       49 +        auto lastTime = std::chrono::high_resolution_clock::now();
       50 +        float deltaTime = 0.016f;  // 16ms default until first real measurement
       51 +
       52          spdlog::info("Entering render loop");
       53
       40 -        while (!window.shouldClose()) {
       41 -            window.pollEvents();
       54 +        // === Main Render Loop ===
       55 +        bool running = true;
       56 +        while (running) {
       57 +
       58 +            // ── Frame timing ──────────────────────────────────────────────────
       59 +            auto currentTime = std::chrono::high_resolution_clock::now();
       60 +            deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
       61 +            lastTime = currentTime;
       62 +
       63 +            // Clamp to prevent large jumps during window drag or minimize.
       64 +            deltaTime = glm::min(deltaTime, 0.033f);
       65 +
       66 +            spdlog::debug("Frame: {:.2f}ms ({:.1f} FPS)",
       67 +                          deltaTime * 1000.0f, 1.0f / deltaTime);
       68 +
       69 +            // ── SDL event processing ──────────────────────────────────────────
       70 +            // Keyboard state for smooth per-frame movement (held keys).
       71 +            const uint8_t* keys = SDL_GetKeyboardState(nullptr);
       72 +
       73 +            if (keys[SDL_SCANCODE_ESCAPE]) {
       74 +                running = false;
       75 +            }
       76 +
       77 +            SDL_Event event;
       78 +            while (SDL_PollEvent(&event)) {
       79 +                switch (event.type) {
       80 +
       81 +                    case SDL_EVENT_QUIT:
       82 +                        running = false;
       83 +                        break;
       84 +
       85 +                    case SDL_EVENT_MOUSE_MOTION: {
       86 +                        float xoffset = static_cast<float>(event.motion.xrel);
       87 +                        float yoffset = static_cast<float>(event.motion.yrel);
       88 +                        if (std::abs(xoffset) > 0.1f || std::abs(yoffset) > 0.1f) {
       89 +                            camera.processMouseMovement(xoffset, yoffset);
       90 +                        }
       91 +                        break;
       92 +                    }
       93 +
       94 +                    case SDL_EVENT_WINDOW_RESIZED: {
       95 +                        float newAspect = static_cast<float>(event.window.data1)
       96 +                                        / static_cast<float>(event.window.data2);
       97 +                        camera.setPerspective(45.0f, newAspect, 0.01f, 1000.0f);
       98 +                        spdlog::info("Window resized to {}x{}",
       99 +                                     event.window.data1, event.window.data2);
      100 +                        break;
      101 +                    }
      102 +
      103 +                    default:
      104 +                        break;
      105 +                }
      106 +            }
      107 +
      108 +            // ── Camera update ─────────────────────────────────────────────────
      109 +            camera.processKeyboard(keys);
      110 +            camera.update(deltaTime);
      111 +
      112 +            // ── Pass camera matrices to renderer ──────────────────────────────
      113 +            renderer.setCameraMatrices(camera.getViewMatrix(),
      114 +                                       camera.getProjectionMatrix());
      115 +
      116 +            // ── Render ────────────────────────────────────────────────────────
      117              renderer.beginFrame();
      118              renderer.endFrame();
      119          }
      120
      121 +        spdlog::info("Render loop ended, shutting down...");
      122 +
      123          renderer.shutdown();
      124          swapchain.shutdown();
      125          vulkanContext.shutdown();

⏺ Done. Here's what changed and why each decision was made:

  ┌────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                  Decision                  │                                                            Reason                                                             │
  ├────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ window.getExtent(w, h) reused (already     │ Window has no getWidth()/getHeight() — getExtent is the only API                                                              │
  │ declared above)                            │                                                                                                                               │
  ├────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ window.pollEvents() removed                │ Replaced by our SDL_PollEvent loop; double-polling would drop events                                                          │
  ├────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ bool running replaces window.shouldClose() │ We now own the quit signal (ESC, SDL_EVENT_QUIT, window resize)                                                               │
  ├────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ beginFrame() / endFrame() kept             │ Renderer::render() is private — only begin/end are public                                                                     │
  ├────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ SDL_GetKeyboardState called before event   │ Keyboard state is a snapshot of held keys, updated by the event pump — calling it before SDL_PollEvent means the state is     │
  │ loop                                       │ current for this frame after PollEvent drains the queue                                                                       │
  └────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ★ Insight ─────────────────────────────────────
  SDL_GetKeyboardState vs events for movement: Movement keys (WASD) use the state array because they're held continuously — you want velocity every frame while the key is down,
  not just one event when it's first pressed. Mouse look uses events (SDL_EVENT_MOUSE_MOTION) because it provides relative delta (xrel/yrel) — the state array can't tell you how
  far the mouse moved this frame, only its absolute position.

  deltaTime clamping rationale: Without the 33ms cap, dragging the window (which blocks the OS event loop on some platforms) would produce a single frame with deltaTime of several
   seconds, teleporting the camera. The 33ms cap means the camera moves at most one "slow frame" worth per actual stall, which is imperceptible.
  ─────────────────────────────────────────────────

  Camera Integration UB0
  ⏺ ★ Insight ─────────────────────────────────────
  Host-visible vs device-local for UBOs: The prompt suggests createDeviceLocal for the camera UBO, but that's wrong for frequently-updated per-frame data. Device-local memory
  requires a staging buffer + GPU copy every update — expensive for data that changes every frame. createHostVisible gives us a persistently-mapped CPU pointer so upload() is just
   a memcpy. Vertex/index data is written once and benefits from device-local bandwidth; UBOs are written every frame and benefit from direct CPU access.

  m_cameraSetLayout already exists: Phase 1.1 already created the descriptor set layout in createPbrPipeline() and baked it into the pipeline layout. Creating a second layout
  would be redundant and wasteful. We just need to allocate a descriptor set from it.
  ─────────────────────────────────────────────────

    include/core/Renderer.h — Added to private section:
  - CameraUBO struct (view + projection glm::mat4)
  - m_cameraUBOBuffer, m_descriptorPool, m_descriptorSet members
  - createCameraUBO(), createDescriptorPool(), createDescriptorSet() private method declarations

  src/core/Renderer.cpp — 5 targeted changes:

  ┌───────────────────────┬─────────────────────────────────────────────────────────────────┐
  │       Location        │                             Change                              │
  ├───────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ Constructor init list │ Added m_cameraUBOBuffer(ctx)                                    │
  ├───────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ init()                │ Calls 3 new methods after createPbrPipeline()                   │
  ├───────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ destroyPipeline()     │ Destroys m_descriptorPool first (implicitly frees set)          │
  ├───────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ setCameraMatrices()   │ Replaced placeholder — now does upload(&ubo, sizeof(CameraUBO)) │
  ├───────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ render()              │ Added vkCmdBindDescriptorSets before vertex buffer binding      │
  └───────────────────────┴─────────────────────────────────────────────────────────────────┘

  ★ Insight ─────────────────────────────────────
  Descriptor pool without FREE_DESCRIPTOR_SET_BIT: Pools created without VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT don't support vkFreeDescriptorSets on individual sets —
  but vkDestroyDescriptorPool frees everything at once. This is intentional: pool-level reset is cheaper than per-set tracking, and this renderer only ever needs one camera set.
  That's why destroyPipeline() just destroys the pool and nullifies m_descriptorSet — no explicit free needed.

  setCameraMatrices has no change-detection guard: The prompt suggested comparing lastViewMatrix == view before uploading. Skipped deliberately — glm::mat4 equality is 16 float
  comparisons, and the UBO upload() is a memcpy of 128 bytes. The memcpy is likely faster than the comparison on modern CPUs with cache-warm data, and the camera moves nearly
  every frame anyway.
  ─────────────────────────────────────────────────

    Phase 1.4.1 — Light UBO (Renderer.h + Renderer.cpp)
  - LightUBO struct added with lightDirection, lightIntensity, lightColor, ambientIntensity (std140-compatible layout with float padding)
  - m_lightUBOBuffer(ctx) added to constructor initializer list
  - createLightUBO() creates a host-visible buffer and sets a default sun: direction (0.5, 1.0, 0.3), white, intensity 1.0, ambient 0.2
  - setLightParameters() is public — call it any time to update the light at runtime without recreating anything

  Phase 1.4.2 — GGX BRDF Fragment Shader (pbr.frag)
  - Full Cook-Torrance BRDF: GGX NDF + Fresnel-Schlick + Smith geometry
  - Energy conservation: kD = (1 - F) * (1 - metallic) for diffuse fraction
  - Reinhard tone mapping + gamma correction (1/2.2)
  - Placeholder material (baseColor=0.8, metallic=0, roughness=0.5) — ready for texture bindings in Phase 1.6

  Phase 1.4.3 — Descriptor Wiring (Renderer.cpp)
  - createPbrPipeline(): descriptor set layout now has 2 bindings — camera (binding 0, vertex stage) + light (binding 1, fragment stage)
  - createDescriptorPool(): descriptorCount bumped from 1 → 2
  - createDescriptorSet(): writes both UBO buffer infos with vkUpdateDescriptorSets

  Phase 1.5.1 — Depth Image
  - m_depthImage(ctx) added to the constructor initializer list (same pattern as all other resource members)
  - createDepthImage() allocates a device-local D32_SFLOAT image at swapchain dimensions using the existing Image::create() API with VK_IMAGE_ASPECT_DEPTH_BIT
  - Called in init() before createPbrPipeline() so the format is ready

  Phase 1.5.2 — Render Loop and Pipeline
  - transitionImage() gained an aspectMask parameter (defaulting to COLOR_BIT) — fixes the hardcoded aspect that would have caused a validation error on depth barriers
  - Depth barrier: UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL each frame — safe because LOAD_OP_CLEAR discards old contents anyway, avoiding layout tracking state
  - depthAttachment.storeOp = DONT_CARE — depth isn't sampled after the frame, so the driver can skip writing it back to memory (bandwidth saving)
  - Pipeline depth state: depthTestEnable=TRUE, depthWriteEnable=TRUE, compareOp=GREATER for reverse-Z
  - VkPipelineRenderingCreateInfo.depthAttachmentFormat = D32_SFLOAT — required for the pipeline to be compatible with the dynamic rendering attachment

  ★ Insight ─────────────────────────────────────
  - STORE_OP_DONT_CARE for depth is a meaningful optimization: on tile-based GPUs (like Apple Silicon), if depth data never needs to be read back from memory, the tile memory can
  be discarded without a writeback — saving significant bandwidth.
  - The UNDEFINED old layout every frame is valid precisely because we clear — the Vulkan spec allows UNDEFINED→anything when you don't care about previous contents, and it avoids
   tracking "what was the depth layout last frame?"
  - Reverse-Z improves floating-point depth precision for distant geometry: since floats have more precision near 0, mapping far=0.0 concentrates precision where the eye can't
  distinguish Z-fighting anyway.
  ─────────────────────────────────────────────────

  ★ Insight ─────────────────────────────────────
  - The releaseStaging() pattern solves a key GPU/CPU sync problem: staging buffers must stay alive until the GPU finishes reading them. Destroying too early causes silent
  corruption; destroying never (only at Image::destroy()) wastes GPU memory for the whole texture lifetime.
  - SamplerDescHash uses the boost-style XOR-shift combine (seed ^ (val + 0x9e3779b9 + ...)) rather than a plain XOR, because plain XOR distributes bits poorly when field values
  are small integers — the golden ratio constant spreads the hash bits more uniformly.
  - stb_image's STBI_rgb_alpha flag (force 4 channels) is critical here: it normalizes JPEGs (which are RGB) and PNGs with alpha into a single R8G8B8A8 layout, so createFromData
  can always use one code path and one VkFormat.
  ─────────────────────────────────────────────────

  Here's a summary of everything created:

  ┌─────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │              File               │                                                         What it does                                                         │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ external/stb/stb_image.h        │ stb_image v2.30 header                                                                                                       │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ src/resource/StbImageImpl.cpp   │ Single-TU stb_image implementation (mirrors VmaImpl.cpp)                                                                     │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ include/resource/Image.h        │ Added releaseStaging() declaration                                                                                           │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ src/resource/Image.cpp          │ Implemented releaseStaging() — targeted staging buffer free without destroying the image                                     │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ include/resource/SamplerCache.h │ SamplerDesc struct + hash functor + SamplerCache class                                                                       │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ src/resource/SamplerCache.cpp   │ Hash/equality impl, vkCreateSampler on cache miss, anisotropy clamped to device limits, shutdown() iterates and destroys all │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ include/resource/Texture.h      │ Texture class — owns Image, borrows VkSampler from cache                                                                     │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ src/resource/Texture.cpp        │ loadFromFile (stbi decode → staging upload), loadFromMemory, createSolidColor (1×1 white fallback), RAII move ops            │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ CMakeLists.txt                  │ Added 3 new sources + external/stb include dir                                                                               │
  └─────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Phase 2.2 (glTF material parsing) can now call Texture::loadFromFile() and SamplerCache::getSampler() with material-specific configs. The VK_FORMAT_R8G8B8A8_SRGB vs UNORM
  distinction for normal maps is the natural next hook point in loadFromFile.