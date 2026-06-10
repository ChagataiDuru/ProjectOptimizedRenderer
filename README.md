# ProjectOptimizedRenderer

ProjectOptimizedRenderer is a C++20 Vulkan renderer focused on native image quality, explicit GPU control, and a future engine-ready renderer boundary.

The repository is intentionally a renderer-first codebase rather than a generic game engine. It is meant to keep the graphics stack inspectable, measurable, and owned by the project while leaving room for a future host/editor layer outside the renderer core.

## Project goal

Short term, this project is a Vulkan research viewer for developing and validating renderer features against real scenes.

Long term, the goal is an engine-ready renderer backend with a stable API boundary. The renderer core should remain C++/Vulkan. A future Odin host/editor may exist separately, but Odin should not replace or invade the renderer core; integration should happen through a narrow C ABI using plain data structures and opaque handles.

Relevant design notes:

- [`docs/analysis/00-current-state.md`](docs/analysis/00-current-state.md)
- [`docs/analysis/01-architecture-boundaries.md`](docs/analysis/01-architecture-boundaries.md)
- [`docs/analysis/04-engine-api-direction.md`](docs/analysis/04-engine-api-direction.md)
- [`docs/decisions/ADR-0003-cpp-core-odin-host-boundary.md`](docs/decisions/ADR-0003-cpp-core-odin-host-boundary.md)

## Supported platforms

The intended active development presets are:

| Platform | Configure/build preset | Notes |
|---|---|---|
| macOS Apple Silicon | `macos-debug` | Uses MoltenVK through the Vulkan SDK or Homebrew-style paths. `debug` may remain as a compatibility alias. |
| Windows | `win-debug` | Native Vulkan SDK install expected. NVIDIA is the primary target. |
| Linux | `linux-debug` | Native Vulkan loader/runtime expected. |

Preset definitions live in [`CMakePresets.json`](CMakePresets.json). If a documented preset is not present in a local branch, update the preset file before relying on the command in automation.

## Dependencies

Required local tools:

- CMake 3.25 or newer
- Conan 2
- Ninja
- A C++20 compiler
- Vulkan SDK and a Vulkan-capable runtime/driver
- Git with submodule support

Third-party C/C++ packages are resolved with Conan. The repository also has source dependencies under `external/`, including the VMA submodule. Initialize submodules before configuring a fresh checkout:

```bash
git submodule update --init --recursive
```

## Configure and build

Run all commands from the repository root.

### 1. Install Conan dependencies

Use the same Conan output folder for all debug platform presets:

```bash
conan profile detect --force
conan install . --output-folder=build/conan-debug --build=missing -s build_type=Debug
```

### 2. Configure CMake

Choose the platform debug preset for your machine.

macOS:

```bash
cmake --preset macos-debug
```

Windows:

```bash
cmake --preset win-debug
```

Linux:

```bash
cmake --preset linux-debug
```

### 3. Build

macOS:

```bash
cmake --build --preset macos-debug --config Debug --parallel
```

Windows:

```bash
cmake --build --preset win-debug --config Debug --parallel
```

Linux:

```bash
cmake --build --preset linux-debug --config Debug --parallel
```

## Repository layout

```txt
.
├── assets/                 # Local test assets; large assets may be absent from git
├── cmake/                  # CMake helper modules
├── docs/
│   ├── analysis/           # Architecture, roadmap, and technical-debt notes
│   ├── archive/            # Historical scratch notes kept out of the root docs path
│   └── decisions/          # ADR-style project decisions
├── external/               # Third-party source dependencies/submodules
├── include/                # C++ headers
├── shaders/                # GLSL shader sources
├── src/                    # C++ implementation
├── CMakeLists.txt
├── CMakePresets.json
├── conanfile.txt
├── documentation.md        # Concise documentation index
└── README.md
```

## Current renderer baseline

Implemented or partially implemented systems include Vulkan initialization, `volk`, VMA allocation, SDL3 windowing, dynamic rendering, synchronization2, reverse-Z depth, glTF/Sponza loading, mesh and texture upload, material descriptor sets, Cook-Torrance PBR direct lighting, normal mapping, HDR offscreen rendering, multiple tone mappers, cascaded shadow maps, VSM blur, procedural/HDR sky support, ImGui panels, GPU timing, render statistics, screenshot support, and runtime model reload.

Planned research directions include SMAA, perceptual VRS, improved render-pass ownership, a clearer frame-packet model, renderer resource handles, and an eventual C ABI boundary.

## Shaders

GLSL shader sources live in [`shaders/`](shaders/). Shader compilation is integrated through [`cmake/CompileShaders.cmake`](cmake/CompileShaders.cmake).

Compiled SPIR-V files (`*.spv`) are build artifacts. Do not commit generated SPIR-V. If a run fails because shaders are missing or stale, rebuild the relevant CMake target/preset so the shader compile step can regenerate build-local artifacts.

## Assets

The renderer is typically exercised with a Sponza glTF scene at a path like:

```txt
assets/models/sponza/glTF/Sponza.gltf
```

Large models, textures, captures, and other binary assets may be intentionally absent from git. If startup fails due to a missing model or texture, install the expected local assets under `assets/` or adjust the runtime model path for your test scene.

## Runtime controls

Common viewer controls:

| Input | Action |
|---|---|
| `W`, `A`, `S`, `D` | Move camera |
| Mouse | Look around when mouse capture is enabled |
| `F1` | Toggle mouse capture / UI interaction mode |
| `F2` | Request screenshot |
| `F11` | Toggle ImGui overlay visibility |
| `Escape` | Exit application |

## Known limitations

- The project is a renderer research viewer, not a finished engine/editor.
- The documented `macos-debug`, `win-debug`, and `linux-debug` presets are the intended platform debug presets; branches may temporarily lag while preset cleanup is in progress.
- Vulkan behavior varies by platform and driver. macOS uses Vulkan portability through MoltenVK rather than a native Vulkan implementation.
- Large assets are not guaranteed to be present in a fresh checkout.
- Generated files, build directories, shader outputs, screenshots, and local captures should stay out of version control unless there is an explicit reason to archive a small example.

## Contributor notes

- Do not change renderer behavior when making documentation-only updates.
- Keep documentation current with the intended CMake/Conan workflow above.
- Prefer concise architecture notes over long scratch transcripts in root-level docs.
- Keep future Odin work separate from the C++ renderer core and route any integration through the documented C ABI direction.
