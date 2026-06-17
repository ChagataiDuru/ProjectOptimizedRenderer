# Developer Tooling

This repository now includes a small developer experience toolkit for environment validation, repeatable local builds, and standalone shader compilation.

The toolkit is intentionally tooling-only. It does not change renderer behavior, Vulkan runtime code, or shader source content.

## Scope

- macOS and Windows wrapper scripts are first-class.
- Linux can still use the Python doctor and shader tooling, but this toolkit does not add Linux wrapper scripts yet.
- The wrapper scripts use the existing checked-in preset names from [`CMakePresets.json`](../CMakePresets.json).

## Quickstart

### macOS

```bash
./scripts/doctor-macos.sh
./scripts/bootstrap-macos.sh Debug
./scripts/build-macos.sh Debug
./scripts/run-macos.sh Debug
```

Common build types:

- `Debug` -> `debug`
- `RelWithDebInfo` -> `relwithdebinfo`
- `Release` -> `release`

### Windows

```powershell
.\scripts\doctor-windows.ps1
.\scripts\bootstrap-windows.ps1 Debug
.\scripts\build-windows.ps1 Debug
.\scripts\run-windows.ps1 Debug
```

Common build types:

- `Debug` -> `win-debug`
- `RelWithDebInfo` -> `win-relwithdebinfo`
- `Release` -> `win-release`

The Windows wrappers automatically try to import the Visual Studio developer environment if `cl.exe` is not already available. They use `vswhere` to find the latest installation with the C++ toolchain.

## Environment Doctor

The environment doctor validates the local machine setup and exits non-zero when a required prerequisite is missing.

Entry points:

- `./scripts/doctor-macos.sh`
- `.\scripts\doctor-windows.ps1`
- `python tools/doctor/doctor.py`

Required checks include:

- CMake `>= 3.25`
- Conan 2
- Ninja
- Python 3
- writable `build/`
- VMA submodule presence
- platform and architecture detection
- Vulkan runtime availability
- `glslc` or `glslangValidator`
- default Sponza asset at `assets/source/Sponza.gltf`

Warning-only checks include:

- missing HDR panorama assets under `assets/`
- Linux host detection for the wrapper scripts
- unset `VULKAN_SDK` when the runtime is still otherwise discoverable

Typical output looks like:

```txt
ProjectOptimizedRenderer Environment Check

[OK] CMake: 3.31.6 (/opt/homebrew/bin/cmake)
[OK] Conan 2: 2.16.1 (/opt/homebrew/bin/conan)
[OK] Ninja: 1.12.1 (/opt/homebrew/bin/ninja)
[OK] Python 3: 3.11.9 (/opt/homebrew/bin/python3)
[OK] Architecture: Apple Silicon detected (arm64)
[OK] Vulkan runtime: MoltenVK found at /opt/homebrew/lib/libMoltenVK.dylib
[OK] Shader compiler: glslangValidator (/opt/homebrew/bin/glslangValidator)
[WARN] HDR panorama assets: No .hdr files were found under assets/.
```

## Build Wrapper Scripts

The wrapper scripts standardize the local workflow without replacing CMake or Conan.

### Bootstrap

`bootstrap-*` scripts do three things:

1. sync submodules
2. run `conan profile detect --force`
3. run `conan install . --output-folder=build/conan --build=missing -s build_type=<BuildType>`

### Build

`build-*` scripts:

- normalize the build type
- map it to the correct preset
- rerun bootstrap if `build/conan/conan_toolchain.cmake` is missing
- rerun bootstrap if the generated `CMakeUserPresets.json` is stale
- configure with the resolved preset and the explicit Conan toolchain path
- build with the matching preset

Example:

```bash
./scripts/build-macos.sh Release
```

```powershell
.\scripts\build-windows.ps1 RelWithDebInfo
```

### Run

`run-*` scripts:

- build automatically if the renderer executable is missing
- activate the Conan run environment when it exists
- launch `ProjectOptimizedRenderer`

### Clean

Default clean removes generated preset build folders only:

```bash
./scripts/clean-macos.sh
```

```powershell
.\scripts\clean-windows.ps1
```

Extended clean also removes `build/conan` and the Conan-generated `CMakeUserPresets.json`:

```bash
./scripts/clean-macos.sh --all
```

```powershell
.\scripts\clean-windows.ps1 -All
```

## Shader Tooling

Standalone shader tooling lives under [`tools/shaders/`](../tools/shaders/).

Main entry point:

```bash
python tools/shaders/compile_shaders.py
```

Supported options:

- `--clean`
- `--validate-only`
- `--compiler auto`
- `--compiler glslc`
- `--compiler glslangValidator`
- `--preset <name>`
- `--build-dir <path>`

Examples:

```bash
python tools/shaders/compile_shaders.py
python tools/shaders/compile_shaders.py --clean
python tools/shaders/compile_shaders.py --validate-only
python tools/shaders/compile_shaders.py --compiler glslc --preset debug
python tools/shaders/validate_shaders.py --preset debug
```

Behavior:

- discovers GLSL sources recursively under `shaders/`
- supports `.vert`, `.frag`, and `.comp`
- writes SPIR-V outputs under `build/<preset>/shaders/` by default
- recompiles only missing or stale outputs unless `--clean` is used
- reports missing or stale outputs as failures in validation mode
- warns about orphan `.spv` files that do not correspond to a current source shader

The standalone compiler uses the same preference order as the CMake helper:

1. `glslc`
2. `glslangValidator`

It also searches the same general location classes: Conan-derived hints when available, platform-specific SDK or Homebrew locations, then `PATH`.

## Manual Configure Note

The wrapper scripts always pass the Conan-generated toolchain explicitly:

```txt
build/conan/conan_toolchain.cmake
```

If you configure manually, pass that toolchain path yourself. Do not rely on a stale local `CMakeUserPresets.json` from an older Conan output folder.
