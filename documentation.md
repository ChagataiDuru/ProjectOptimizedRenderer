# Documentation index

This file is the root documentation index for ProjectOptimizedRenderer. It intentionally stays concise so long-context AI agents and future contributors can find the right source of truth quickly.

## Start here

- [`README.md`](README.md) — project goal, supported platforms, dependencies, build commands, shaders, assets, runtime controls, and known limitations.
- [`AGENTS.md`](AGENTS.md) — AI-agent workflow, exact debug commands, collaboration rules, and architecture guardrails.

## Architecture and planning

- [`docs/analysis/00-current-state.md`](docs/analysis/00-current-state.md) — current renderer state and practical next steps.
- [`docs/analysis/01-architecture-boundaries.md`](docs/analysis/01-architecture-boundaries.md) — renderer/application boundaries and ownership concerns.
- [`docs/analysis/02-renderer-technical-debt.md`](docs/analysis/02-renderer-technical-debt.md) — known technical debt.
- [`docs/analysis/03-feature-roadmap.md`](docs/analysis/03-feature-roadmap.md) — feature roadmap and research direction.
- [`docs/analysis/04-engine-api-direction.md`](docs/analysis/04-engine-api-direction.md) — future engine-facing renderer API direction.

## Decisions

- [`docs/decisions/ADR-0003-cpp-core-odin-host-boundary.md`](docs/decisions/ADR-0003-cpp-core-odin-host-boundary.md) — decision record for keeping the renderer core in C++ and treating Odin as a future separate host/editor option.

## Archive

- [`docs/archive/documentation.md`](docs/archive/documentation.md) — historical scratch implementation notes that used to live at the repository root. These notes are retained for context but are not the current build or architecture guide.

## Current debug command summary

```bash
conan profile detect --force
conan install . --output-folder=build/conan-debug --build=missing -s build_type=Debug
cmake --preset <macos-debug|win-debug|linux-debug>
cmake --build --preset <macos-debug|win-debug|linux-debug> --config Debug --parallel
```
