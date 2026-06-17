#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.common.project_tooling import (  # noqa: E402
    SUPPORTED_SHADER_EXTENSIONS,
    ToolMatch,
    build_dir_for_preset,
    default_preset,
    find_repo_root,
    find_shader_compiler,
    platform_key,
)


@dataclass(frozen=True)
class ShaderTask:
    source: Path
    output: Path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile or validate standalone GLSL shaders.")
    parser.add_argument("--clean", action="store_true", help="Remove generated shader outputs before compiling.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate that expected SPIR-V outputs exist and are up to date.",
    )
    parser.add_argument(
        "--compiler",
        choices=["auto", "glslc", "glslangValidator"],
        default="auto",
        help="Shader compiler to use when compilation is required.",
    )
    parser.add_argument(
        "--preset",
        help="Preset whose build/<preset>/shaders directory should receive outputs. "
        "Defaults to the platform's primary preset.",
    )
    parser.add_argument(
        "--build-dir",
        help="Explicit build directory to use instead of build/<preset>.",
    )
    return parser.parse_args(argv)


def resolve_build_dir(repo_root: Path, preset: Optional[str], build_dir: Optional[str]) -> Path:
    if build_dir:
        return Path(build_dir).expanduser().resolve()
    selected_preset = preset or default_preset(platform_key())
    return build_dir_for_preset(repo_root, selected_preset)


def discover_shader_tasks(repo_root: Path, output_dir: Path) -> list[ShaderTask]:
    source_root = repo_root / "shaders"
    tasks: list[ShaderTask] = []
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or source.suffix not in SUPPORTED_SHADER_EXTENSIONS:
            continue
        relative = source.relative_to(source_root)
        output = output_dir / relative.parent / f"{relative.name}.spv"
        tasks.append(ShaderTask(source=source, output=output))
    return tasks


def display_path(path: Path, repo_root: Path) -> Path:
    try:
        return path.relative_to(repo_root)
    except ValueError:
        return path


def output_is_current(task: ShaderTask) -> bool:
    if not task.output.is_file():
        return False
    return task.output.stat().st_mtime >= task.source.stat().st_mtime


def stale_outputs(tasks: list[ShaderTask]) -> list[ShaderTask]:
    return [task for task in tasks if not output_is_current(task)]


def orphan_outputs(output_dir: Path, tasks: list[ShaderTask]) -> list[Path]:
    expected = {task.output.resolve() for task in tasks}
    if not output_dir.is_dir():
        return []
    result: list[Path] = []
    for output in sorted(output_dir.rglob("*.spv")):
        if output.resolve() not in expected:
            result.append(output)
    return result


def build_compile_command(compiler: ToolMatch, task: ShaderTask) -> list[str]:
    if compiler.name == "glslc":
        return [
            str(compiler.path),
            "--target-env=vulkan1.4",
            "-O",
            "-o",
            str(task.output),
            str(task.source),
        ]

    return [
        str(compiler.path),
        "--target-env",
        "vulkan1.4",
        "--spirv-val",
        "-V",
        "-o",
        str(task.output),
        str(task.source),
    ]


def print_failure(task: ShaderTask, output: str) -> None:
    print(f"[FAIL] {display_path(task.source, REPO_ROOT)}")
    if output.strip():
        print(output.rstrip())


def compile_tasks(tasks: list[ShaderTask], compiler: ToolMatch) -> int:
    failed = 0
    for task in tasks:
        task.output.parent.mkdir(parents=True, exist_ok=True)
        command = build_compile_command(compiler, task)
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            failed += 1
            print_failure(task, completed.stdout)
            continue
        print(f"[OK] Compiled {task.source.relative_to(REPO_ROOT)}")
    return failed


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.clean and args.validate_only:
        print("Error: --clean and --validate-only cannot be used together.", file=sys.stderr)
        return 2

    repo_root = find_repo_root()
    build_dir = resolve_build_dir(repo_root, args.preset, args.build_dir)
    output_dir = build_dir / "shaders"
    tasks = discover_shader_tasks(repo_root, output_dir)

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    stale = stale_outputs(tasks)
    orphans = orphan_outputs(output_dir, tasks)

    if args.validate_only:
        for task in stale:
            relative_output = display_path(task.output, repo_root)
            if task.output.exists():
                print(f"[FAIL] Stale output: {relative_output} is older than {display_path(task.source, repo_root)}")
            else:
                print(f"[FAIL] Missing output: {relative_output}")
        if orphans:
            for orphan in orphans:
                print(f"[WARN] Orphan output: {display_path(orphan, repo_root)}")

        unchanged = len(tasks) - len(stale)
        print()
        print(f"Shaders discovered: {len(tasks)}")
        print("Compiled: 0")
        print(f"Unchanged: {unchanged}")
        print(f"Failed: {len(stale)}")
        print(f"Output directory: {output_dir}")
        return 0 if not stale else 1

    compiler = find_shader_compiler(repo_root, args.compiler)
    if not compiler and stale:
        print(
            "Error: neither glslc nor glslangValidator is available. "
            "Install the Vulkan SDK or glslang, or choose a different --compiler.",
            file=sys.stderr,
        )
        return 1

    if compiler:
        print(f"Compiler: {compiler.name} ({compiler.path})")
    if orphans:
        for orphan in orphans:
            print(f"[WARN] Orphan output retained: {display_path(orphan, repo_root)}")

    failed = compile_tasks(stale, compiler) if stale and compiler else 0

    validated_failures = 0
    for task in tasks:
        if not output_is_current(task):
            validated_failures += 1
            relative_output = display_path(task.output, repo_root)
            if task.output.exists():
                print(f"[FAIL] Stale output after compile: {relative_output}")
            else:
                print(f"[FAIL] Missing output after compile: {relative_output}")

    print()
    print(f"Shaders discovered: {len(tasks)}")
    print(f"Compiled: {len(stale) - failed if stale else 0}")
    print(f"Unchanged: {len(tasks) - len(stale)}")
    print(f"Failed: {failed + validated_failures}")
    print(f"Output directory: {output_dir}")

    return 0 if failed == 0 and validated_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
