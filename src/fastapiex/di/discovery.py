from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import pkgutil
import sys
from collections.abc import Sequence

from .constants import SERVICE_DEFINITION_ATTR
from .errors import DIValueError
from .registry import AppServiceRegistry, _is_origin_included, _ServiceDefinition


def resolve_service_package_paths(
    package_names: str | Sequence[str],
) -> tuple[str, ...]:
    resolved: list[str] = []
    seen: set[str] = set()
    for package_name in _normalize_package_names(package_names):
        try:
            spec = importlib.util.find_spec(package_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            continue

        candidate_paths: list[str] = []
        if spec.submodule_search_locations:
            candidate_paths.extend(spec.submodule_search_locations)
        elif spec.origin and spec.origin not in {"built-in", "frozen"}:
            candidate_paths.append(spec.origin)

        for path in candidate_paths:
            normalized = os.path.realpath(os.path.abspath(path))
            if normalized in seen:
                continue
            seen.add(normalized)
            resolved.append(normalized)
    return tuple(resolved)


def _normalize_package_names(
    package_names: str | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(package_names, str):
        return (package_names,)
    normalized = tuple(name for name in package_names if name)
    if not normalized:
        raise DIValueError("At least one service package must be provided.")
    return normalized


def import_service_modules(
    package_names: str | Sequence[str] = "app.services",
    *,
    allow_private_modules: bool = False,
) -> list[str]:
    imported_modules: list[str] = []
    for package_name in _normalize_package_names(package_names):
        package = importlib.import_module(package_name)
        imported_modules.append(package.__name__)
        package_path = getattr(package, "__path__", None)
        if package_path is None:
            continue

        base_parts_len = len(package.__name__.split("."))
        for module_info in pkgutil.walk_packages(
            package_path,
            prefix=f"{package.__name__}.",
        ):
            module_name = module_info.name
            if not allow_private_modules:
                parts = module_name.split(".")
                if any(part.startswith("_") for part in parts[base_parts_len:]):
                    continue
            importlib.import_module(module_name)
            imported_modules.append(module_name)
    return imported_modules


def _module_service_definitions(module_name: str) -> list[_ServiceDefinition]:
    module = sys.modules.get(module_name)
    if module is None:
        return []

    definitions: list[_ServiceDefinition] = []
    for _name, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module_name:
            continue
        definition = getattr(cls, SERVICE_DEFINITION_ATTR, None)
        if isinstance(definition, _ServiceDefinition):
            definitions.append(definition)
    return definitions


def register_module_service_definitions(
    registry: AppServiceRegistry,
    module_names: Sequence[str],
    *,
    include_packages: Sequence[str] | None = None,
) -> None:
    include = tuple(include_packages) if include_packages is not None else None
    for module_name in module_names:
        for definition in _module_service_definitions(module_name):
            if include is not None and not _is_origin_included(definition.origin, include):
                continue
            registry.register(definition)
