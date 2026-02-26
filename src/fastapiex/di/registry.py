from __future__ import annotations

import asyncio
import contextvars
import importlib
import importlib.util
import inspect
import logging
import os
import pkgutil
import sys
import threading
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Literal,
    TypeVar,
    cast,
    overload,
)

from .constants import SERVICE_DEFAULT_DESTROY_MARKER, SERVICE_DEFINITION_ATTR
from .container import ServiceContainer, ServiceLifetime
from .errors import (
    CircularServiceDependencyError,
    DITypeError,
    DIValueError,
    InvalidServiceDefinitionError,
    ServiceDependencyError,
    ServiceMappingError,
    ServiceRegistryError,
)
from .types import CallableWithSignature, Ctor, Dtor

logger = logging.getLogger(__name__)

LifetimeLike = ServiceLifetime | int | Literal[
    "Singleton",
    "Transient",
    "singleton",
    "transient",
]
ExpandedDefinition = tuple[
    "_ServiceDefinition",
    str | None,
    tuple[object, ...],
    dict[str, object],
    dict[str, "RequiredDependency"],
    inspect.Signature,
]
ResolvedByInternalId = tuple[
    "_ServiceDefinition",
    dict[str, "RequiredDependency"],
    inspect.Signature,
    tuple[object, ...],
    dict[str, object],
    object,
    str | None,
]
ResolvedSpec = tuple[
    "_ServiceDefinition",
    inspect.Signature,
    tuple[object, ...],
    dict[str, object],
    tuple["_ResolvedDependency", ...],
    object,
    str | None,
]
_S = TypeVar("_S", bound=object)


@dataclass(frozen=True)
class RequiredService:
    target: str | type[object]
    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = field(default_factory=dict)
    allow_transient: bool = False

    def render_for_dict_key(self, dict_key: str) -> RequiredService:
        def _render(value: object) -> object:
            if isinstance(value, str) and "{}" in value:
                return value.replace("{}", str(dict_key))
            return value

        rendered_target = _render(self.target)
        rendered_args = tuple(_render(arg) for arg in self.args)
        rendered_kwargs = {
            key: _render(value)
            for key, value in self.kwargs.items()
        }
        if (
            rendered_target == self.target
            and rendered_args == self.args
            and rendered_kwargs == self.kwargs
        ):
            return self
        return RequiredService(
            target=cast(str | type[object], rendered_target),
            args=rendered_args,
            kwargs=rendered_kwargs,
            allow_transient=self.allow_transient,
        )


RequiredDependency = RequiredService


@dataclass(frozen=True)
class _ServiceDefinition:
    origin: str
    module_path: str | None
    service_cls: type[object]
    key_template: str | None
    lifetime: ServiceLifetime
    eager: bool
    ctor: Ctor
    dtor: Dtor
    dependencies: dict[str, RequiredDependency]
    source: Mapping[str, object] | Callable[[], Mapping[str, object]] | None = None
    exposed_type: object | None = None


@dataclass(frozen=True)
class _ResolvedDependency:
    param_name: str
    required: RequiredDependency


@dataclass(frozen=True)
class _CompiledService:
    origin: str
    internal_id: str
    key: str | None
    lifetime: ServiceLifetime
    eager: bool
    ctor: Ctor
    dtor: Dtor
    signature: inspect.Signature
    static_args: tuple[object, ...]
    static_kwargs: dict[str, object]
    dependencies: tuple[_ResolvedDependency, ...]
    service_type: object


@dataclass(frozen=True)
class RegisteredService:
    key: str | None
    origin: str
    service_type: object


@dataclass(frozen=True)
class _RuntimeRegistryBinding:
    registry: AppServiceRegistry
    package_paths: tuple[str, ...]
    use_global_service_registry: bool


class AppServiceRegistry:
    """
    App-scoped registry for resolved service definitions.

    Each FastAPI app should use its own registry instance to avoid cross-app
    state interference.
    """

    def __init__(
        self,
        definitions: Sequence[_ServiceDefinition] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._definitions_by_origin: dict[str, _ServiceDefinition] = {}
        self._definition_order: list[str] = []
        if definitions:
            for definition in definitions:
                self.register(definition)

    def register(self, definition: _ServiceDefinition) -> None:
        with self._lock:
            existing = self._definitions_by_origin.get(definition.origin)
            if existing is not None:
                return
            self._definitions_by_origin[definition.origin] = definition
            self._definition_order.append(definition.origin)

    def definitions(self) -> list[_ServiceDefinition]:
        with self._lock:
            return [
                self._definitions_by_origin[origin]
                for origin in self._definition_order
                if origin in self._definitions_by_origin
            ]


_GLOBAL_SERVICE_REGISTRY = AppServiceRegistry()
_RUNTIME_REGISTRY_BINDINGS: dict[int, _RuntimeRegistryBinding] = {}
_RUNTIME_REGISTRY_BINDINGS_LOCK = threading.Lock()


def get_global_service_definitions() -> list[_ServiceDefinition]:
    """
    Return globally collected service definitions.

    Global definitions are collected at decorator execution time and can be
    reused by multiple FastAPI app instances when install_di(..., use_global_service_registry=True).
    """
    return _GLOBAL_SERVICE_REGISTRY.definitions()


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


def register_runtime_registry_binding(
    registry: AppServiceRegistry,
    *,
    package_paths: Sequence[str],
    use_global_service_registry: bool,
) -> None:
    normalized_paths = tuple(
        os.path.realpath(os.path.abspath(path))
        for path in package_paths
    )
    binding = _RuntimeRegistryBinding(
        registry=registry,
        package_paths=normalized_paths,
        use_global_service_registry=use_global_service_registry,
    )
    with _RUNTIME_REGISTRY_BINDINGS_LOCK:
        _RUNTIME_REGISTRY_BINDINGS[id(registry)] = binding


def unregister_runtime_registry_binding(registry: AppServiceRegistry) -> None:
    with _RUNTIME_REGISTRY_BINDINGS_LOCK:
        _RUNTIME_REGISTRY_BINDINGS.pop(id(registry), None)


def _runtime_registry_bindings_snapshot() -> tuple[tuple[_RuntimeRegistryBinding, ...], bool]:
    with _RUNTIME_REGISTRY_BINDINGS_LOCK:
        bindings = tuple(_RUNTIME_REGISTRY_BINDINGS.values())
    maintain_global = any(
        binding.use_global_service_registry
        for binding in bindings
    )
    return bindings, maintain_global


@dataclass(frozen=True)
class _RegistrationCapture:
    registry: AppServiceRegistry
    include_packages: tuple[str, ...] | None


_ACTIVE_REGISTRATION_CAPTURE: contextvars.ContextVar[_RegistrationCapture | None] = (
    contextvars.ContextVar("_svc_active_registration_capture", default=None)
)


def _register_definition_into_active_registry(definition: _ServiceDefinition) -> None:
    capture = _ACTIVE_REGISTRATION_CAPTURE.get()
    if capture is None:
        return
    include = capture.include_packages
    if include is not None and not _is_origin_included(definition.origin, include):
        return
    capture.registry.register(definition)


def _definition_matches_package_paths(
    definition: _ServiceDefinition,
    package_paths: Sequence[str],
) -> bool:
    module_path = definition.module_path
    if module_path is None:
        return False
    normalized_module = os.path.realpath(os.path.abspath(module_path))
    for raw_base in package_paths:
        normalized_base = os.path.realpath(os.path.abspath(raw_base))
        if os.path.isfile(normalized_base):
            if normalized_module == normalized_base:
                return True
            continue
        try:
            if os.path.commonpath([normalized_module, normalized_base]) == normalized_base:
                return True
        except ValueError:
            continue
    return False


def _register_definition_into_runtime_registries(
    definition: _ServiceDefinition,
) -> None:
    bindings, maintain_global = _runtime_registry_bindings_snapshot()
    matched = False
    for binding in bindings:
        if not _definition_matches_package_paths(definition, binding.package_paths):
            continue
        binding.registry.register(definition)
        matched = True

    if not matched and maintain_global:
        _GLOBAL_SERVICE_REGISTRY.register(definition)


def Require(
    target: str | type[object],
    *args: object,
    allow_transient: bool = False,
    **kwargs: object,
) -> RequiredService:
    if not isinstance(target, (str, type)):
        raise DITypeError("Require() expects a service key (str) or a service type")
    return RequiredService(
        target=target,
        args=tuple(args),
        kwargs=dict(kwargs),
        allow_transient=allow_transient,
    )


def _normalize_lifetime(lifetime: LifetimeLike) -> ServiceLifetime:
    if isinstance(lifetime, ServiceLifetime):
        return lifetime
    if isinstance(lifetime, int):
        try:
            return ServiceLifetime(lifetime)
        except ValueError as exc:
            raise DIValueError(
                f"Unsupported lifetime value: {lifetime!r}. Use ServiceLifetime.SINGLETON/TRANSIENT or 0/1."
            ) from exc
    if isinstance(lifetime, str):
        normalized = lifetime.strip().lower()
        if normalized == "singleton":
            return ServiceLifetime.SINGLETON
        if normalized == "transient":
            return ServiceLifetime.TRANSIENT
        raise DIValueError(
            f"Unsupported lifetime string: {lifetime!r}. Use 'Singleton' or 'Transient'."
        )
    raise DITypeError(
        "Invalid lifetime type: "
        f"{type(lifetime)!r}. Use ServiceLifetime, int (0/1), or 'Singleton'/'Transient'."
    )


def _extract_ctors(service_cls: type[object]) -> tuple[Ctor, Dtor]:
    if inspect.isabstract(service_cls):
        raise InvalidServiceDefinitionError(
            f"{service_cls!r} is abstract; define a concrete @classmethod create()."
        )

    raw_create = inspect.getattr_static(service_cls, "create", None)
    if raw_create is None:
        raise InvalidServiceDefinitionError(f"{service_cls!r} is missing required classmethod create().")
    if not isinstance(raw_create, classmethod):
        raise InvalidServiceDefinitionError(f"{service_cls!r}.create must be defined as @classmethod.")

    ctor = getattr(service_cls, "create", None)
    if ctor is None or not callable(ctor):
        raise InvalidServiceDefinitionError(f"{service_cls!r}.create must be callable.")

    raw_destroy = inspect.getattr_static(service_cls, "destroy", None)
    if raw_destroy is None:
        return ctor, None
    if not isinstance(raw_destroy, classmethod):
        raise InvalidServiceDefinitionError(f"{service_cls!r}.destroy must be defined as @classmethod.")
    if getattr(raw_destroy, SERVICE_DEFAULT_DESTROY_MARKER, False):
        # Inherited default no-op hook from BaseService.
        return ctor, None

    dtor = getattr(service_cls, "destroy", None)
    if dtor is None or not callable(dtor):
        raise InvalidServiceDefinitionError(f"{service_cls!r}.destroy must be callable.")
    if not inspect.iscoroutinefunction(dtor):
        raise InvalidServiceDefinitionError(
            f"{service_cls!r}.destroy must be an async @classmethod."
        )
    return ctor, dtor


def _extract_dependencies(ctor: Ctor) -> dict[str, RequiredDependency]:
    deps: dict[str, RequiredDependency] = {}
    sig = inspect.signature(ctor)

    for name, parameter in sig.parameters.items():
        default = parameter.default
        if isinstance(default, RequiredService):
            deps[name] = default
    return deps


def _resolve_service_class_module_path(service_cls: type[object]) -> str | None:
    module = sys.modules.get(service_cls.__module__)
    if module is not None:
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str):
            return os.path.realpath(os.path.abspath(module_file))
    return None


def _register_service_class(
    service_cls: type[object],
    *,
    key: str | None,
    source: Mapping[str, object] | Callable[[], Mapping[str, object]] | None = None,
    lifetime: LifetimeLike = ServiceLifetime.SINGLETON,
    eager: bool = False,
    exposed_type: object | None = None,
) -> type[object]:
    ctor, dtor = _extract_ctors(service_cls)
    deps = _extract_dependencies(ctor)

    resolved_lifetime = _normalize_lifetime(lifetime)
    if eager and resolved_lifetime != ServiceLifetime.SINGLETON:
        raise DIValueError(
            f"Service '{service_cls.__name__}' cannot be eager with transient lifetime."
        )
    if source is not None and key is None:
        raise DIValueError(
            f"ServiceMap '{service_cls.__name__}' requires a non-empty key template."
        )
    definition = _ServiceDefinition(
        origin=f"{service_cls.__module__}.{service_cls.__qualname__}",
        module_path=_resolve_service_class_module_path(service_cls),
        service_cls=service_cls,
        key_template=key,
        lifetime=resolved_lifetime,
        eager=eager,
        ctor=ctor,
        dtor=dtor,
        dependencies=deps,
        source=source,
        exposed_type=exposed_type,
    )
    setattr(service_cls, SERVICE_DEFINITION_ATTR, definition)
    _register_definition_into_active_registry(definition)
    _register_definition_into_runtime_registries(definition)
    return service_cls


@overload
def Service(
    key: type[_S],
    *,
    lifetime: LifetimeLike = ServiceLifetime.SINGLETON,
    eager: bool = False,
    exposed_type: object | None = None,
) -> type[_S]:
    ...


@overload
def Service(
    key: str | None = None,
    *,
    lifetime: LifetimeLike = ServiceLifetime.SINGLETON,
    eager: bool = False,
    exposed_type: object | None = None,
) -> Callable[[type[_S]], type[_S]]:
    ...


def Service(
    key: str | type[object] | None = None,
    *,
    lifetime: LifetimeLike = ServiceLifetime.SINGLETON,
    eager: bool = False,
    exposed_type: object | None = None,
) -> Callable[[type[object]], type[object]] | type[object]:
    """
    Register a service class.

    Supported forms:
    - @Service("my_key")
    - @Service("my_key", lifetime="transient")
    - @Service
    - @Service()
    """

    def _register_with_key(
        service_cls: type[object],
        resolved_key: str | None,
    ) -> type[object]:
        return _register_service_class(
            service_cls,
            key=resolved_key,
            source=None,
            lifetime=lifetime,
            eager=eager,
            exposed_type=exposed_type,
        )

    if isinstance(key, type):
        return _register_with_key(key, None)
    if key is not None and not isinstance(key, str):
        raise DITypeError(
            "Service() expects a key string, a service class, or no positional argument."
        )

    def _decorator(service_cls: type[object]) -> type[object]:
        return _register_with_key(service_cls, key)

    return _decorator


def ServiceMap(
    key: str,
    *,
    mapping: Mapping[str, object] | Callable[[], Mapping[str, object]],
    lifetime: LifetimeLike = ServiceLifetime.SINGLETON,
    eager: bool = False,
    exposed_type: object | None = None,
) -> Callable[[type[object]], type[object]]:
    def _decorator(service_cls: type[object]) -> type[object]:
        return _register_service_class(
            service_cls,
            key=key,
            source=mapping,
            lifetime=lifetime,
            eager=eager,
            exposed_type=exposed_type,
        )

    return _decorator


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


def _render_service_key(key_template: str, dict_key: str) -> str:
    if "{}" in key_template:
        return key_template.replace("{}", dict_key)
    return f"{dict_key}_{key_template}"


def _coerce_mapping_value(
    value: object,
    *,
    signature: inspect.Signature,
    dependency_params: set[str],
) -> tuple[tuple[object, ...], dict[str, object]]:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        raw = value.model_dump()
        if not isinstance(raw, Mapping):
            raise ServiceMappingError("model_dump() must return a mapping")
        mapping_kwargs = dict(raw)
        _validate_mapping_kwargs(signature=signature, mapping_kwargs=mapping_kwargs)
        return (), mapping_kwargs

    if isinstance(value, Mapping):
        mapping_kwargs = dict(value)
        _validate_mapping_kwargs(signature=signature, mapping_kwargs=mapping_kwargs)
        return (), mapping_kwargs

    # For scalar shorthand values we only accept deterministic bindings:
    # 1) Prefer keyword-capable parameters to avoid positional drift.
    # 2) Fallback to positional-only only when it is the first positional slot.
    keyword_target: str | None = None
    for name, parameter in signature.parameters.items():
        if name in dependency_params:
            continue
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            keyword_target = name
            break
    if keyword_target is not None:
        return (), {keyword_target: value}

    positional_target: inspect.Parameter | None = None
    for name, parameter in signature.parameters.items():
        if name in dependency_params:
            continue
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional_target = parameter
            break
    if positional_target is None:
        raise ServiceMappingError("Unable to map ServiceMap value to ctor parameters")

    positional_slots = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    target_index = next(
        index
        for index, parameter in enumerate(positional_slots)
        if parameter.name == positional_target.name
    )
    if target_index != 0:
        raise ServiceMappingError(
            "Scalar ServiceMap value cannot bind positional-only parameter "
            f"'{positional_target.name}' because earlier positional parameters exist. "
            "Use mapping={...} for explicit binding."
        )
    return (value,), {}


def _validate_mapping_kwargs(
    *,
    signature: inspect.Signature,
    mapping_kwargs: Mapping[str, object],
) -> None:
    positional_only = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY
    }
    for name in mapping_kwargs:
        if name in positional_only:
            raise ServiceMappingError(
                "ServiceMap mapping cannot pass positional-only "
                f"parameter '{name}' by key."
            )


def _resolve_source(
    source: Mapping[str, object] | Callable[[], Mapping[str, object]],
) -> Mapping[str, object]:
    resolved = source() if callable(source) else source
    if not isinstance(resolved, Mapping):
        raise ServiceMappingError("ServiceMap source must resolve to a mapping")
    return resolved


def _is_origin_included(origin: str, include_packages: tuple[str, ...]) -> bool:
    for package in include_packages:
        if origin == package or origin.startswith(f"{package}."):
            return True
    return False


@contextmanager
def capture_service_registrations(
    registry: AppServiceRegistry,
    *,
    include_packages: Sequence[str] | None = None,
) -> Iterator[None]:
    include = tuple(include_packages) if include_packages is not None else None
    token = _ACTIVE_REGISTRATION_CAPTURE.set(
        _RegistrationCapture(
            registry=registry,
            include_packages=include,
        )
    )
    try:
        yield
    finally:
        _ACTIVE_REGISTRATION_CAPTURE.reset(token)


def _expand_definitions(
    *,
    registry: AppServiceRegistry,
    include_packages: tuple[str, ...] | None = None,
) -> list[ExpandedDefinition]:
    expanded: list[ExpandedDefinition] = []

    for definition in registry.definitions():
        if include_packages is not None and not _is_origin_included(
            definition.origin,
            include_packages,
        ):
            continue
        signature = inspect.signature(definition.ctor)
        dependency_params = set(definition.dependencies.keys())

        if definition.source is None:
            expanded.append(
                (
                    definition,
                    definition.key_template,
                    (),
                    {},
                    definition.dependencies,
                    signature,
                )
            )
            continue

        if definition.key_template is None:
            raise ServiceRegistryError(
                f"Service definition '{definition.origin}' has dict source but no key template."
            )

        source_mapping = _resolve_source(definition.source)
        for raw_dict_key, raw_value in source_mapping.items():
            dict_key = str(raw_dict_key)
            service_key = _render_service_key(definition.key_template, dict_key)
            static_args, static_kwargs = _coerce_mapping_value(
                raw_value,
                signature=signature,
                dependency_params=dependency_params,
            )
            static_bound: inspect.BoundArguments
            try:
                static_bound = signature.bind_partial(*static_args, **static_kwargs)
            except TypeError as exc:
                raise ServiceMappingError(
                    f"Invalid ServiceMap mapping value for key '{service_key}' in "
                    f"'{definition.origin}': {exc}"
                ) from exc
            overridden_params = set(static_bound.arguments.keys())
            deps = {
                param_name: dep.render_for_dict_key(dict_key)
                for param_name, dep in definition.dependencies.items()
                if param_name not in overridden_params
            }
            expanded.append(
                (
                    definition,
                    service_key,
                    static_args,
                    static_kwargs,
                    deps,
                    signature,
                )
            )

    return expanded


def _resolve_dependency_targets(
    compiled: list[ExpandedDefinition],
) -> list[_CompiledService]:
    by_internal_id: dict[str, ResolvedByInternalId] = {}
    public_key_index: dict[str, str] = {}
    type_index: dict[object, list[str]] = defaultdict(list)

    anon_seq = 0
    for definition, key, static_args, static_kwargs, deps, signature in compiled:
        if key is not None:
            existing_internal_id = public_key_index.get(key)
            if existing_internal_id is not None:
                first = by_internal_id[existing_internal_id][0]
                raise ServiceRegistryError(
                    f"Duplicate service key '{key}' from {first.origin} and {definition.origin}."
                )
            internal_id = key
            public_key_index[key] = internal_id
        else:
            anon_seq += 1
            internal_id = f"anonymous:{definition.origin}:{anon_seq}"

        service_type = definition.exposed_type
        if service_type is None:
            service_type = ServiceContainer._infer_service_type(definition.ctor)
        if service_type is None:
            service_type = definition.service_cls
        by_internal_id[internal_id] = (
            definition,
            deps,
            signature,
            static_args,
            static_kwargs,
            service_type,
            key,
        )
        type_index[service_type].append(internal_id)

    dependency_map: dict[str, set[str]] = {}
    resolved: dict[str, ResolvedSpec] = {}

    def _service_label(internal_id: str) -> str:
        definition, _, _, _, _, _, registration_key = by_internal_id[internal_id]
        return registration_key or definition.origin

    for internal_id, (
        definition,
        deps,
        signature,
        static_args,
        static_kwargs,
        _service_type,
        registration_key,
    ) in by_internal_id.items():
        service_label = registration_key or definition.origin
        dep_internal_ids: set[str] = set()
        resolved_deps: list[_ResolvedDependency] = []

        for param_name, dep in deps.items():
            if isinstance(dep.target, str):
                dep_key = dep.target
                dep_internal_id = public_key_index.get(dep_key)
                if dep_internal_id is None:
                    raise ServiceDependencyError(
                        f"Service '{service_label}' depends on '{dep_key}', but that service is not registered."
                    )
                resolved_deps.append(
                    _ResolvedDependency(
                        param_name=param_name,
                        required=dep,
                    )
                )
            else:
                candidate_plan_keys = type_index.get(dep.target, [])
                if not candidate_plan_keys:
                    raise ServiceDependencyError(
                        f"Service '{service_label}' depends on type {dep.target!r}, but no service provides that type."
                    )
                if len(candidate_plan_keys) > 1:
                    candidate_labels = [_service_label(candidate_id) for candidate_id in candidate_plan_keys]
                    raise ServiceDependencyError(
                        "Service "
                        f"'{service_label}' depends on type {dep.target!r}, "
                        f"but multiple services provide it: {candidate_labels}."
                    )
                dep_internal_id = candidate_plan_keys[0]
                resolved_deps.append(
                    _ResolvedDependency(
                        param_name=param_name,
                        required=dep,
                    )
                )

            target_definition = by_internal_id[dep_internal_id][0]
            target_key = by_internal_id[dep_internal_id][6]
            target_label = target_key or target_definition.origin
            if (
                definition.lifetime == ServiceLifetime.SINGLETON
                and target_definition.lifetime == ServiceLifetime.TRANSIENT
                and not dep.allow_transient
            ):
                raise ServiceDependencyError(
                    f"Singleton service '{service_label}' depends on transient service '{target_label}'. "
                    "Use Require(..., allow_transient=True) only if this is intentional."
                )

            dep_internal_ids.add(dep_internal_id)

        dependency_map[internal_id] = dep_internal_ids
        resolved[internal_id] = (
            definition,
            signature,
            static_args,
            static_kwargs,
            tuple(resolved_deps),
            by_internal_id[internal_id][5],
            registration_key,
        )

    registration_order = _topological_registration_order(dependency_map)

    result: list[_CompiledService] = []
    for internal_id in registration_order:
        (
            definition,
            signature,
            static_args,
            static_kwargs,
            resolved_dependencies,
            service_type,
            registration_key,
        ) = resolved[internal_id]
        result.append(
            _CompiledService(
                origin=definition.origin,
                internal_id=internal_id,
                key=registration_key,
                lifetime=definition.lifetime,
                eager=definition.eager,
                ctor=definition.ctor,
                dtor=definition.dtor,
                signature=signature,
                static_args=static_args,
                static_kwargs=static_kwargs,
                dependencies=resolved_dependencies,
                service_type=service_type,
            )
        )

    return result


def _topological_registration_order(dependency_map: dict[str, set[str]]) -> list[str]:
    indegree = dict.fromkeys(dependency_map, 0)
    outgoing: dict[str, set[str]] = defaultdict(set)

    for node, deps in dependency_map.items():
        indegree[node] = len(deps)
        for dep in deps:
            outgoing[dep].add(node)

    queue = deque(node for node, degree in indegree.items() if degree == 0)
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for successor in outgoing.get(node, set()):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)

    if len(order) != len(dependency_map):
        cycle = _detect_cycle(dependency_map)
        if cycle:
            cycle_path = " -> ".join(cycle)
            raise CircularServiceDependencyError(
                f"Detected circular service dependency: {cycle_path}"
            )
        raise CircularServiceDependencyError("Detected circular service dependency")

    return order


def _detect_cycle(dependency_map: dict[str, set[str]]) -> list[str] | None:
    UNVISITED = 0
    VISITING = 1
    VISITED = 2

    state: dict[str, int] = dict.fromkeys(dependency_map, UNVISITED)
    stack: list[str] = []

    def dfs(node: str) -> list[str] | None:
        state[node] = VISITING
        stack.append(node)

        for dep in dependency_map.get(node, set()):
            dep_state = state.get(dep, UNVISITED)
            if dep_state == UNVISITED:
                cycle = dfs(dep)
                if cycle is not None:
                    return cycle
            elif dep_state == VISITING:
                start = stack.index(dep)
                return stack[start:] + [dep]

        stack.pop()
        state[node] = VISITED
        return None

    for node in dependency_map:
        if state[node] == UNVISITED:
            cycle = dfs(node)
            if cycle is not None:
                return cycle

    return None


def build_service_plan(
    *,
    registry: AppServiceRegistry,
    include_packages: Sequence[str] | None = None,
) -> list[_CompiledService]:
    include = tuple(include_packages) if include_packages is not None else None
    expanded = _expand_definitions(registry=registry, include_packages=include)
    compiled = _resolve_dependency_targets(expanded)
    _validate_singleton_specs(compiled)
    return compiled


def _validate_singleton_specs(specs: Sequence[_CompiledService]) -> None:
    for spec in specs:
        if spec.lifetime != ServiceLifetime.SINGLETON:
            continue
        dependency_defaults = {
            dependency.param_name: dependency.required
            for dependency in spec.dependencies
        }
        service_label = spec.key or spec.origin
        try:
            _compose_ctor_call_arguments(
                signature=spec.signature,
                dependency_defaults=dependency_defaults,
                static_args=spec.static_args,
                static_kwargs=spec.static_kwargs,
                runtime_args=(),
                runtime_kwargs={},
                service_label=service_label,
            )
        except TypeError as exc:
            raise InvalidServiceDefinitionError(
                f"Invalid singleton service '{service_label}': {exc}"
            ) from exc


def _compose_ctor_call_arguments(
    *,
    signature: inspect.Signature,
    dependency_defaults: Mapping[str, RequiredDependency],
    static_args: tuple[object, ...],
    static_kwargs: Mapping[str, object],
    runtime_args: tuple[object, ...],
    runtime_kwargs: Mapping[str, object],
    service_label: str,
) -> tuple[tuple[object, ...], dict[str, object]]:
    parameter_map = signature.parameters
    positional_params: list[inspect.Parameter] = []
    kw_only_params: list[inspect.Parameter] = []
    has_var_keyword = False
    has_var_positional = False

    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_params.append(parameter)
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            kw_only_params.append(parameter)
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True

    base_bound = signature.bind_partial(*static_args, **dict(static_kwargs))

    values: dict[str, object] = {}
    var_positional_values: list[object] = []
    var_keyword_values: dict[str, object] = {}
    for name, value in base_bound.arguments.items():
        parameter = parameter_map[name]
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            var_positional_values.extend(cast(tuple[object, ...], value))
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            var_keyword_values.update(cast(dict[str, object], value))
            continue
        values[name] = value

    unresolved_required_positional = [
        parameter.name
        for parameter in positional_params
        if parameter.name not in values
        and parameter.name not in dependency_defaults
        and parameter.default is inspect.Parameter.empty
    ]

    runtime_positional_values = list(runtime_args)
    runtime_positional_assigned: set[str] = set()
    for param_name in unresolved_required_positional:
        if not runtime_positional_values:
            break
        values[param_name] = runtime_positional_values.pop(0)
        runtime_positional_assigned.add(param_name)

    if runtime_positional_values:
        if not has_var_positional:
            raise InvalidServiceDefinitionError(
                f"Service '{service_label}' received too many positional arguments."
            )
        var_positional_values.extend(runtime_positional_values)

    for key, value in runtime_kwargs.items():
        runtime_param = parameter_map.get(key)
        if runtime_param is None:
            if not has_var_keyword:
                raise InvalidServiceDefinitionError(
                    f"Service '{service_label}' got an unexpected keyword argument '{key}'."
                )
            var_keyword_values[key] = value
            continue
        if runtime_param.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise InvalidServiceDefinitionError(
                f"Service '{service_label}' positional-only parameter '{key}' passed as keyword."
            )
        if runtime_param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise InvalidServiceDefinitionError(
                f"Service '{service_label}' invalid keyword argument for *{key}."
            )
        if key in runtime_positional_assigned:
            raise InvalidServiceDefinitionError(
                f"Service '{service_label}' got multiple values for argument '{key}'."
            )
        if runtime_param.kind == inspect.Parameter.VAR_KEYWORD:
            var_keyword_values[key] = value
            continue
        values[key] = value

    for parameter in (*positional_params, *kw_only_params):
        if parameter.name in values:
            continue
        dependency_default = dependency_defaults.get(parameter.name)
        if dependency_default is not None:
            values[parameter.name] = dependency_default
            continue
        if parameter.default is not inspect.Parameter.empty:
            values[parameter.name] = parameter.default
            continue
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            raise InvalidServiceDefinitionError(
                f"Service '{service_label}' missing required keyword-only argument '{parameter.name}'."
            )
        raise InvalidServiceDefinitionError(
            f"Service '{service_label}' missing required argument '{parameter.name}'."
        )

    final_args = [values[parameter.name] for parameter in positional_params]
    if has_var_positional:
        final_args.extend(var_positional_values)
    final_kwargs = {
        parameter.name: values[parameter.name]
        for parameter in kw_only_params
        if parameter.name in values
    }
    if has_var_keyword:
        final_kwargs.update(var_keyword_values)

    return tuple(final_args), final_kwargs


async def _resolve_required_service_value(
    *,
    container: ServiceContainer,
    request: object,
    value: object,
) -> object:
    if not isinstance(value, RequiredService):
        return value

    dep_kwargs = dict(value.kwargs)
    if request is not None:
        dep_kwargs[container.request_kwarg_name()] = request
    if isinstance(value.target, str):
        return await container.aget_by_key(value.target, *value.args, **dep_kwargs)
    return await container.aget_by_type(value.target, *value.args, **dep_kwargs)


def _make_bound_ctor(container: ServiceContainer, spec: _CompiledService) -> Ctor:
    ctor = spec.ctor
    signature = spec.signature.replace(return_annotation=spec.service_type)
    dependency_defaults = {
        dependency.param_name: dependency.required
        for dependency in spec.dependencies
    }
    service_label = spec.key or spec.origin

    async def _bound_ctor(*args: object, **kwargs: object) -> object:
        final_args, final_kwargs = _compose_ctor_call_arguments(
            signature=signature,
            dependency_defaults=dependency_defaults,
            static_args=spec.static_args,
            static_kwargs=spec.static_kwargs,
            runtime_args=args,
            runtime_kwargs=kwargs,
            service_label=service_label,
        )
        request = container.current_request()
        resolved_args = [
            await _resolve_required_service_value(
                container=container,
                request=request,
                value=value,
            )
            for value in final_args
        ]
        resolved_kwargs = {
            key: await _resolve_required_service_value(
                container=container,
                request=request,
                value=value,
            )
            for key, value in final_kwargs.items()
        }

        if inspect.iscoroutinefunction(ctor):
            return await ctor(*resolved_args, **resolved_kwargs)
        if inspect.isasyncgenfunction(ctor):
            return ctor(*resolved_args, **resolved_kwargs)

        result = await asyncio.to_thread(ctor, *resolved_args, **resolved_kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    name_suffix = spec.key or spec.origin
    safe_suffix = "".join(ch if ch.isalnum() else "_" for ch in name_suffix).strip("_") or "service"
    _bound_ctor.__name__ = f"autoreg_ctor_{safe_suffix}"
    _bound_ctor.__qualname__ = _bound_ctor.__name__
    bound_ctor_with_signature = cast(CallableWithSignature, _bound_ctor)
    bound_ctor_with_signature.__signature__ = signature
    annotations = dict(getattr(ctor, "__annotations__", {}))
    annotations["return"] = spec.service_type
    _bound_ctor.__annotations__ = annotations
    return _bound_ctor


def _select_unregistered_specs(
    plan: Sequence[_CompiledService],
    registered_origins: set[str],
) -> list[_CompiledService]:
    return [spec for spec in plan if spec.origin not in registered_origins]


def _registered_service_from_spec(spec: _CompiledService) -> RegisteredService:
    return RegisteredService(
        key=spec.key,
        origin=spec.origin,
        service_type=spec.service_type,
    )


async def _resolve_eager_service(
    container: ServiceContainer,
    spec: _CompiledService,
    *,
    eager_init_timeout_sec: float | None,
) -> None:
    async def _resolve() -> object:
        if spec.key is not None:
            return await container.aget_by_key(spec.key)
        return await container.aget_by_type(spec.service_type)

    if eager_init_timeout_sec is None:
        await _resolve()
        return
    await asyncio.wait_for(_resolve(), timeout=eager_init_timeout_sec)


async def _register_compiled_specs(
    container: ServiceContainer,
    specs: Sequence[_CompiledService],
    *,
    eager_init_timeout_sec: float | None = None,
    registered_origins: set[str] | None = None,
) -> list[RegisteredService]:
    registered_services: list[RegisteredService] = []
    for spec in specs:
        await container.register(
            spec.key,
            spec.lifetime,
            _make_bound_ctor(container, spec),
            spec.dtor,
        )
        if spec.eager:
            await _resolve_eager_service(
                container,
                spec,
                eager_init_timeout_sec=eager_init_timeout_sec,
            )
        registered_services.append(_registered_service_from_spec(spec))
        if registered_origins is not None:
            registered_origins.add(spec.origin)
    return registered_services


def _merge_global_definitions_into_registry(
    registry: AppServiceRegistry,
    *,
    include_global_registry: bool,
) -> None:
    if not include_global_registry:
        return
    for definition in get_global_service_definitions():
        registry.register(definition)


async def register_services_from_registry(
    container: ServiceContainer,
    *,
    registry: AppServiceRegistry,
    include_packages: Sequence[str] | None = None,
    eager_init_timeout_sec: float | None = None,
) -> list[RegisteredService]:
    plan = build_service_plan(
        registry=registry,
        include_packages=include_packages,
    )
    registered_services = await _register_compiled_specs(
        container,
        plan,
        eager_init_timeout_sec=eager_init_timeout_sec,
    )
    logger.debug("[LIFESPAN] Auto-registered services count=%s", len(registered_services))
    return registered_services


async def refresh_services_for_container(
    container: ServiceContainer,
    *,
    registry: AppServiceRegistry,
    registered_origins: set[str],
    include_global_registry: bool,
    eager_init_timeout_sec: float | None = None,
) -> list[RegisteredService]:
    """
    Refresh container registrations from an app registry and register only new ones.

    When include_global_registry is True, global definitions are merged into the app
    registry before compiling the incremental registration plan.
    """
    _merge_global_definitions_into_registry(
        registry,
        include_global_registry=include_global_registry,
    )
    plan = build_service_plan(registry=registry)
    new_specs = _select_unregistered_specs(plan, registered_origins)
    if not new_specs:
        return []

    registered_services = await _register_compiled_specs(
        container,
        new_specs,
        eager_init_timeout_sec=eager_init_timeout_sec,
        registered_origins=registered_origins,
    )
    logger.debug(
        "[LIFESPAN] Runtime refresh registered services count=%s",
        len(registered_services),
    )
    return registered_services
