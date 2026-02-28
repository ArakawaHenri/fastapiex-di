from __future__ import annotations

import contextvars
import inspect
import logging
import os
import sys
import threading
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, TypeVar, cast, overload

from .constants import SERVICE_DEFAULT_DESTROY_MARKER, SERVICE_DEFINITION_ATTR
from .contracts import Ctor, Dtor, ServiceLifetime
from .errors import DITypeError, DIValueError, InvalidServiceDefinitionError

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
_RUNTIME_REGISTRY_BINDINGS: weakref.WeakKeyDictionary[
    AppServiceRegistry,
    _RuntimeRegistryBinding,
] = weakref.WeakKeyDictionary()
_RUNTIME_REGISTRY_BINDINGS_LOCK = threading.Lock()


def get_global_service_definitions() -> list[_ServiceDefinition]:
    """
    Return globally collected service definitions.

    Global definitions are collected at decorator execution time and can be
    reused by multiple FastAPI app instances when install_di(..., use_global_service_registry=True).
    """
    return _GLOBAL_SERVICE_REGISTRY.definitions()


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
        package_paths=normalized_paths,
        use_global_service_registry=use_global_service_registry,
    )
    with _RUNTIME_REGISTRY_BINDINGS_LOCK:
        _RUNTIME_REGISTRY_BINDINGS[registry] = binding


def unregister_runtime_registry_binding(registry: AppServiceRegistry) -> None:
    with _RUNTIME_REGISTRY_BINDINGS_LOCK:
        _RUNTIME_REGISTRY_BINDINGS.pop(registry, None)


def _runtime_registry_bindings_snapshot() -> tuple[
    tuple[tuple[AppServiceRegistry, _RuntimeRegistryBinding], ...],
    bool,
]:
    with _RUNTIME_REGISTRY_BINDINGS_LOCK:
        bindings = tuple(_RUNTIME_REGISTRY_BINDINGS.items())
    maintain_global = any(
        binding.use_global_service_registry
        for _, binding in bindings
    )
    return bindings, maintain_global


@dataclass(frozen=True)
class _RegistrationCapture:
    registry: AppServiceRegistry
    include_packages: tuple[str, ...] | None


_ACTIVE_REGISTRATION_CAPTURE: contextvars.ContextVar[_RegistrationCapture | None] = (
    contextvars.ContextVar("_svc_active_registration_capture", default=None)
)


def _is_origin_included(origin: str, include_packages: tuple[str, ...]) -> bool:
    for package in include_packages:
        if origin == package or origin.startswith(f"{package}."):
            return True
    return False


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
    for registry, binding in bindings:
        if not _definition_matches_package_paths(definition, binding.package_paths):
            continue
        registry.register(definition)
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
    if source is not None and (not isinstance(key, str) or not key):
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
    if not isinstance(key, str):
        raise DITypeError("ServiceMap() expects a non-empty key template string.")
    if not key:
        raise DIValueError("ServiceMap() requires a non-empty key template string.")

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
