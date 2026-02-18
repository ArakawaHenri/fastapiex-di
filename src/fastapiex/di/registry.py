from __future__ import annotations

import asyncio
import contextvars
import importlib
import inspect
import logging
import pkgutil
import sys
import threading
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar, cast, overload

from .container import ServiceContainer, ServiceLifetime

logger = logging.getLogger(__name__)

Ctor = Callable[..., object]
Dtor = Callable[[object], None | Awaitable[None]] | None
_DEFAULT_DESTROY_MARKER = "__service_default_destroy_noop__"
_SERVICE_DEFINITION_ATTR = "__fastapi_di_definition__"
LifetimeLike = ServiceLifetime | int | Literal[
    "Singleton",
    "Transient",
    "singleton",
    "transient",
]
ExpandedDefinition = tuple[
    "_ServiceDefinition",
    str | None,
    dict[str, object],
    dict[str, "RequiredService"],
    inspect.Signature,
]
ResolvedByInternalId = tuple[
    "_ServiceDefinition",
    dict[str, "RequiredService"],
    inspect.Signature,
    dict[str, object],
    object,
    str | None,
]
ResolvedSpec = tuple[
    "_ServiceDefinition",
    inspect.Signature,
    dict[str, object],
    tuple["_ResolvedDependency", ...],
    object,
    str | None,
]
_S = TypeVar("_S", bound=object)


class _CallableWithSignature(Protocol):
    __signature__: inspect.Signature


@dataclass(frozen=True)
class RequiredService:
    target: str | type[object]
    allow_transient: bool = False

    def render_for_dict_key(self, dict_key: str) -> RequiredService:
        if isinstance(self.target, str) and "{}" in self.target:
            return RequiredService(
                target=self.target.replace("{}", str(dict_key)),
                allow_transient=self.allow_transient,
            )
        return self


@dataclass(frozen=True)
class _ServiceDefinition:
    origin: str
    service_cls: type[object]
    key_template: str | None
    lifetime: ServiceLifetime
    eager: bool
    ctor: Ctor
    dtor: Dtor
    dependencies: dict[str, RequiredService]
    source: Mapping[str, object] | Callable[[], Mapping[str, object]] | None = None
    exposed_type: object | None = None


@dataclass(frozen=True)
class _ResolvedDependency:
    param_name: str
    dep_key: str | None
    dep_type: type[object] | None


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
    static_kwargs: dict[str, object]
    dependencies: tuple[_ResolvedDependency, ...]
    service_type: object


@dataclass(frozen=True)
class RegisteredService:
    key: str | None
    origin: str
    service_type: object


class AppServiceRegistry:
    """
    App-scoped registry for resolved service definitions.

    Each FastAPI app should use its own registry instance to avoid cross-app
    state interference (for example freeze/unfreeze side effects).
    """

    def __init__(
        self,
        definitions: Sequence[_ServiceDefinition] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._definitions_by_origin: dict[str, _ServiceDefinition] = {}
        self._definition_order: list[str] = []
        self._frozen = False
        if definitions:
            for definition in definitions:
                self.register(definition)

    def register(self, definition: _ServiceDefinition) -> None:
        with self._lock:
            existing = self._definitions_by_origin.get(definition.origin)
            if existing is not None:
                return
            if self._frozen:
                raise RuntimeError(
                    "AppServiceRegistry is frozen and cannot accept new service definitions."
                )
            self._definitions_by_origin[definition.origin] = definition
            self._definition_order.append(definition.origin)

    def definitions(self) -> list[_ServiceDefinition]:
        with self._lock:
            return [
                self._definitions_by_origin[origin]
                for origin in self._definition_order
                if origin in self._definitions_by_origin
            ]

    def freeze(self) -> None:
        with self._lock:
            self._frozen = True

    def unfreeze(self) -> None:
        with self._lock:
            self._frozen = False

    def is_frozen(self) -> bool:
        with self._lock:
            return self._frozen

    def clear_for_tests(self) -> None:
        with self._lock:
            self._definitions_by_origin.clear()
            self._definition_order.clear()
            self._frozen = False


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
        raise RuntimeError(
            "No active app service registry capture. "
            "Decorated services must be imported during install_di() startup."
        )
    include = capture.include_packages
    if include is not None and not _is_origin_included(definition.origin, include):
        return
    capture.registry.register(definition)


def require(target: str | type[object], *, allow_transient: bool = False) -> RequiredService:
    if not isinstance(target, (str, type)):
        raise TypeError("require() expects a service key (str) or a service type")
    return RequiredService(target=target, allow_transient=allow_transient)


def _normalize_lifetime(lifetime: LifetimeLike) -> ServiceLifetime:
    if isinstance(lifetime, ServiceLifetime):
        return lifetime
    if isinstance(lifetime, int):
        try:
            return ServiceLifetime(lifetime)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported lifetime value: {lifetime!r}. Use ServiceLifetime.SINGLETON/TRANSIENT or 0/1."
            ) from exc
    if isinstance(lifetime, str):
        normalized = lifetime.strip().lower()
        if normalized == "singleton":
            return ServiceLifetime.SINGLETON
        if normalized == "transient":
            return ServiceLifetime.TRANSIENT
        raise ValueError(
            f"Unsupported lifetime string: {lifetime!r}. Use 'Singleton' or 'Transient'."
        )
    raise TypeError(
        "Invalid lifetime type: "
        f"{type(lifetime)!r}. Use ServiceLifetime, int (0/1), or 'Singleton'/'Transient'."
    )


def _extract_ctors(service_cls: type[object]) -> tuple[Ctor, Dtor]:
    if inspect.isabstract(service_cls):
        raise TypeError(
            f"{service_cls!r} is abstract; define a concrete @classmethod create()."
        )

    raw_create = inspect.getattr_static(service_cls, "create", None)
    if raw_create is None:
        raise TypeError(f"{service_cls!r} is missing required classmethod create().")
    if not isinstance(raw_create, classmethod):
        raise TypeError(f"{service_cls!r}.create must be defined as @classmethod.")

    ctor = getattr(service_cls, "create", None)
    if ctor is None or not callable(ctor):
        raise TypeError(f"{service_cls!r}.create must be callable.")

    raw_destroy = inspect.getattr_static(service_cls, "destroy", None)
    if raw_destroy is None:
        return ctor, None
    if not isinstance(raw_destroy, classmethod):
        raise TypeError(f"{service_cls!r}.destroy must be defined as @classmethod.")
    if getattr(raw_destroy, _DEFAULT_DESTROY_MARKER, False):
        # Inherited default no-op hook from BaseService.
        return ctor, None

    dtor = getattr(service_cls, "destroy", None)
    if dtor is None or not callable(dtor):
        raise TypeError(f"{service_cls!r}.destroy must be callable.")
    return ctor, dtor


def _extract_dependencies(ctor: Ctor) -> dict[str, RequiredService]:
    deps: dict[str, RequiredService] = {}
    sig = inspect.signature(ctor)
    for name, parameter in sig.parameters.items():
        default = parameter.default
        if isinstance(default, RequiredService):
            deps[name] = default
    return deps


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
        raise ValueError(
            f"Service '{service_cls.__name__}' cannot be eager with transient lifetime."
        )
    if source is not None and key is None:
        raise ValueError(
            f"ServiceDict '{service_cls.__name__}' requires a non-empty key template."
        )
    definition = _ServiceDefinition(
        origin=f"{service_cls.__module__}.{service_cls.__qualname__}",
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
    setattr(service_cls, _SERVICE_DEFINITION_ATTR, definition)
    _register_definition_into_active_registry(definition)
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
        raise TypeError(
            "Service() expects a key string, a service class, or no positional argument."
        )

    def _decorator(service_cls: type[object]) -> type[object]:
        return _register_with_key(service_cls, key)

    return _decorator


def ServiceDict(
    key: str,
    *,
    dict: Mapping[str, object] | Callable[[], Mapping[str, object]],
    lifetime: LifetimeLike = ServiceLifetime.SINGLETON,
    eager: bool = False,
    exposed_type: object | None = None,
) -> Callable[[type[object]], type[object]]:
    def _decorator(service_cls: type[object]) -> type[object]:
        return _register_service_class(
            service_cls,
            key=key,
            source=dict,
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
        raise ValueError("At least one service package must be provided.")
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
        definition = getattr(cls, _SERVICE_DEFINITION_ATTR, None)
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
) -> dict[str, object]:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        raw = value.model_dump()
        if not isinstance(raw, Mapping):
            raise TypeError("model_dump() must return a mapping")
        return dict(raw)

    if isinstance(value, Mapping):
        return dict(value)

    for name, parameter in signature.parameters.items():
        if name in dependency_params:
            continue
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        return {name: value}

    raise TypeError("Unable to map ServiceDict value to ctor parameters")


def _resolve_source(
    source: Mapping[str, object] | Callable[[], Mapping[str, object]],
) -> Mapping[str, object]:
    resolved = source() if callable(source) else source
    if not isinstance(resolved, Mapping):
        raise TypeError("ServiceDict source must resolve to a mapping")
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


def create_app_service_registry(
    *,
    include_packages: Sequence[str] | None = None,
) -> AppServiceRegistry:
    _ = include_packages
    return AppServiceRegistry()


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
            expanded.append((definition, definition.key_template, {}, definition.dependencies, signature))
            continue

        if definition.key_template is None:
            raise RuntimeError(
                f"Service definition '{definition.origin}' has dict source but no key template."
            )

        source_mapping = _resolve_source(definition.source)
        for raw_dict_key, raw_value in source_mapping.items():
            dict_key = str(raw_dict_key)
            service_key = _render_service_key(definition.key_template, dict_key)
            static_kwargs = _coerce_mapping_value(
                raw_value,
                signature=signature,
                dependency_params=dependency_params,
            )
            deps = {
                param_name: dep.render_for_dict_key(dict_key)
                for param_name, dep in definition.dependencies.items()
            }
            expanded.append((definition, service_key, static_kwargs, deps, signature))

    return expanded


def _resolve_dependency_targets(
    compiled: list[ExpandedDefinition],
) -> list[_CompiledService]:
    by_internal_id: dict[str, ResolvedByInternalId] = {}
    public_key_index: dict[str, str] = {}
    type_index: dict[object, list[str]] = defaultdict(list)

    anon_seq = 0
    for definition, key, static_kwargs, deps, signature in compiled:
        if key is not None:
            existing_internal_id = public_key_index.get(key)
            if existing_internal_id is not None:
                first = by_internal_id[existing_internal_id][0]
                raise RuntimeError(
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
            static_kwargs,
            service_type,
            key,
        )
        type_index[service_type].append(internal_id)

    dependency_map: dict[str, set[str]] = {}
    resolved: dict[str, ResolvedSpec] = {}

    def _service_label(internal_id: str) -> str:
        definition, _, _, _, _, registration_key = by_internal_id[internal_id]
        return registration_key or definition.origin

    for internal_id, (
        definition,
        deps,
        signature,
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
                    raise RuntimeError(
                        f"Service '{service_label}' depends on '{dep_key}', but that service is not registered."
                    )
                resolved_deps.append(
                    _ResolvedDependency(
                        param_name=param_name,
                        dep_key=dep_key,
                        dep_type=None,
                    )
                )
            else:
                candidate_plan_keys = type_index.get(dep.target, [])
                if not candidate_plan_keys:
                    raise RuntimeError(
                        f"Service '{service_label}' depends on type {dep.target!r}, but no service provides that type."
                    )
                if len(candidate_plan_keys) > 1:
                    candidate_labels = [_service_label(candidate_id) for candidate_id in candidate_plan_keys]
                    raise RuntimeError(
                        "Service "
                        f"'{service_label}' depends on type {dep.target!r}, "
                        f"but multiple services provide it: {candidate_labels}."
                    )
                dep_internal_id = candidate_plan_keys[0]
                resolved_deps.append(
                    _ResolvedDependency(
                        param_name=param_name,
                        dep_key=None,
                        dep_type=dep.target,
                    )
                )

            target_definition = by_internal_id[dep_internal_id][0]
            target_key = by_internal_id[dep_internal_id][5]
            target_label = target_key or target_definition.origin
            if (
                definition.lifetime == ServiceLifetime.SINGLETON
                and target_definition.lifetime == ServiceLifetime.TRANSIENT
                and not dep.allow_transient
            ):
                raise RuntimeError(
                    f"Singleton service '{service_label}' depends on transient service '{target_label}'. "
                    "Use require(..., allow_transient=True) only if this is intentional."
                )

            dep_internal_ids.add(dep_internal_id)

        dependency_map[internal_id] = dep_internal_ids
        resolved[internal_id] = (
            definition,
            signature,
            static_kwargs,
            tuple(resolved_deps),
            by_internal_id[internal_id][4],
            registration_key,
        )

    registration_order = _topological_registration_order(dependency_map)

    result: list[_CompiledService] = []
    for internal_id in registration_order:
        (
            definition,
            signature,
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
            raise RuntimeError(f"Detected circular service dependency: {cycle_path}")
        raise RuntimeError("Detected circular service dependency")

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
    return _resolve_dependency_targets(expanded)


def _make_bound_ctor(container: ServiceContainer, spec: _CompiledService) -> Ctor:
    ctor = spec.ctor
    signature = spec.signature.replace(return_annotation=spec.service_type)

    async def _bound_ctor(*args: object, **kwargs: object) -> object:
        provided_names: set[str] = set()
        try:
            provided_names = set(signature.bind_partial(*args, **kwargs).arguments)
        except TypeError:
            # Defer detailed signature errors to the original ctor call.
            provided_names = set()

        call_kwargs = dict(spec.static_kwargs)
        for name in provided_names:
            call_kwargs.pop(name, None)
        call_kwargs.update(kwargs)

        request = container.current_request()
        request_kwarg_name = container.request_kwarg_name()

        for dep in spec.dependencies:
            if dep.param_name in provided_names or dep.param_name in call_kwargs:
                continue
            dep_kwargs: dict[str, object] = {}
            if request is not None:
                dep_kwargs[request_kwarg_name] = request
            if dep.dep_key is not None:
                call_kwargs[dep.param_name] = await container.aget_by_key(dep.dep_key, **dep_kwargs)
            else:
                if dep.dep_type is None:  # pragma: no cover - defensive
                    raise RuntimeError(
                        f"Invalid dependency spec for parameter '{dep.param_name}': missing key and type."
                    )
                call_kwargs[dep.param_name] = await container.aget_by_type(dep.dep_type, **dep_kwargs)

        if inspect.iscoroutinefunction(ctor):
            return await ctor(*args, **call_kwargs)
        if inspect.isasyncgenfunction(ctor):
            return ctor(*args, **call_kwargs)

        result = await asyncio.to_thread(ctor, *args, **call_kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    name_suffix = spec.key or spec.origin
    safe_suffix = "".join(ch if ch.isalnum() else "_" for ch in name_suffix).strip("_") or "service"
    _bound_ctor.__name__ = f"autoreg_ctor_{safe_suffix}"
    _bound_ctor.__qualname__ = _bound_ctor.__name__
    bound_ctor_with_signature = cast(_CallableWithSignature, _bound_ctor)
    bound_ctor_with_signature.__signature__ = signature
    annotations = dict(getattr(ctor, "__annotations__", {}))
    annotations["return"] = spec.service_type
    _bound_ctor.__annotations__ = annotations
    return _bound_ctor


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
    registered_services: list[RegisteredService] = []
    for spec in plan:
        bound_ctor = _make_bound_ctor(container, spec)
        await container.register(
            spec.key,
            spec.lifetime,
            bound_ctor,
            spec.dtor,
        )
        if spec.eager:
            if spec.key is not None:
                if eager_init_timeout_sec is not None:
                    await asyncio.wait_for(
                        container.aget_by_key(spec.key),
                        timeout=eager_init_timeout_sec,
                    )
                else:
                    await container.aget_by_key(spec.key)
            else:
                if eager_init_timeout_sec is not None:
                    await asyncio.wait_for(
                        container.aget_by_type(spec.service_type),
                        timeout=eager_init_timeout_sec,
                    )
                else:
                    await container.aget_by_type(spec.service_type)
        registered_services.append(
            RegisteredService(
                key=spec.key,
                origin=spec.origin,
                service_type=spec.service_type,
            )
        )

    logger.debug("[LIFESPAN] Auto-registered services count=%s", len(plan))
    return registered_services
