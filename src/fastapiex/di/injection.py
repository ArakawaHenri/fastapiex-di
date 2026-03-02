from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Protocol, cast

from fastapi.params import Depends
from starlette.requests import HTTPConnection

from .activator import refresh_services_for_container
from .constants import (
    APP_STATE_DI_CONFIG_ATTR,
    APP_STATE_DI_GLOBAL_REFRESH_LOCK_ATTR,
    APP_STATE_DI_REGISTERED_SERVICE_ORIGINS_ATTR,
    APP_STATE_DI_SERVICE_REGISTRY_ATTR,
    APP_STATE_SC_REGISTRY_ATTR,
)
from .container import ServiceContainer
from .exceptions import (
    InvalidServiceDefinitionError,
    ServiceContainerStateError,
    UnregisteredServiceError,
)
from .types import CallableWithSignature

logger = logging.getLogger(__name__)


class ServiceContainerRegistry:
    """
    Thread-safe registry mapping the current (process, thread, event loop)
    execution context to a ServiceContainer.

    This lets one FastAPI app object host multiple independent containers when
    workers are implemented as threads (e.g., free-threaded runtimes).
    """

    def __init__(self) -> None:
        self._containers: dict[tuple[int, int, int], ServiceContainer] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _current_key() -> tuple[int, int, int]:
        loop = asyncio.get_running_loop()
        return (os.getpid(), threading.get_ident(), id(loop))

    def register_current(self, container: ServiceContainer) -> tuple[int, int, int]:
        key = self._current_key()
        with self._lock:
            existing = self._containers.get(key)
            if existing is not None and existing is not container:
                raise ServiceContainerStateError(
                    "A different ServiceContainer is already registered for the current event loop."
                )
            self._containers[key] = container
        return key

    def get_current(self) -> ServiceContainer | None:
        try:
            key = self._current_key()
        except RuntimeError:
            return None
        with self._lock:
            return self._containers.get(key)

    def unregister_current(
        self,
        *,
        expected: ServiceContainer | None = None,
    ) -> ServiceContainer | None:
        try:
            key = self._current_key()
        except RuntimeError:
            return None
        with self._lock:
            existing = self._containers.get(key)
            if existing is None:
                return None
            if expected is not None and existing is not expected:
                logger.warning(
                    "ServiceContainer registry mismatch on unregister; keeping current mapping."
                )
                return None
            return self._containers.pop(key)


_APP_STATE_REGISTRY_LOCK = threading.Lock()


class _AppStateWithRegistry(Protocol):
    sc_registry: ServiceContainerRegistry


def get_or_create_service_container_registry(
    app_state: object,
) -> ServiceContainerRegistry:
    """
    Return the app-level ServiceContainerRegistry, creating it if necessary.
    """
    registry = getattr(app_state, APP_STATE_SC_REGISTRY_ATTR, None)
    if isinstance(registry, ServiceContainerRegistry):
        return registry

    with _APP_STATE_REGISTRY_LOCK:
        registry = getattr(app_state, APP_STATE_SC_REGISTRY_ATTR, None)
        if isinstance(registry, ServiceContainerRegistry):
            return registry
        registry = ServiceContainerRegistry()
        state = cast(_AppStateWithRegistry, app_state)
        setattr(state, APP_STATE_SC_REGISTRY_ATTR, registry)
        return registry


def resolve_service_container(app_state: object) -> ServiceContainer | None:
    """
    Resolve the ServiceContainer for the current execution context.

    Resolution source:
    - loop-local registry on app.state (free-threaded safe)
    """
    if app_state is None:
        return None

    registry = getattr(app_state, APP_STATE_SC_REGISTRY_ATTR, None)
    if isinstance(registry, ServiceContainerRegistry):
        return registry.get_current()
    return None


async def _di_cleanup_scope_dependency(request: HTTPConnection) -> AsyncIterator[None]:
    services = resolve_service_container(getattr(request.app, "state", None))
    if services is None:
        msg = (
            "Service container not initialized for the current event loop on FastAPI app state (sc_registry)."
        )
        logger.error(msg)
        raise ServiceContainerStateError(msg)

    if services._resolve_cleanup_scope_stack(request) is not None:
        yield
        return

    async with AsyncExitStack() as cleanup_stack:
        services._bind_cleanup_scope_stack(request, cleanup_stack)
        try:
            yield
        finally:
            services._clear_cleanup_scope_stack(request, expected=cleanup_stack)


async def _di_transaction_scope_dependency(
    request: HTTPConnection,
    _di_cleanup_scope: object = Depends(
        _di_cleanup_scope_dependency,
        use_cache=True,
        scope="request",
    ),
) -> AsyncIterator[None]:
    _ = _di_cleanup_scope
    services = resolve_service_container(getattr(request.app, "state", None))
    if services is None:
        msg = (
            "Service container not initialized for the current event loop on FastAPI app state (sc_registry)."
        )
        logger.error(msg)
        raise ServiceContainerStateError(msg)

    if services._resolve_transaction_scope_stack(request) is not None:
        yield
        return

    async with AsyncExitStack() as transaction_stack:
        services._bind_transaction_scope_stack(request, transaction_stack)
        try:
            yield
        finally:
            services._clear_transaction_scope_stack(
                request,
                expected=transaction_stack,
            )


def Inject(
        target: str | type[object],
        *args: object,
        **kwargs: object,
) -> Depends:
    """
    Create a FastAPI dependency marker for a registered service.

    `target` is required and must be either:
    - a service key string, or
    - a service type used for unique type-based lookup.

    In endpoints you can write:

        @router.get("/items")
        async def endpoint(db: DbConn = Inject(DbConn)):
            ...

    Resolution modes
    ----------------
    1. Key-based resolution:
        - Inject("my_key")

    2. Type-based resolution:
        - Inject(MyType)

       This requires that exactly one service of type MyType is registered.

    Calling `Inject()` without a target is not supported.
    """
    lookup_value: str | type[object]
    if isinstance(target, str):
        key_specified = True
        lookup_value = target
    elif isinstance(target, type):
        key_specified = False
        lookup_value = target
    else:
        raise InvalidServiceDefinitionError(
            "Inject() expects either a service key (str) or a service type."
        )

    params = [
        inspect.Parameter(
            "request",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=HTTPConnection,
        )
    ]
    params.append(
        inspect.Parameter(
            "_di_transaction_scope",
            inspect.Parameter.KEYWORD_ONLY,
            default=Depends(
                _di_transaction_scope_dependency,
                use_cache=True,
                scope="function",
            ),
        )
    )
    pos_dep_map: dict[int, str] = {}
    pos_static: dict[int, object] = {}
    kw_dep_map: dict[str, str] = {}
    kw_static: dict[str, object] = {}

    for i, value in enumerate(args):
        if isinstance(value, Depends):
            name = f"_dep_arg_{i}_{uuid.uuid4().hex[:8]}"
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=value,
                )
            )
            pos_dep_map[i] = name
            continue
        pos_static[i] = value

    for key, value in kwargs.items():
        if isinstance(value, Depends):
            name = f"_dep_kw_{key}_{uuid.uuid4().hex[:8]}"
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=value,
                )
            )
            kw_dep_map[key] = name
            continue
        kw_static[key] = value

    signature = inspect.Signature(params)

    async def _resolve_once(
        services: ServiceContainer,
        *runtime_args: object,
        **runtime_kwargs: object,
    ) -> object:
        if isinstance(lookup_value, str):
            return await services.aget_by_key(lookup_value, *runtime_args, **runtime_kwargs)
        return await services.aget_by_type(lookup_value, *runtime_args, **runtime_kwargs)

    async def _refresh_container_registrations(
        request: HTTPConnection,
        services: ServiceContainer,
    ) -> bool:
        app_state = getattr(request.app, "state", None)
        di_config = getattr(app_state, APP_STATE_DI_CONFIG_ATTR, None)
        app_service_registry = getattr(app_state, APP_STATE_DI_SERVICE_REGISTRY_ATTR, None)
        registered_origins = getattr(
            app_state,
            APP_STATE_DI_REGISTERED_SERVICE_ORIGINS_ATTR,
            None,
        )
        refresh_lock = getattr(app_state, APP_STATE_DI_GLOBAL_REFRESH_LOCK_ATTR, None)
        if app_service_registry is None or not isinstance(registered_origins, set):
            return False

        async def _run_refresh() -> None:
            await refresh_services_for_container(
                services,
                registry=app_service_registry,
                registered_origins=registered_origins,
                include_global_registry=bool(
                    getattr(di_config, "use_global_service_registry", False)
                ),
                eager_init_timeout_sec=getattr(
                    di_config,
                    "eager_init_timeout_sec",
                    None,
                ),
            )

        if isinstance(refresh_lock, asyncio.Lock):
            async with refresh_lock:
                await _run_refresh()
        else:
            await _run_refresh()
        return True

    async def _dependency_callable(
        request: HTTPConnection,
        **resolved_deps: object,
    ) -> object:
        services = resolve_service_container(getattr(request.app, "state", None))
        if services is None:
            msg = (
                "Service container not initialized for the current event loop on FastAPI app state (sc_registry)."
            )
            logger.error(msg)
            raise ServiceContainerStateError(msg)

        final_args = [
            resolved_deps[pos_dep_map[index]]
            if index in pos_dep_map
            else pos_static[index]
            for index in range(len(args))
        ]
        final_kwargs = {
            key: resolved_deps[kw_dep_map[key]]
            if key in kw_dep_map
            else kw_static[key]
            for key in kwargs
        }
        final_kwargs[services.request_kwarg_name()] = request

        try:
            return await _resolve_once(services, *final_args, **final_kwargs)
        except UnregisteredServiceError:
            refreshed = await _refresh_container_registrations(request, services)
            if not refreshed:
                raise
            return await _resolve_once(services, *final_args, **final_kwargs)

    name_suffix = (
        str(lookup_value)
        if key_specified
        else getattr(lookup_value, "__name__", repr(lookup_value))
    )

    _dependency_callable.__name__ = (
        f"inject_{'key' if key_specified else 'type'}_{name_suffix}"
    )
    _dependency_callable.__qualname__ = _dependency_callable.__name__
    dependency_callable_with_signature = cast(
        CallableWithSignature, _dependency_callable
    )
    dependency_callable_with_signature.__signature__ = signature

    return Depends(_dependency_callable)
