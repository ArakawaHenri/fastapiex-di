from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Protocol

from fastapi import FastAPI

from .activator import register_services_from_registry
from .constants import (
    APP_STATE_DI_CONFIG_ATTR,
    APP_STATE_DI_GLOBAL_REFRESH_LOCK_ATTR,
    APP_STATE_DI_REGISTERED_SERVICE_ORIGINS_ATTR,
    APP_STATE_DI_SERVICE_REGISTRY_ATTR,
)
from .container import ServiceContainer
from .discovery import (
    import_service_modules,
    register_module_service_definitions,
    resolve_service_package_paths,
)
from .exceptions import DIAlreadyInstalledError, DIConfigurationError
from .injection import ServiceContainerRegistry, get_or_create_service_container_registry
from .registry import (
    AppServiceRegistry,
    capture_service_registrations,
    get_global_service_definitions,
    register_runtime_registry_binding,
    unregister_runtime_registry_binding,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DIConfig:
    service_packages: tuple[str, ...]
    service_package_imports: tuple[str, ...]
    strict: bool = True
    use_global_service_registry: bool = False
    allow_private_modules: bool = False
    eager_init_timeout_sec: float | None = None


class _ContainerRegistryLike(Protocol):
    def unregister_current(
        self,
        *,
        expected: ServiceContainer | None = None,
    ) -> object | None:
        ...


def _normalize_service_packages(service_packages: str | Sequence[str]) -> tuple[str, ...]:
    normalized: tuple[str, ...]
    if isinstance(service_packages, str):
        normalized = (service_packages,)
    else:
        normalized = tuple(p for p in service_packages if p)
    if not normalized:
        raise DIConfigurationError("install_di() requires at least one service package.")
    return normalized


def _resolve_runtime_config(config: DIConfig) -> DIConfig:
    resolved_paths = resolve_service_package_paths(config.service_package_imports)
    if resolved_paths == config.service_packages:
        return config
    return replace(config, service_packages=resolved_paths)


def _build_app_service_registry(config: DIConfig) -> AppServiceRegistry:
    global_defs = (
        get_global_service_definitions()
        if config.use_global_service_registry
        else None
    )
    return AppServiceRegistry(definitions=global_defs)


def _initialize_runtime_app_state(
    app: FastAPI,
    *,
    app_service_registry: AppServiceRegistry,
) -> None:
    setattr(app.state, APP_STATE_DI_SERVICE_REGISTRY_ATTR, app_service_registry)
    setattr(app.state, APP_STATE_DI_REGISTERED_SERVICE_ORIGINS_ATTR, set())
    setattr(app.state, APP_STATE_DI_GLOBAL_REFRESH_LOCK_ATTR, asyncio.Lock())


def _clear_runtime_app_state(app: FastAPI) -> None:
    setattr(app.state, APP_STATE_DI_SERVICE_REGISTRY_ATTR, None)
    setattr(app.state, APP_STATE_DI_REGISTERED_SERVICE_ORIGINS_ATTR, set())
    setattr(app.state, APP_STATE_DI_GLOBAL_REFRESH_LOCK_ATTR, None)


def _load_registry_definitions(
    app_service_registry: AppServiceRegistry,
    *,
    config: DIConfig,
) -> None:
    with capture_service_registrations(
        app_service_registry,
        include_packages=config.service_package_imports,
    ):
        imported_modules = import_service_modules(
            config.service_package_imports,
            allow_private_modules=config.allow_private_modules,
        )
    register_module_service_definitions(
        app_service_registry,
        imported_modules,
        include_packages=config.service_package_imports,
    )


def _unregister_runtime_binding_if_needed(
    *,
    binding_registered: bool,
    app_service_registry: AppServiceRegistry | None,
) -> bool:
    if not binding_registered or app_service_registry is None:
        return binding_registered
    unregister_runtime_registry_binding(app_service_registry)
    return False


async def _destruct_container_if_needed(
    *,
    services: ServiceContainer | None,
    sc_registry: _ContainerRegistryLike | None,
) -> None:
    if services is None:
        return
    try:
        await services.destruct_all_singletons()
    finally:
        if sc_registry is not None:
            sc_registry.unregister_current(expected=services)


def install_di(
    app: FastAPI,
    *,
    service_packages: str | Sequence[str],
    strict: bool = True,
    use_global_service_registry: bool = False,
    allow_private_modules: bool = False,
    eager_init_timeout_sec: float | None = None,
) -> DIConfig:
    """
    Install DI/service-registry lifecycle into a FastAPI app.

    This function wires startup/shutdown automatically and can be called once
    during app creation.
    """
    if eager_init_timeout_sec is not None and eager_init_timeout_sec <= 0:
        raise DIConfigurationError("eager_init_timeout_sec must be > 0 when provided.")
    if isinstance(getattr(app.state, APP_STATE_DI_CONFIG_ATTR, None), DIConfig):
        raise DIAlreadyInstalledError(
            "install_di() has already been called for this FastAPI app.")

    service_package_imports = _normalize_service_packages(service_packages)
    config = DIConfig(
        service_packages=resolve_service_package_paths(service_package_imports),
        service_package_imports=service_package_imports,
        strict=strict,
        use_global_service_registry=use_global_service_registry,
        allow_private_modules=allow_private_modules,
        eager_init_timeout_sec=eager_init_timeout_sec,
    )

    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _combined_lifespan(inner_app: FastAPI) -> AsyncIterator[None]:
        services: ServiceContainer | None = None
        sc_registry: ServiceContainerRegistry | None = None
        app_service_registry: AppServiceRegistry | None = None
        binding_registered = False
        runtime_config = config

        try:
            try:
                services = ServiceContainer()
                sc_registry = get_or_create_service_container_registry(
                    inner_app.state)
                sc_registry.register_current(services)

                runtime_config = _resolve_runtime_config(runtime_config)
                if runtime_config != config:
                    setattr(inner_app.state, APP_STATE_DI_CONFIG_ATTR, runtime_config)

                app_service_registry = _build_app_service_registry(runtime_config)
                _initialize_runtime_app_state(
                    inner_app,
                    app_service_registry=app_service_registry,
                )

                register_runtime_registry_binding(
                    app_service_registry,
                    package_paths=runtime_config.service_packages,
                    use_global_service_registry=runtime_config.use_global_service_registry,
                )
                binding_registered = True
                _load_registry_definitions(
                    app_service_registry,
                    config=runtime_config,
                )

                registered_services = await register_services_from_registry(
                    services,
                    registry=app_service_registry,
                    eager_init_timeout_sec=runtime_config.eager_init_timeout_sec,
                )
                setattr(inner_app.state, APP_STATE_DI_REGISTERED_SERVICE_ORIGINS_ATTR, {
                    item.origin for item in registered_services
                })

            except Exception:
                if config.strict:
                    raise

                logger.exception(
                    "DI startup failed; continuing without DI because strict=False"
                )
                await _destruct_container_if_needed(
                    services=services,
                    sc_registry=sc_registry,
                )
                binding_registered = _unregister_runtime_binding_if_needed(
                    binding_registered=binding_registered,
                    app_service_registry=app_service_registry,
                )
                services = None
                sc_registry = None
                _clear_runtime_app_state(inner_app)

            async with previous_lifespan(inner_app):
                yield

        finally:
            await _destruct_container_if_needed(
                services=services,
                sc_registry=sc_registry,
            )
            _unregister_runtime_binding_if_needed(
                binding_registered=binding_registered,
                app_service_registry=app_service_registry,
            )

    app.router.lifespan_context = _combined_lifespan
    setattr(app.state, APP_STATE_DI_CONFIG_ATTR, config)
    _clear_runtime_app_state(app)
    return config
