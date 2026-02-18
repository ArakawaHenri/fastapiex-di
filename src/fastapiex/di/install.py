from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Sequence

from fastapi import FastAPI

from .container import (
    ServiceContainer,
    TransientServiceFinalizerMiddleware,
    get_or_create_service_container_registry,
)
from .registry import (
    AppServiceRegistry,
    capture_service_registrations,
    import_service_modules,
    register_module_service_definitions,
    register_services_from_registry,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DISettings:
    service_packages: tuple[str, ...]
    strict: bool = True
    allow_private_modules: bool = False
    auto_add_finalizer_middleware: bool = True
    freeze_container_after_startup: bool = True
    freeze_service_registry_after_startup: bool = False
    unfreeze_service_registry_on_shutdown: bool = True
    eager_init_timeout_sec: float | None = None


def _normalize_service_packages(service_packages: str | Sequence[str]) -> tuple[str, ...]:
    normalized: tuple[str, ...]
    if isinstance(service_packages, str):
        normalized = (service_packages,)
    else:
        normalized = tuple(p for p in service_packages if p)
    if not normalized:
        raise ValueError("install_di() requires at least one service package.")
    return normalized


def _has_middleware(app: FastAPI, middleware_cls: type[object]) -> bool:
    for middleware in app.user_middleware:
        middleware_spec = middleware
        if getattr(middleware_spec, "cls", None) is middleware_cls:
            return True
    return False


def install_di(
    app: FastAPI,
    *,
    service_packages: str | Sequence[str],
    strict: bool = True,
    allow_private_modules: bool = False,
    auto_add_finalizer_middleware: bool = True,
    freeze_container_after_startup: bool = True,
    freeze_service_registry_after_startup: bool = False,
    unfreeze_service_registry_on_shutdown: bool = True,
    eager_init_timeout_sec: float | None = None,
) -> DISettings:
    """
    Install DI/service-registry lifecycle into a FastAPI app.

    This function wires startup/shutdown automatically and can be called once
    during app creation.
    """
    if eager_init_timeout_sec is not None and eager_init_timeout_sec <= 0:
        raise ValueError("eager_init_timeout_sec must be > 0 when provided.")
    if isinstance(getattr(app.state, "di_settings", None), DISettings):
        raise RuntimeError("install_di() has already been called for this FastAPI app.")

    settings = DISettings(
        service_packages=_normalize_service_packages(service_packages),
        strict=strict,
        allow_private_modules=allow_private_modules,
        auto_add_finalizer_middleware=auto_add_finalizer_middleware,
        freeze_container_after_startup=freeze_container_after_startup,
        freeze_service_registry_after_startup=freeze_service_registry_after_startup,
        unfreeze_service_registry_on_shutdown=unfreeze_service_registry_on_shutdown,
        eager_init_timeout_sec=eager_init_timeout_sec,
    )

    if settings.auto_add_finalizer_middleware and not _has_middleware(
        app,
        TransientServiceFinalizerMiddleware,
    ):
        # Register cleanup early so user middlewares wrap the full request chain.
        app.add_middleware(TransientServiceFinalizerMiddleware)

    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _combined_lifespan(inner_app: FastAPI) -> AsyncIterator[None]:
        services: ServiceContainer | None = None
        sc_registry = None
        app_service_registry: AppServiceRegistry | None = None

        try:
            try:
                services = ServiceContainer()
                sc_registry = get_or_create_service_container_registry(inner_app.state)
                sc_registry.register_current(services)

                app_service_registry = AppServiceRegistry()
                inner_app.state.di_service_registry = app_service_registry
                with capture_service_registrations(
                    app_service_registry,
                    include_packages=settings.service_packages,
                ):
                    imported_modules = import_service_modules(
                        settings.service_packages,
                        allow_private_modules=settings.allow_private_modules,
                    )
                register_module_service_definitions(
                    app_service_registry,
                    imported_modules,
                    include_packages=settings.service_packages,
                )
                await register_services_from_registry(
                    services,
                    registry=app_service_registry,
                    eager_init_timeout_sec=settings.eager_init_timeout_sec,
                )

                if settings.freeze_container_after_startup:
                    services.freeze_registrations()

                if (
                    settings.freeze_service_registry_after_startup
                    and app_service_registry is not None
                ):
                    app_service_registry.freeze()

            except Exception:
                if settings.strict:
                    raise

                logger.exception(
                    "DI startup failed; continuing without DI because strict=False"
                )
                if services is not None:
                    try:
                        await services.destruct_all_singletons()
                    finally:
                        if sc_registry is not None:
                            sc_registry.unregister_current(expected=services)
                services = None
                sc_registry = None
                inner_app.state.di_service_registry = None

            async with previous_lifespan(inner_app):
                yield

        finally:
            if services is not None:
                try:
                    await services.destruct_all_singletons()
                finally:
                    if sc_registry is not None:
                        sc_registry.unregister_current(expected=services)

            if (
                settings.freeze_service_registry_after_startup
                and settings.unfreeze_service_registry_on_shutdown
                and app_service_registry is not None
            ):
                app_service_registry.unfreeze()

    app.router.lifespan_context = _combined_lifespan
    app.state.di_settings = settings
    app.state.di_service_registry = None
    return settings
