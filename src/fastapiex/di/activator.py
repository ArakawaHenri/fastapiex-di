from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Sequence
from typing import cast

from .container import ServiceContainer
from .planner import _compose_ctor_call_arguments, build_service_plan
from .registry import (
    AppServiceRegistry,
    RegisteredService,
    RequiredService,
    _CompiledService,
    get_global_service_definitions,
)
from .types import CallableWithSignature, Ctor

logger = logging.getLogger(__name__)


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
    registered_service_ids: set[str],
) -> list[_CompiledService]:
    return [
        spec for spec in plan
        if spec.registration_id not in registered_service_ids
    ]


def _registered_service_from_spec(spec: _CompiledService) -> RegisteredService:
    return RegisteredService(
        registration_id=spec.registration_id,
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
    registered_service_ids: set[str] | None = None,
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
        if registered_service_ids is not None:
            registered_service_ids.add(spec.registration_id)
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
    registered_service_ids: set[str],
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
    new_specs = _select_unregistered_specs(plan, registered_service_ids)
    if not new_specs:
        return []

    registered_services = await _register_compiled_specs(
        container,
        new_specs,
        eager_init_timeout_sec=eager_init_timeout_sec,
        registered_service_ids=registered_service_ids,
    )
    logger.debug(
        "[LIFESPAN] Runtime refresh registered services count=%s",
        len(registered_services),
    )
    return registered_services
