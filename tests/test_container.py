from __future__ import annotations

import os
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.requests import Request

from fastapiex.di import (
    REQUEST_FAILED_STATE_KEY,
    ServiceContainer,
    ServiceContainerRegistry,
    ServiceLifetime,
    get_or_create_service_container_registry,
    resolve_service_container,
)


@pytest.mark.asyncio
async def test_singleton_lifecycle():
    """Test that singleton services are created only once"""
    container = ServiceContainer()

    call_count = 0

    async def factory():
        nonlocal call_count
        call_count += 1
        return {"instance": call_count}

    await container.register("test", ServiceLifetime.SINGLETON, factory, None)

    instance1 = cast(dict[str, int], await container.aget_by_key("test"))
    instance2 = cast(dict[str, int], await container.aget_by_key("test"))

    assert instance1 is instance2
    assert call_count == 1
    assert instance1["instance"] == 1


@pytest.mark.asyncio
async def test_transient_lifecycle():
    """Test that transient services create new instances each time"""
    container = ServiceContainer()

    call_count = 0

    async def factory():
        nonlocal call_count
        call_count += 1
        return {"instance": call_count}

    await container.register("test", ServiceLifetime.TRANSIENT, factory, None)

    # Transient services create new instances
    instance1 = await container.aget_by_key("test")
    instance2 = await container.aget_by_key("test")

    assert instance1 is not instance2
    assert call_count == 2


@pytest.mark.asyncio
async def test_type_based_injection():
    """Test type-based service injection"""
    container = ServiceContainer()

    class MyService:
        value: int = 42

    async def factory() -> MyService:
        return MyService()

    # Anonymous registration (accessible only by type)
    await container.register(None, ServiceLifetime.SINGLETON, factory, None)

    instance = await container.aget_by_type(MyService)
    assert isinstance(instance, MyService)
    assert instance.value == 42


@pytest.mark.asyncio
async def test_anonymous_service_without_type_fails():
    """Test that anonymous service registration fails when type cannot be inferred"""
    container = ServiceContainer()

    # Factory function without type annotation
    async def factory():
        return {"data": "test"}

    with pytest.raises(TypeError, match="unable to infer service type"):
        await container.register(None, ServiceLifetime.SINGLETON, factory, None)


@pytest.mark.asyncio
async def test_async_generator_service():
    """Test async generator service"""
    container = ServiceContainer()

    cleanup_called = False

    async def factory() -> AsyncIterator[dict]:
        nonlocal cleanup_called
        try:
            yield {"data": "test"}
        finally:
            cleanup_called = True

    await container.register("gen", ServiceLifetime.TRANSIENT, factory, None)

    instance = cast(dict[str, str], await container.aget_by_key("gen"))
    assert instance["data"] == "test"

    # Note: In actual usage, cleanup is called automatically at request end


@pytest.mark.asyncio
async def test_process_isolation():
    """Test process isolation check (using mock simulation)"""
    container = ServiceContainer()

    # Save original PID
    original_pid = container._pid

    async def factory():
        return {}

    await container.register("test", ServiceLifetime.SINGLETON, factory, None)

    # Restore correct PID to avoid registration failure
    container._pid = os.getpid()

    # Now modify PID to simulate cross-process access
    container._pid = original_pid + 1

    with pytest.raises(RuntimeError, match="accessed from different process"):
        await container.aget_by_key("test")


@pytest.mark.asyncio
async def test_singleton_destruction():
    """Test singleton service destruction"""
    container = ServiceContainer()

    destroyed = []

    async def factory():
        return {"id": 1}

    async def destructor(instance: object) -> None:
        payload = cast(dict[str, int], instance)
        destroyed.append(payload["id"])

    await container.register("test", ServiceLifetime.SINGLETON, factory, destructor)

    instance = cast(dict[str, int], await container.aget_by_key("test"))
    assert instance["id"] == 1

    await container.destruct_all_singletons()

    assert destroyed == [1]


@pytest.mark.asyncio
async def test_duplicate_key_registration_fails():
    """Test that registering the same key twice fails"""
    container = ServiceContainer()

    async def factory():
        return {}

    await container.register("test", ServiceLifetime.SINGLETON, factory, None)

    with pytest.raises(RuntimeError, match="Duplicate service registration"):
        await container.register("test", ServiceLifetime.SINGLETON, factory, None)


@pytest.mark.asyncio
async def test_register_rejects_non_callable_destructor():
    container = ServiceContainer()

    async def factory() -> dict[str, str]:
        return {"ok": "1"}

    with pytest.raises(TypeError, match="Invalid destructor"):
        await container.register(
            "bad_dtor",
            ServiceLifetime.SINGLETON,
            factory,
            cast(Any, "not-callable"),
        )


@pytest.mark.asyncio
async def test_register_rejects_invalid_lifetime():
    container = ServiceContainer()

    async def factory() -> dict[str, str]:
        return {"ok": "1"}

    with pytest.raises(TypeError, match="Invalid lifetime"):
        await container.register(
            "bad_lifetime",
            cast(Any, 2),
            factory,
            None,
        )


@pytest.mark.asyncio
async def test_register_rejects_destructor_without_instance_param():
    container = ServiceContainer()

    async def factory() -> dict[str, str]:
        return {"ok": "1"}

    async def bad_destructor() -> None:
        return None

    with pytest.raises(TypeError, match="Invalid destructor signature"):
        await container.register(
            "bad_dtor_signature",
            ServiceLifetime.SINGLETON,
            factory,
            cast(Any, bad_destructor),
        )


@pytest.mark.asyncio
async def test_key_based_injection():
    """Test key-based service injection"""
    container = ServiceContainer()

    async def factory():
        return {"message": "Hello from service"}

    await container.register("my_service", ServiceLifetime.SINGLETON, factory, None)

    instance = cast(dict[str, str], await container.aget_by_key("my_service"))
    assert instance["message"] == "Hello from service"


@pytest.mark.asyncio
async def test_async_generator_transient_finalizer_exhausts_generator():
    """Transient async generator finalization must execute post-yield logic."""
    container = ServiceContainer()
    marks: list[str] = []

    async def factory() -> AsyncIterator[dict[str, bool]]:
        try:
            yield {"ok": True}
        except Exception:
            marks.append("except")
            raise
        else:
            marks.append("else")
        finally:
            marks.append("finally")

    await container.register("gen", ServiceLifetime.TRANSIENT, factory, None)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "state": {},
    }
    request = Request(scope)

    instance = await container.aget_by_key(
        "gen", **{container.request_kwarg_name(): request}
    )
    assert instance == {"ok": True}

    ctx = container._get_or_create_request_ctx(request)
    for finalizer in reversed(ctx["transient_finalizers"]):
        await finalizer()

    assert marks == ["else", "finally"]


@pytest.mark.asyncio
async def test_async_generator_transient_finalizer_uses_throw_on_failed_request():
    """Failed requests should drive transient async generators through rollback path."""
    container = ServiceContainer()
    marks: list[str] = []

    async def factory() -> AsyncIterator[dict[str, bool]]:
        try:
            yield {"ok": True}
        except Exception:
            marks.append("except")
            raise
        else:
            marks.append("else")
        finally:
            marks.append("finally")

    await container.register("gen", ServiceLifetime.TRANSIENT, factory, None)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "state": {},
    }
    request = Request(scope)
    setattr(request.state, REQUEST_FAILED_STATE_KEY, True)

    instance = await container.aget_by_key(
        "gen", **{container.request_kwarg_name(): request}
    )
    assert instance == {"ok": True}

    ctx = container._get_or_create_request_ctx(request)
    for finalizer in reversed(ctx["transient_finalizers"]):
        await finalizer()

    assert marks == ["except", "finally"]


@pytest.mark.asyncio
async def test_async_generator_transient_finalizer_rejects_iterator_style_factory():
    """Iterator-style async generators (multiple yields) violate single-yield contract."""
    container = ServiceContainer()
    marks: list[str] = []

    async def factory() -> AsyncIterator[int]:
        try:
            yield 1
            yield 2
            yield 3
        finally:
            marks.append("finally")

    await container.register("gen", ServiceLifetime.TRANSIENT, factory, None)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "state": {},
    }
    request = Request(scope)

    instance = await container.aget_by_key(
        "gen", **{container.request_kwarg_name(): request}
    )
    assert instance == 1

    ctx = container._get_or_create_request_ctx(request)
    for finalizer in reversed(ctx["transient_finalizers"]):
        with pytest.raises(RuntimeError, match="must yield exactly once"):
            await finalizer()

    assert marks == ["finally"]


@pytest.mark.asyncio
async def test_generator_transient_finalizer_rejects_iterator_style_factory():
    """Iterator-style generators (multiple yields) violate single-yield contract."""
    container = ServiceContainer()
    marks: list[str] = []

    def factory():
        try:
            yield 1
            yield 2
            yield 3
        finally:
            marks.append("finally")

    await container.register("gen", ServiceLifetime.TRANSIENT, factory, None)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "state": {},
    }
    request = Request(scope)

    instance = await container.aget_by_key(
        "gen", **{container.request_kwarg_name(): request}
    )
    assert instance == 1

    ctx = container._get_or_create_request_ctx(request)
    for finalizer in reversed(ctx["transient_finalizers"]):
        with pytest.raises(RuntimeError, match="must yield exactly once"):
            await finalizer()

    assert marks == ["finally"]


@pytest.mark.asyncio
async def test_service_container_registry_register_get_unregister():
    registry = ServiceContainerRegistry()
    container = ServiceContainer()

    registry.register_current(container)
    assert registry.get_current() is container
    assert registry.unregister_current(expected=container) is container
    assert registry.get_current() is None


@pytest.mark.asyncio
async def test_service_container_registry_rejects_different_container_on_same_loop():
    registry = ServiceContainerRegistry()
    container_a = ServiceContainer()
    container_b = ServiceContainer()

    registry.register_current(container_a)
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register_current(container_b)
    assert registry.unregister_current(expected=container_a) is container_a


@pytest.mark.asyncio
async def test_resolve_service_container_prefers_registry():
    current = ServiceContainer()
    app_state = SimpleNamespace()
    registry = get_or_create_service_container_registry(app_state)
    registry.register_current(current)
    try:
        assert resolve_service_container(app_state) is current
    finally:
        registry.unregister_current(expected=current)
