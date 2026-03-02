from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.params import Depends
from starlette.requests import Request

from fastapiex.di import Inject, Require, ServiceLifetime
from fastapiex.di.container import ServiceContainer
from fastapiex.di.injection import (
    ServiceContainerRegistry,
    _di_cleanup_scope_dependency,
    _di_transaction_scope_dependency,
    get_or_create_service_container_registry,
    resolve_service_container,
)
from fastapiex.di.exceptions import (
    ServiceContainerAccessError,
    ServiceFactoryContractError,
    UnregisteredServiceByKeyError,
    UnregisteredServiceByTypeError,
)


@asynccontextmanager
async def _build_request_with_scopes(
    container: ServiceContainer,
) -> AsyncIterator[tuple[Request, AsyncExitStack, AsyncExitStack]]:
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
    transaction_stack = AsyncExitStack()
    await transaction_stack.__aenter__()
    cleanup_stack = AsyncExitStack()
    await cleanup_stack.__aenter__()
    container._bind_transaction_scope_stack(request, transaction_stack)
    container._bind_cleanup_scope_stack(request, cleanup_stack)
    try:
        yield request, transaction_stack, cleanup_stack
    except BaseException as exc:
        container._clear_transaction_scope_stack(request, expected=transaction_stack)
        container._clear_cleanup_scope_stack(request, expected=cleanup_stack)
        suppressed = await transaction_stack.__aexit__(
            type(exc),
            exc,
            exc.__traceback__,
        )
        await cleanup_stack.__aexit__(None, None, None)
        if suppressed:
            return
        raise
    else:
        container._clear_transaction_scope_stack(request, expected=transaction_stack)
        container._clear_cleanup_scope_stack(request, expected=cleanup_stack)
        await transaction_stack.__aexit__(None, None, None)
        await cleanup_stack.__aexit__(None, None, None)


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
async def test_singleton_ignores_runtime_args_on_first_resolution(caplog) -> None:
    container = ServiceContainer()

    async def factory(value: int = 1) -> dict[str, int]:
        return {"value": value}

    await container.register("singleton_with_default", ServiceLifetime.SINGLETON, factory, None)

    with caplog.at_level(logging.WARNING):
        first = cast(
            dict[str, int],
            await container.aget_by_key("singleton_with_default", 99),
        )
        second = cast(
            dict[str, int],
            await container.aget_by_key("singleton_with_default", value=7),
        )

    assert first is second
    assert first["value"] == 1
    ignored_messages = [
        rec.message
        for rec in caplog.records
        if "Arguments given for singleton service 'singleton_with_default' are ignored." in rec.message
    ]
    assert len(ignored_messages) == 2



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


def test_inject_uses_cached_hidden_scope_dependencies() -> None:
    depends_marker = Inject("test")
    dependency = depends_marker.dependency
    assert dependency is not None

    signature = inspect.signature(dependency)
    scope_param = signature.parameters["_di_transaction_scope"]
    scope_dep = scope_param.default

    assert isinstance(scope_dep, Depends)
    assert scope_dep.dependency is _di_transaction_scope_dependency
    assert scope_dep.use_cache is True
    assert scope_dep.scope == "function"

    transaction_signature = inspect.signature(_di_transaction_scope_dependency)
    cleanup_param = transaction_signature.parameters["_di_cleanup_scope"]
    cleanup_dep = cleanup_param.default

    assert isinstance(cleanup_dep, Depends)
    assert cleanup_dep.dependency is _di_cleanup_scope_dependency
    assert cleanup_dep.use_cache is True
    assert cleanup_dep.scope == "request"


@pytest.mark.asyncio
async def test_request_scope_stacks_are_isolated_per_container() -> None:
    container_a = ServiceContainer()
    container_b = ServiceContainer()
    async with _build_request_with_scopes(container_a) as (request, tx_a, cleanup_a):
        tx_b = AsyncExitStack()
        cleanup_b = AsyncExitStack()
        await tx_b.__aenter__()
        await cleanup_b.__aenter__()
        container_b._bind_transaction_scope_stack(request, tx_b)
        container_b._bind_cleanup_scope_stack(request, cleanup_b)

        try:
            assert container_a._resolve_transaction_scope_stack(request) is tx_a
            assert container_a._resolve_cleanup_scope_stack(request) is cleanup_a
            assert container_b._resolve_transaction_scope_stack(request) is tx_b
            assert container_b._resolve_cleanup_scope_stack(request) is cleanup_b

            container_a._clear_transaction_scope_stack(request, expected=tx_a)
            container_a._clear_cleanup_scope_stack(request, expected=cleanup_a)

            assert container_a._resolve_transaction_scope_stack(request) is None
            assert container_a._resolve_cleanup_scope_stack(request) is None
            assert container_b._resolve_transaction_scope_stack(request) is tx_b
            assert container_b._resolve_cleanup_scope_stack(request) is cleanup_b

            container_b._clear_transaction_scope_stack(request, expected=tx_b)
            container_b._clear_cleanup_scope_stack(request, expected=cleanup_b)
            assert container_b._resolve_transaction_scope_stack(request) is None
            assert container_b._resolve_cleanup_scope_stack(request) is None
        finally:
            await tx_b.__aexit__(None, None, None)
            await cleanup_b.__aexit__(None, None, None)


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

    async with _build_request_with_scopes(container) as (request, _, _):
        instance = cast(
            dict[str, str],
            await container.aget_by_key("gen", **{container.request_kwarg_name(): request}),
        )
        assert instance["data"] == "test"

    assert cleanup_called is True


@pytest.mark.asyncio
async def test_async_generator_factory_does_not_use_to_thread(monkeypatch) -> None:
    container = ServiceContainer()
    to_thread_calls = 0
    original_to_thread = asyncio.to_thread

    async def tracked_to_thread(func, /, *args, **kwargs):
        nonlocal to_thread_calls
        to_thread_calls += 1
        return await original_to_thread(func, *args, **kwargs)

    async def factory() -> AsyncIterator[dict[str, bool]]:
        yield {"ok": True}

    await container.register("gen", ServiceLifetime.TRANSIENT, factory, None)
    monkeypatch.setattr(asyncio, "to_thread", tracked_to_thread)

    async with _build_request_with_scopes(container) as (request, _, _):
        instance = await container.aget_by_key(
            "gen", **{container.request_kwarg_name(): request}
        )
        assert instance == {"ok": True}

    assert to_thread_calls == 0


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
async def test_register_rejects_sync_destructor():
    container = ServiceContainer()

    async def factory() -> dict[str, str]:
        return {"ok": "1"}

    def sync_destructor(instance: object) -> None:
        _ = instance

    with pytest.raises(TypeError, match="must be defined as async def"):
        await container.register(
            "bad_sync_dtor",
            ServiceLifetime.SINGLETON,
            factory,
            cast(Any, sync_destructor),
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
async def test_unregistered_resolution_raises_domain_specific_errors() -> None:
    container = ServiceContainer()

    class MissingType:
        pass

    with pytest.raises(UnregisteredServiceByKeyError):
        await container.aget_by_key("missing_key")

    with pytest.raises(UnregisteredServiceByTypeError):
        await container.aget_by_type(MissingType)


@pytest.mark.asyncio
async def test_inject_passthrough_does_not_resolve_require_default_for_singleton() -> None:
    container = ServiceContainer()

    class Consumer:
        def __init__(self, repo: object) -> None:
            self.repo = repo

    async def consumer_factory(repo=Require("repo")) -> Consumer:
        return Consumer(repo)

    await container.register("consumer", ServiceLifetime.SINGLETON, consumer_factory, None)

    app = FastAPI()
    registry = get_or_create_service_container_registry(app.state)
    registry.register_current(container)
    try:
        request = Request(
            {
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
                "app": app,
            }
        )

        depends_marker = Inject("consumer")
        dependency = depends_marker.dependency
        assert dependency is not None
        consumer = await dependency(request)
        from fastapiex.di.registry import RequiredService

        assert isinstance(consumer, Consumer)
        assert isinstance(consumer.repo, RequiredService)
        assert consumer.repo.target == "repo"
    finally:
        registry.unregister_current(expected=container)


@pytest.mark.asyncio
async def test_aget_passthrough_does_not_resolve_require_defaults() -> None:
    container = ServiceContainer()

    class Consumer:
        def __init__(self, repo: object) -> None:
            self.repo = repo

    async def consumer_factory(repo=Require("repo")) -> Consumer:
        return Consumer(repo)

    await container.register("consumer", ServiceLifetime.SINGLETON, consumer_factory, None)

    consumer = await container.aget_by_key("consumer")
    from fastapiex.di.registry import RequiredService

    assert isinstance(consumer, Consumer)
    assert isinstance(consumer.repo, RequiredService)


@pytest.mark.asyncio
async def test_inject_passthrough_does_not_resolve_positional_only_require_default() -> None:
    container = ServiceContainer()

    class Consumer:
        def __init__(self, repo: object) -> None:
            self.repo = repo

    async def consumer_factory(repo=Require("repo"), /) -> Consumer:
        return Consumer(repo)

    await container.register("consumer", ServiceLifetime.SINGLETON, consumer_factory, None)

    app = FastAPI()
    registry = get_or_create_service_container_registry(app.state)
    registry.register_current(container)
    try:
        request = Request(
            {
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
                "app": app,
            }
        )

        depends_marker = Inject("consumer")
        dependency = depends_marker.dependency
        assert dependency is not None
        consumer = await dependency(request)
        from fastapiex.di.registry import RequiredService

        assert isinstance(consumer, Consumer)
        assert isinstance(consumer.repo, RequiredService)
        assert consumer.repo.target == "repo"
    finally:
        registry.unregister_current(expected=container)


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

    async with _build_request_with_scopes(container) as (request, _, _):
        instance = await container.aget_by_key(
            "gen", **{container.request_kwarg_name(): request}
        )
        assert instance == {"ok": True}

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

    async with _build_request_with_scopes(container) as (
        request,
        transaction_stack,
        _,
    ):
        instance = await container.aget_by_key(
            "gen", **{container.request_kwarg_name(): request}
        )
        assert instance == {"ok": True}

        err = RuntimeError("boom")
        suppressed = await transaction_stack.__aexit__(
            RuntimeError,
            err,
            err.__traceback__,
        )
        assert suppressed is False

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

    async with _build_request_with_scopes(container) as (
        request,
        transaction_stack,
        _,
    ):
        instance = await container.aget_by_key(
            "gen", **{container.request_kwarg_name(): request}
        )
        assert instance == 1

        with pytest.raises(ServiceFactoryContractError, match="must yield exactly once"):
            await transaction_stack.__aexit__(None, None, None)

    assert marks == ["finally"]


@pytest.mark.asyncio
async def test_async_generator_transient_requires_di_transaction_scope():
    container = ServiceContainer()

    async def factory() -> AsyncIterator[dict[str, bool]]:
        yield {"ok": True}

    await container.register("gen", ServiceLifetime.TRANSIENT, factory, None)

    with pytest.raises(ServiceContainerAccessError, match="active DI transaction scope"):
        await container.aget_by_key("gen")


@pytest.mark.asyncio
async def test_transient_plain_dtor_requires_di_cleanup_scope():
    container = ServiceContainer()

    async def factory() -> dict[str, bool]:
        return {"ok": True}

    async def dtor(instance: object) -> None:
        _ = instance

    await container.register("plain", ServiceLifetime.TRANSIENT, factory, dtor)

    with pytest.raises(ServiceContainerAccessError, match="active DI cleanup scope"):
        await container.aget_by_key("plain")


@pytest.mark.asyncio
async def test_transient_plain_dtor_logs_and_does_not_escape(caplog) -> None:
    container = ServiceContainer()

    async def factory() -> dict[str, bool]:
        return {"ok": True}

    async def dtor(instance: object) -> None:
        _ = instance
        raise RuntimeError("dtor boom")

    await container.register("plain", ServiceLifetime.TRANSIENT, factory, dtor)

    with caplog.at_level(logging.ERROR):
        async with _build_request_with_scopes(container) as (request, _, _):
            instance = await container.aget_by_key(
                "plain", **{container.request_kwarg_name(): request}
            )
            assert instance == {"ok": True}

    assert "Error running transient destructor for service 'plain'." in caplog.text


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

    async with _build_request_with_scopes(container) as (
        request,
        transaction_stack,
        _,
    ):
        instance = await container.aget_by_key(
            "gen", **{container.request_kwarg_name(): request}
        )
        assert instance == 1

        with pytest.raises(ServiceFactoryContractError, match="must yield exactly once"):
            await transaction_stack.__aexit__(None, None, None)

    assert marks == ["finally"]


@pytest.mark.asyncio
async def test_generator_transient_finalizer_uses_throw_on_cancelled_request():
    """Cancelled requests should drive sync generators through BaseException rollback path."""
    container = ServiceContainer()
    marks: list[str] = []

    def factory():
        try:
            yield {"ok": True}
        except Exception:
            marks.append("except")
            raise
        except BaseException as exc:
            marks.append(f"base:{type(exc).__name__}")
            raise
        else:
            marks.append("else")
        finally:
            marks.append("finally")

    await container.register("gen", ServiceLifetime.TRANSIENT, factory, None)

    async with _build_request_with_scopes(container) as (
        request,
        transaction_stack,
        _,
    ):
        instance = await container.aget_by_key(
            "gen", **{container.request_kwarg_name(): request}
        )
        assert instance == {"ok": True}

        err = asyncio.CancelledError()
        suppressed = await transaction_stack.__aexit__(
            asyncio.CancelledError,
            err,
            err.__traceback__,
        )
        assert suppressed is False

    assert marks == ["base:CancelledError", "finally"]


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
