from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import threading
import types
import uuid
from pathlib import Path
from typing import cast

import anyio
import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from fastapiex.di import Inject, ServiceLifetime, install_di
from fastapiex.di.injection import resolve_service_container
from tests.di_test_services.helpers import set_order_sink


def _import_or_reload(module_name: str) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        importlib.import_module(module_name)
        return
    importlib.reload(module)


def _register_dynamic_private_service(
    *,
    module_name: str,
    service_key: str,
    payload: str,
) -> None:
    module = types.ModuleType(module_name)
    module.__file__ = f"/Users/henri/Code/fastapiex-di/tests/di_test_services/{module_name.rsplit('.', 1)[-1]}.py"
    module.__package__ = "tests.di_test_services"
    sys.modules[module_name] = module
    exec(
        (
            "from fastapiex.di import BaseService, Service\n"
            f"@Service({service_key!r})\n"
            "class DynamicPrivateService(BaseService):\n"
            "    @classmethod\n"
            "    async def create(cls) -> dict[str, str]:\n"
            f"        return {{'scope': {payload!r}}}\n"
        ),
        module.__dict__,
    )


def test_install_di_wires_injection_and_allows_runtime_registration() -> None:
    app = FastAPI()
    install_di(
        app,
        service_packages=["tests.di_test_services"],
    )

    @app.get("/ping")
    async def ping(_svc=Inject("ping_service")):
        return {"ok": True}

    @app.post("/register")
    async def register_runtime_service():
        services = resolve_service_container(app.state)
        assert services is not None

        async def runtime_factory() -> dict[str, bool]:
            return {"ok": True}

        try:
            await services.register(
                "runtime_service",
                ServiceLifetime.SINGLETON,
                runtime_factory,
                None,
            )
        except RuntimeError as exc:
            return {"error": str(exc)}

        return {"ok": True}

    with TestClient(app) as client:
        ping_response = client.get("/ping")
        assert ping_response.status_code == 200
        assert ping_response.json() == {"ok": True}

        register_response = client.post("/register")
        assert register_response.status_code == 200
        assert register_response.json() == {"ok": True}


def test_install_di_inject_does_not_retry_on_non_registration_runtime_error() -> None:
    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])
    call_count = 0

    @app.post("/register-boom")
    async def register_runtime_boom_service() -> dict[str, bool]:
        nonlocal call_count
        services = resolve_service_container(app.state)
        assert services is not None

        async def boom_factory() -> dict[str, bool]:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom from ctor")

        await services.register(
            "boom_service",
            ServiceLifetime.TRANSIENT,
            boom_factory,
            None,
        )
        return {"ok": True}

    @app.get("/boom")
    async def boom(_svc=Inject("boom_service")):
        return {"ok": True}

    with TestClient(app) as client:
        register_response = client.post("/register-boom")
        assert register_response.status_code == 200
        assert register_response.json() == {"ok": True}

        with pytest.raises(RuntimeError, match="boom from ctor"):
            client.get("/boom")

    assert call_count == 1


def test_install_di_stores_service_packages_as_global_paths() -> None:
    app = FastAPI()
    config = install_di(app, service_packages=["tests.di_test_services"])

    assert config.service_packages
    for path in config.service_packages:
        resolved = Path(path)
        assert resolved.is_absolute()
        assert resolved.exists()


def test_install_di_transaction_scope_runs_before_background_tasks_and_dtor_runs_after() -> None:
    order: list[str] = []
    set_order_sink(order)

    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    @app.get("/order")
    async def order_endpoint(
        background_tasks: BackgroundTasks,
        _svc=Inject("tracked_transactional_transient_with_dtor"),
    ):
        background_tasks.add_task(order.append, "background")
        order.append("handler")
        return {"ok": True}

    try:
        with TestClient(app) as client:
            response = client.get("/order")
            assert response.status_code == 200
        assert order == [
            "handler",
            "commit",
            "generator-finally",
            "background",
            "dtor",
        ]
    finally:
        set_order_sink(None)


def test_install_di_generator_transient_rolls_back_on_httpexception() -> None:
    order: list[str] = []
    set_order_sink(order)

    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    @app.get("/tx-http")
    async def tx_http(_svc=Inject("tracked_transactional_transient")):
        raise HTTPException(status_code=400, detail="bad")

    try:
        with TestClient(app) as client:
            response = client.get("/tx-http")
            assert response.status_code == 400
            assert response.json() == {"detail": "bad"}
        assert order == ["rollback", "finalizer"]
    finally:
        set_order_sink(None)


def test_install_di_generator_transient_rolls_back_with_exception_handler() -> None:
    order: list[str] = []
    set_order_sink(order)

    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
        _ = request, exc
        return JSONResponse(status_code=500, content={"handled": True})

    @app.get("/tx-key")
    async def tx_key(_svc=Inject("tracked_transactional_transient")):
        raise KeyError("boom")

    try:
        with TestClient(app) as client:
            response = client.get("/tx-key")
            assert response.status_code == 500
            assert response.json() == {"handled": True}
        assert order == ["rollback", "finalizer"]
    finally:
        set_order_sink(None)


def test_install_di_generator_transient_commits_on_raw_400_response() -> None:
    order: list[str] = []
    set_order_sink(order)

    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    @app.get("/tx-raw-400")
    async def tx_raw_400(_svc=Inject("tracked_transactional_transient")):
        return JSONResponse(status_code=400, content={"ok": False})

    try:
        with TestClient(app) as client:
            response = client.get("/tx-raw-400")
            assert response.status_code == 400
            assert response.json() == {"ok": False}
        assert order == ["commit", "finalizer"]
    finally:
        set_order_sink(None)


def test_install_di_generator_exit_runs_before_dtor() -> None:
    order: list[str] = []
    set_order_sink(order)

    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    @app.get("/tx-dtor")
    async def tx_dtor(_svc=Inject("tracked_transactional_transient_with_dtor")):
        return {"ok": True}

    try:
        with TestClient(app) as client:
            response = client.get("/tx-dtor")
            assert response.status_code == 200
            assert response.json() == {"ok": True}
        assert order == ["commit", "generator-finally", "dtor"]
    finally:
        set_order_sink(None)


def test_install_di_nested_require_generator_transient_commits_on_success() -> None:
    order: list[str] = []
    set_order_sink(order)

    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    @app.get("/tx-nested-ok")
    async def tx_nested_ok(_svc=Inject("tracked_nested_transactional_consumer")):
        return {"ok": True}

    try:
        with TestClient(app) as client:
            response = client.get("/tx-nested-ok")
            assert response.status_code == 200
            assert response.json() == {"ok": True}
        assert order == ["nested-commit", "nested-finalizer"]
    finally:
        set_order_sink(None)


def test_install_di_nested_require_generator_transient_rolls_back_on_httpexception() -> None:
    order: list[str] = []
    set_order_sink(order)

    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    @app.get("/tx-nested-http")
    async def tx_nested_http(_svc=Inject("tracked_nested_transactional_consumer")):
        raise HTTPException(status_code=400, detail="bad")

    try:
        with TestClient(app) as client:
            response = client.get("/tx-nested-http")
            assert response.status_code == 400
            assert response.json() == {"detail": "bad"}
        assert order == ["nested-rollback", "nested-finalizer"]
    finally:
        set_order_sink(None)


def test_install_di_websocket_finalizer_runs_on_disconnect() -> None:
    order: list[str] = []
    set_order_sink(order)

    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    @app.websocket("/ws")
    async def ws_endpoint(
        websocket: WebSocket,
        _svc=Inject("tracked_transient"),
    ) -> None:
        await websocket.accept()
        await websocket.send_text("ok")
        await websocket.close()

    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                assert ws.receive_text() == "ok"
        assert order == ["finalizer"]
    finally:
        set_order_sink(None)


def test_install_di_websocket_finalizer_runs_on_server_exception_disconnect() -> None:
    order: list[str] = []
    set_order_sink(order)

    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    @app.websocket("/ws-error")
    async def ws_endpoint(
        websocket: WebSocket,
        _svc=Inject("tracked_transient"),
    ) -> None:
        await websocket.accept()
        raise RuntimeError("ws boom")

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            try:
                with client.websocket_connect("/ws-error") as ws:
                    ws.receive_text()
            except (RuntimeError, WebSocketDisconnect, anyio.ClosedResourceError):
                pass
        assert order == ["finalizer"]
    finally:
        set_order_sink(None)


def test_install_di_websocket_finalizer_runs_on_websocketdisconnect() -> None:
    order: list[str] = []
    set_order_sink(order)

    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    @app.websocket("/ws-disconnect")
    async def ws_endpoint(
        websocket: WebSocket,
        _svc=Inject("tracked_transient"),
    ) -> None:
        await websocket.accept()
        await websocket.send_text("ready")
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            return

    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws-disconnect") as ws:
                assert ws.receive_text() == "ready"
        assert order == ["finalizer"]
    finally:
        set_order_sink(None)


def test_install_di_websocket_long_connection_multi_inject_cleanup_order() -> None:
    order: list[str] = []
    set_order_sink(order)

    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    @app.websocket("/ws-long")
    async def ws_endpoint(
        websocket: WebSocket,
        _scope=Inject("ping_service"),
    ) -> None:
        _ = _scope
        await websocket.accept()
        services = resolve_service_container(websocket.app.state)
        assert services is not None
        request_kwarg_name = services.request_kwarg_name()

        try:
            while True:
                command = await websocket.receive_text()
                if command != "inject":
                    await websocket.close()
                    return

                payload = await services.aget_by_key(
                    "tracked_transient_ordered",
                    **{request_kwarg_name: websocket},
                )
                payload = cast(dict[str, int], payload)
                seq = payload["seq"]
                await websocket.send_text(f"ok:{seq}")
        except WebSocketDisconnect:
            return

    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws-long") as ws:
                ws.send_text("inject")
                assert ws.receive_text() == "ok:1"
                ws.send_text("inject")
                assert ws.receive_text() == "ok:2"
                ws.send_text("inject")
                assert ws.receive_text() == "ok:3"
                ws.send_text("close")

        assert order == [
            "finalizer:3",
            "finalizer:2",
            "finalizer:1",
        ]
    finally:
        set_order_sink(None)


def test_install_di_strict_false_allows_startup_without_di() -> None:
    app = FastAPI()
    install_di(
        app,
        service_packages=["nonexistent_pkg_for_di_tests"],
        strict=False,
    )

    @app.get("/health")
    async def health():
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"ok": True}


def test_install_di_strict_true_fails_on_missing_service_package() -> None:
    app = FastAPI()
    install_di(
        app,
        service_packages=["nonexistent_pkg_for_di_tests"],
        strict=True,
    )

    with pytest.raises(ModuleNotFoundError):
        with TestClient(app):
            pass


def test_install_di_exposes_registry_on_startup() -> None:
    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    assert app.state.di_service_registry is None
    with TestClient(app):
        scoped_registry = app.state.di_service_registry
        assert scoped_registry is not None
    assert app.state.di_service_registry is not None


def test_install_di_uses_distinct_app_scoped_registries() -> None:
    app_a = FastAPI()
    app_b = FastAPI()
    install_di(app_a, service_packages=["tests.di_test_services"])
    install_di(app_b, service_packages=["tests.di_test_services"])

    with TestClient(app_a):
        registry_a = app_a.state.di_service_registry
        assert registry_a is not None
        with TestClient(app_b):
            registry_b = app_b.state.di_service_registry
            assert registry_b is not None
            assert registry_a is not registry_b


def test_install_di_reregisters_services_when_modules_are_cached() -> None:
    app_first = FastAPI()
    install_di(app_first, service_packages=["tests.di_test_services"])

    @app_first.get("/resolve")
    async def resolve_first(_svc=Inject("tracked_transient")):
        return {"ok": True}

    with TestClient(app_first) as client:
        first = client.get("/resolve")
        assert first.status_code == 200

    app_second = FastAPI()
    install_di(app_second, service_packages=["tests.di_test_services"])

    @app_second.get("/resolve")
    async def resolve_second(_svc=Inject("tracked_transient")):
        return {"ok": True}

    with TestClient(app_second) as client:
        second = client.get("/resolve")
        assert second.status_code == 200


def test_install_di_global_registry_true_includes_services_outside_service_packages() -> None:
    app = FastAPI()
    install_di(
        app,
        service_packages=["tests.di_test_services"],
        use_global_service_registry=True,
    )

    @app.get("/global")
    async def resolve_global(_svc=Inject("external_global_service")):
        return {"ok": True}

    with TestClient(app) as client:
        import tests.di_global_services.main  # noqa: F401

        response = client.get("/global")
        assert response.status_code == 200
        assert response.json() == {"ok": True}


def test_install_di_global_registry_false_ignores_services_outside_service_packages() -> None:
    app = FastAPI()
    install_di(
        app,
        service_packages=["tests.di_test_services"],
        use_global_service_registry=False,
    )

    @app.get("/global")
    async def resolve_global(_svc=Inject("external_global_service")):
        return {"ok": True}

    with TestClient(app) as client:
        import tests.di_global_services.main  # noqa: F401

        with pytest.raises(
            RuntimeError,
            match="Requesting unregistered service: external_global_service",
        ):
            client.get("/global")


def test_install_di_refresh_registers_late_private_service_without_global_registry() -> None:
    app = FastAPI()
    install_di(
        app,
        service_packages=["tests.di_test_services"],
        use_global_service_registry=False,
    )

    @app.get("/late-private")
    async def resolve_late_private(_svc=Inject("late_private_service")):
        return {"ok": True}

    with TestClient(app) as client:
        _import_or_reload("tests.di_test_services._late_private")

        response = client.get("/late-private")
        assert response.status_code == 200
        assert response.json() == {"ok": True}


def test_install_di_late_refresh_does_not_emit_error_log(caplog: pytest.LogCaptureFixture) -> None:
    app = FastAPI()
    install_di(
        app,
        service_packages=["tests.di_test_services"],
        use_global_service_registry=False,
    )

    @app.get("/late-private-log")
    async def resolve_late_private(_svc=Inject("late_private_service")):
        return {"ok": True}

    with caplog.at_level(logging.ERROR, logger="fastapiex.di.container"):
        with TestClient(app) as client:
            _import_or_reload("tests.di_test_services._late_private")

            response = client.get("/late-private-log")
            assert response.status_code == 200
            assert response.json() == {"ok": True}

    assert "Requesting unregistered service: late_private_service" not in caplog.text


def test_install_di_shared_app_worker_contexts_refresh_independently() -> None:
    app = FastAPI()
    install_di(
        app,
        service_packages=["tests.di_test_services"],
        use_global_service_registry=False,
    )

    dependency = Inject("late_private_service").dependency
    assert dependency is not None

    ready = threading.Barrier(3)
    first_done = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()
    results: dict[str, dict[str, str]] = {}
    errors: dict[str, BaseException] = {}

    def _build_request() -> Request:
        return Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "path": "/late-private",
                "raw_path": b"/late-private",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 12345),
                "server": ("testserver", 80),
                "scheme": "http",
                "state": {},
                "app": app,
            }
        )

    def _worker(name: str, gate: threading.Event) -> None:
        async def _run() -> None:
            async with app.router.lifespan_context(app):
                await asyncio.to_thread(ready.wait)
                await asyncio.to_thread(gate.wait)
                result = await dependency(_build_request())
                results[name] = cast(dict[str, str], result)
                if name == "first":
                    first_done.set()

        try:
            asyncio.run(_run())
        except BaseException as exc:  # pragma: no cover - exercised on regression
            errors[name] = exc
            if name == "first":
                first_done.set()

    first = threading.Thread(target=_worker, args=("first", release_first))
    second = threading.Thread(target=_worker, args=("second", release_second))
    first.start()
    second.start()

    ready.wait()
    _import_or_reload("tests.di_test_services._late_private")

    release_first.set()
    assert first_done.wait(timeout=5)
    release_second.set()

    first.join(timeout=5)
    second.join(timeout=5)

    assert errors == {}
    assert results == {
        "first": {"scope": "private"},
        "second": {"scope": "private"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("use_global_service_registry", [False, True])
async def test_install_di_later_worker_startup_reuses_explicitly_imported_private_service(
    use_global_service_registry: bool,
) -> None:
    suffix = uuid.uuid4().hex
    module_name = f"tests.di_test_services._dynamic_private_{suffix}"
    service_key = f"dynamic_private_service_{suffix}"

    app = FastAPI()
    install_di(
        app,
        service_packages=["tests.di_test_services"],
        use_global_service_registry=use_global_service_registry,
        allow_private_modules=False,
    )

    dependency = Inject(service_key).dependency
    assert dependency is not None

    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "path": "/dynamic-private",
            "raw_path": b"/dynamic-private",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "scheme": "http",
            "state": {},
            "app": app,
        }
    )

    async with app.router.lifespan_context(app):
        _register_dynamic_private_service(
            module_name=module_name,
            service_key=service_key,
            payload="dynamic-private",
        )
        payload = await dependency(request)
        assert payload == {"scope": "dynamic-private"}

    async with app.router.lifespan_context(app):
        payload = await dependency(request)
        assert payload == {"scope": "dynamic-private"}


def test_install_di_global_registry_true_refreshes_after_startup_late_import() -> None:
    app = FastAPI()
    install_di(
        app,
        service_packages=["tests.di_test_services"],
        use_global_service_registry=True,
    )

    @app.get("/late-global")
    async def resolve_late_global(_svc=Inject("external_late_global_service")):
        return {"ok": True}

    with TestClient(app) as client:
        import tests.di_global_services.late  # noqa: F401

        response = client.get("/late-global")
        assert response.status_code == 200
        assert response.json() == {"ok": True}


def test_install_di_global_registry_false_does_not_refresh_after_startup_late_import() -> None:
    app = FastAPI()
    install_di(
        app,
        service_packages=["tests.di_test_services"],
        use_global_service_registry=False,
    )

    @app.get("/late-global")
    async def resolve_late_global(_svc=Inject("external_late_global_service")):
        return {"ok": True}

    with TestClient(app) as client:
        import tests.di_global_services.late  # noqa: F401

        with pytest.raises(
            RuntimeError,
            match="Requesting unregistered service: external_late_global_service",
        ):
            client.get("/late-global")


def test_install_di_rejects_double_install() -> None:
    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    with pytest.raises(RuntimeError, match="already been called"):
        install_di(app, service_packages=["tests.di_test_services"])
