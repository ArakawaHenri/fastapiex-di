from __future__ import annotations

from typing import cast

import anyio
import pytest
from fastapi import BackgroundTasks, FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from fastapiex.di import Inject, ServiceLifetime, install_di, resolve_service_container
from tests.di_test_services.helpers import set_order_sink


def test_install_di_wires_injection_and_freezes_runtime_registration() -> None:
    app = FastAPI()
    install_di(
        app,
        service_packages=["tests.di_test_services"],
        freeze_container_after_startup=True,
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
        error = register_response.json().get("error", "")
        assert "frozen" in error.lower()


def test_install_di_finalizer_runs_after_background_tasks() -> None:
    order: list[str] = []
    set_order_sink(order)

    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    @app.get("/order")
    async def order_endpoint(
        background_tasks: BackgroundTasks,
        _svc=Inject("tracked_transient"),
    ):
        background_tasks.add_task(order.append, "background")
        order.append("handler")
        return {"ok": True}

    try:
        with TestClient(app) as client:
            response = client.get("/order")
            assert response.status_code == 200
        assert order == ["handler", "background", "finalizer"]
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
    async def ws_endpoint(websocket: WebSocket) -> None:
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


def test_install_di_can_freeze_and_unfreeze_registry() -> None:
    app = FastAPI()
    install_di(
        app,
        service_packages=["tests.di_test_services"],
        freeze_service_registry_after_startup=True,
        unfreeze_service_registry_on_shutdown=True,
    )

    assert app.state.di_service_registry is None
    with TestClient(app):
        scoped_registry = app.state.di_service_registry
        assert scoped_registry is not None
        assert scoped_registry.is_frozen() is True
    assert app.state.di_service_registry is not None
    assert app.state.di_service_registry.is_frozen() is False


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


def test_install_di_rejects_double_install() -> None:
    app = FastAPI()
    install_di(app, service_packages=["tests.di_test_services"])

    with pytest.raises(RuntimeError, match="already been called"):
        install_di(app, service_packages=["tests.di_test_services"])
