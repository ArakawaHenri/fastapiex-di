from __future__ import annotations

import gc
import logging
import weakref
from contextlib import AsyncExitStack
from types import SimpleNamespace

import pytest

from fastapiex.di import Inject
from fastapiex.di.container import ServiceContainer
from fastapiex.di.exceptions import InvalidServiceDefinitionError
from fastapiex.di.registry import (
    AppServiceRegistry,
    _runtime_registry_bindings_snapshot,
    register_runtime_registry_binding,
)


def test_inject_requires_explicit_target() -> None:
    with pytest.raises(TypeError):
        Inject()


def test_inject_rejects_none_target() -> None:
    with pytest.raises(InvalidServiceDefinitionError, match="expects either"):
        Inject(None)


def test_runtime_registry_bindings_do_not_keep_registry_alive() -> None:
    def _register_binding() -> weakref.ReferenceType[AppServiceRegistry]:
        registry = AppServiceRegistry()
        register_runtime_registry_binding(
            registry,
            package_paths=(),
            use_global_service_registry=False,
        )
        return weakref.ref(registry)

    registry_ref = _register_binding()
    gc.collect()

    assert registry_ref() is None
    bindings, maintain_global = _runtime_registry_bindings_snapshot()
    assert bindings == ()
    assert maintain_global is False


def test_scope_stack_mismatch_logs_warning_and_keeps_binding(caplog: pytest.LogCaptureFixture) -> None:
    container = ServiceContainer()
    request = SimpleNamespace(state=SimpleNamespace())
    current = AsyncExitStack()
    expected = AsyncExitStack()

    container._bind_cleanup_scope_stack(request, current)
    container._bind_transaction_scope_stack(request, current)

    with caplog.at_level(logging.WARNING):
        container._clear_cleanup_scope_stack(request, expected=expected)
        container._clear_transaction_scope_stack(request, expected=expected)

    assert container._resolve_cleanup_scope_stack(request) is current
    assert container._resolve_transaction_scope_stack(request) is current
    assert "Cleanup scope stack mismatch on clear" in caplog.text
    assert "Transaction scope stack mismatch on clear" in caplog.text
