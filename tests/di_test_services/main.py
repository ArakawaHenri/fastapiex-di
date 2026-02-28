from __future__ import annotations

from typing import Any

from fastapiex.di import BaseService, Require, Service
from tests.di_test_services.helpers import append_order, next_order_seq


@Service("ping_service", eager=True)
class PingService(BaseService):
    @classmethod
    async def create(cls) -> "PingService":
        return cls()


@Service("tracked_transient", lifetime="transient")
class TrackedTransientServiceT(BaseService):
    @classmethod
    async def create(cls) -> Any:
        try:
            yield {"ok": True}
        finally:
            append_order("finalizer")


@Service("tracked_transient_ordered", lifetime="transient")
class TrackedOrderedTransientServiceT(BaseService):
    @classmethod
    async def create(cls) -> Any:
        current_seq = next_order_seq()
        try:
            yield {"seq": current_seq}
        finally:
            append_order(f"finalizer:{current_seq}")


@Service("tracked_transactional_transient", lifetime="transient")
class TrackedTransactionalTransientServiceT(BaseService):
    @classmethod
    async def create(cls) -> Any:
        try:
            yield {"ok": True}
        except Exception:
            append_order("rollback")
            raise
        else:
            append_order("commit")
        finally:
            append_order("finalizer")


@Service("tracked_transactional_transient_with_dtor", lifetime="transient")
class TrackedTransactionalTransientWithDtorServiceT(BaseService):
    @classmethod
    async def create(cls) -> Any:
        try:
            yield {"ok": True}
        except Exception:
            append_order("rollback")
            raise
        else:
            append_order("commit")
        finally:
            append_order("generator-finally")

    @classmethod
    async def destroy(cls, instance: object) -> None:
        _ = cls, instance
        append_order("dtor")


@Service("tracked_nested_transactional_dep", lifetime="transient")
class TrackedNestedTransactionalDepServiceT(BaseService):
    @classmethod
    async def create(cls) -> Any:
        try:
            yield {"ok": True}
        except Exception:
            append_order("nested-rollback")
            raise
        else:
            append_order("nested-commit")
        finally:
            append_order("nested-finalizer")


@Service("tracked_nested_transactional_consumer", lifetime="transient")
class TrackedNestedTransactionalConsumerServiceT(BaseService):
    @classmethod
    async def create(cls, dep=Require("tracked_nested_transactional_dep")) -> Any:
        _ = cls
        return {"dep": dep}
