from __future__ import annotations

from typing import Any

from fastapiex.di import BaseService, Service
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
